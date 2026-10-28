import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .episode import rewind_checked
from .game import ensure_game
from .reference_line import lateral_distance, progress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THROTTLE = ((False, False), (True, False), (False, True))
STEER = ((True, False), (False, False), (False, True))
ACTIONS = tuple(
    dict(left=s[0], right=s[1], accelerate=t[0], brake=t[1])
    for t in THROTTLE
    for s in STEER
)
THROTTLE_NAMES = ("coast", "gas", "brake")
STEER_NAMES = ("left", "straight", "right")

VEL_SCALE = 100.0
SPEED_SCALE = 500.0
POS_SCALE = 50.0
LOCAL_FROM_WORLD = True


def rotation_of(raw):
    return np.asarray(raw.rotation_matrix, dtype=np.float64).reshape(3, 3)


def to_local(R, v, local_from_world=LOCAL_FROM_WORLD):
    v = np.asarray(v, dtype=np.float64)
    return R.T @ v if local_from_world else R @ v


class TMNFEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, port=8478, map_path=None, vcp_path="data/vcp/A01.npy", action_ms=50,
                 speed=20.0, img_size=64, capture_size=128, lookahead=(1, 2, 3, 5, 8, 12, 16),
                 max_time_ms=60000, stuck_speed_kmh=5.0, stuck_ms=2000, grace_ms=1500,
                 max_lateral_m=30.0, step_penalty=0.01, finish_bonus=10.0, fail_penalty=1.0,
                 render_mode=None, client=None):
        super().__init__()
        self.port = port
        self.map_path = map_path
        self.action_ms = action_ms
        self.speed = speed
        self.img_size = img_size
        self.capture_size = capture_size
        self.lookahead = tuple(lookahead)
        self.max_time_ms = max_time_ms
        self.stuck_speed_kmh = stuck_speed_kmh
        self.stuck_ms = stuck_ms
        self.grace_ms = grace_ms
        self.max_lateral_m = max_lateral_m
        self.step_penalty = step_penalty
        self.finish_bonus = finish_bonus
        self.fail_penalty = fail_penalty
        self.render_mode = render_mode
        vp = vcp_path if os.path.isabs(vcp_path) else os.path.join(ROOT, vcp_path)
        self.vcp = np.load(vp).astype(np.float64)
        self.vcp_len_m = float(np.linalg.norm(np.diff(self.vcp, axis=0), axis=1).sum())
        self.seg_m = self.vcp_len_m / (len(self.vcp) - 1)
        self.client = client
        self.s0 = None
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
            "vector": spaces.Box(-100.0, 100.0, (4 + 3 * len(self.lookahead),), np.float32),
        })
        self.last_full_frame = None
        self.last_R = None
        self.total_env_steps = 0
        self.total_sim_ms = 0
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.last_idx = 0
        self.max_progress = 0.0
        self.stuck_t = 0
        self.t_start = 0
        self.t_now = 0
        self.cp_seen = 0
        self.ep_steps = 0

    def _connect(self):
        if self.client is None:
            self.client, self.s0 = ensure_game(self.port, self.map_path)
        else:
            self.s0 = self.client.get_state()
        self.cp_seen = self.client.cp_events

    def _frame(self):
        bgra = self.client.capture_frame(self.capture_size, self.capture_size)
        rgb = cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2RGB)
        self.last_full_frame = rgb
        return cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

    def _vector(self, st):
        R = rotation_of(st.raw)
        self.last_R = R
        pos = np.asarray(st.position, dtype=np.float64)
        vloc = to_local(R, st.velocity) / VEL_SCALE
        parts = [vloc, [st.speed_kmh / SPEED_SCALE]]
        for k in self.lookahead:
            j = min(self.last_idx + k, len(self.vcp) - 1)
            parts.append(to_local(R, self.vcp[j] - pos) / POS_SCALE)
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    def _obs(self, st):
        return {"image": self._frame(), "vector": self._vector(st)}

    def _info(self, st, lateral, reason, finished):
        return dict(
            race_time_ms=int(st.race_time),
            sim_time_s=(st.race_time - self.t_start) / 1000.0,
            progress=float(self.max_progress),
            progress_m=float(self.max_progress * self.seg_m),
            lateral_m=float(lateral),
            speed_kmh=float(st.speed_kmh),
            cp=self.client.last_cp,
            finished=finished,
            reason=reason,
            total_env_steps=self.total_env_steps,
            total_sim_ms=self.total_sim_ms,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.client is None or self.s0 is None:
            self._connect()
        rewind_checked(self.client, self.s0)
        self.client.set_input()
        self.client.set_speed(self.speed)
        self._reset_episode_state()
        self.cp_seen = self.client.cp_events
        st = self.client.get_state()
        self.t_start = st.race_time
        self.t_now = st.race_time
        pos = np.asarray(st.position)
        idx, p = progress(pos, self.vcp, 0)
        self.last_idx = idx
        self.max_progress = p
        lateral = lateral_distance(pos, self.vcp, idx)
        obs = self._obs(st)
        return obs, self._info(st, lateral, None, False)

    def step(self, action):
        c = self.client
        inp = ACTIONS[int(action)]
        c.set_input(**inp)
        t0 = self.t_now
        t = c.wait_step()
        while t < t0 + self.action_ms:
            t = c.wait_step()
        self.t_now = t
        self.total_env_steps += 1
        self.total_sim_ms += t - t0
        self.ep_steps += 1
        st = c.get_state()
        finished = False
        if c.cp_events != self.cp_seen:
            self.cp_seen = c.cp_events
            a, b = c.last_cp
            if a == b:
                finished = True
                c.prevent_simulation_finish()
        pos = np.asarray(st.position)
        idx, p = progress(pos, self.vcp, self.last_idx)
        self.last_idx = idx
        lateral = lateral_distance(pos, self.vcp, idx)
        gain = max(0.0, p - self.max_progress)
        self.max_progress = max(self.max_progress, p)
        reward = gain - self.step_penalty
        elapsed = st.race_time - self.t_start
        if elapsed > self.grace_ms and st.speed_kmh < self.stuck_speed_kmh:
            self.stuck_t += t - t0
        else:
            self.stuck_t = 0
        reason = None
        terminated = False
        if finished:
            terminated, reason = True, "finish"
            reward += self.finish_bonus
        elif self.stuck_t >= self.stuck_ms:
            terminated, reason = True, "stuck"
            reward -= self.fail_penalty
        elif lateral > self.max_lateral_m:
            terminated, reason = True, "off_track"
            reward -= self.fail_penalty
        truncated = (not terminated) and elapsed >= self.max_time_ms
        if truncated:
            reason = "timeout"
        obs = self._obs(st)
        return obs, float(reward), terminated, truncated, self._info(st, lateral, reason, finished)

    def render(self):
        return self.last_full_frame

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
