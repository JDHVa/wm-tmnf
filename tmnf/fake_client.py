from dataclasses import dataclass

import numpy as np


class _Raw:
    def __init__(self, R):
        self.rotation_matrix = R


@dataclass
class FakeState:
    race_time: int
    position: tuple
    velocity: tuple
    speed_kmh: float
    raw: object
    heading: float = 0.0
    speed: float = 0.0


class FakeClient:
    def __init__(self, vcp_path, seed=0):
        self.vcp = np.load(vcp_path).astype(np.float64)
        self.rng = np.random.default_rng(seed)
        self.cp_events = 0
        self.last_cp = (0, 1)
        self.speed_mult = 1.0
        self.inp = dict(left=False, right=False, accelerate=False, brake=False)
        d = self.vcp[1] - self.vcp[0]
        self.t = 0
        self.pos = self.vcp[0].copy()
        self.heading = float(np.arctan2(d[0], d[2]))
        self.v = 0.0
        self.finished = False

    def connect(self, timeout_s=60):
        pass

    def close(self):
        pass

    def set_input(self, left=False, right=False, accelerate=False, brake=False):
        self.inp = dict(left=left, right=right, accelerate=accelerate, brake=brake)

    def set_speed(self, speed):
        self.speed_mult = speed

    def prevent_simulation_finish(self):
        self.finished = False

    def wait_step(self):
        dt = 0.01
        i = self.inp
        if i["accelerate"]:
            self.v += 25.0 * dt
        if i["brake"]:
            self.v -= 40.0 * dt
        self.v = float(np.clip(self.v * (1 - 0.15 * dt), 0.0, 60.0))
        steer = (1 if i["right"] else 0) - (1 if i["left"] else 0)
        self.heading += steer * 0.9 * dt * min(self.v / 10.0, 1.0)
        fwd = np.array([np.sin(self.heading), 0.0, np.cos(self.heading)])
        self.pos = self.pos + fwd * self.v * dt
        self.t += 10
        if not self.finished and np.linalg.norm(self.pos - self.vcp[-1]) < 8.0:
            self.finished = True
            self.last_cp = (1, 1)
            self.cp_events += 1
        return self.t

    def _R(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def get_state(self):
        fwd = np.array([np.sin(self.heading), 0.0, np.cos(self.heading)])
        return FakeState(self.t, tuple(self.pos), tuple(fwd * self.v), self.v * 3.6,
                         _Raw(self._R()), self.heading, self.v)

    def rewind_to_state(self, state):
        self.t = state.race_time
        self.pos = np.array(state.position, dtype=np.float64)
        self.heading = state.heading
        self.v = state.speed
        self.finished = False
        self.last_cp = (0, 1)

    def capture_frame(self, w, h):
        img = np.zeros((h, w, 4), np.uint8)
        img[..., 3] = 255
        img[: h // 2, :, :3] = (200, 150, 90)
        img[h // 2:, :, :3] = (60, 90, 60)
        d = np.linalg.norm(self.vcp - self.pos, axis=1)
        j = int(np.argmin(d))
        c, s = np.cos(self.heading), np.sin(self.heading)
        for k in range(j, min(j + 12, len(self.vcp))):
            rel = self.vcp[k] - self.pos
            lz = rel[0] * s + rel[2] * c
            lx = rel[0] * c - rel[2] * s
            if lz > 1.0:
                y = int(h / 2 + (h / 2) * 6.0 / lz)
                x = int(w / 2 + (w / 2) * (lx / lz) * 1.5)
                r = max(1, int(48.0 / lz))
                if 0 <= x < w and 0 <= y < h:
                    img[max(y - r // 2, 0): y + r // 2 + 1, max(x - r, 0): x + r + 1, :3] = (80, 80, 80)
        return img
