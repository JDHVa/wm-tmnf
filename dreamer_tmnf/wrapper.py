import datetime
import socket
import time
import uuid

import gymnasium as gym
import numpy as np

from tmnf.env import TMNFEnv
from tmnf.episode import prepare_client
from tmnf.game import find_a01

NET_ERRORS = (ConnectionError, TimeoutError, socket.timeout, OSError)
STEP_TIMEOUT_S = 120


class DreamerTMNF:
    def __init__(self, port=8479, time_limit=1200, on_episode=None, retry_s=30, log=print, **env_kwargs):
        self._env = TMNFEnv(port=port, **env_kwargs)
        self._port = port
        self._retry_s = retry_s
        self._log = log
        self._last_obs = None
        self.reconnects = 0
        self.discarded = 0
        self._time_limit = time_limit
        self._on_episode = on_episode
        self._n = self._env.action_space.n
        self._t = 0
        self._ret = 0.0
        self.id = self._new_id()
        obs_spaces = dict(self._env.observation_space.spaces)
        obs_spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        obs_spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        obs_spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)
        self.observation_space = gym.spaces.Dict(obs_spaces)
        space = gym.spaces.Box(low=0, high=1, shape=(self._n,), dtype=np.float32)
        space.discrete = True
        self.action_space = space

    @staticmethod
    def _new_id():
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex}"

    def _reconnect(self):
        env = self._env
        if env.client is not None:
            try:
                env.client.close()
            except Exception:
                pass
            env.client = None
        while True:
            try:
                client, s0 = prepare_client(self._port, find_a01(), timeout_ms=900000, log=self._log, step_timeout_s=30, connect_timeout_s=30)
                client.sock.settimeout(STEP_TIMEOUT_S)
                env.client = client
                env.s0 = s0
                env.cp_seen = client.cp_events
                self.reconnects += 1
                self._log(f"[{self._port}] reconectado ({self.reconnects})")
                return
            except NET_ERRORS as e:
                self._log(f"[{self._port}] sin conexion con el juego ({type(e).__name__}: {e}); reintento en {self._retry_s}s")
                time.sleep(self._retry_s)

    def reset(self):
        self.id = self._new_id()
        while True:
            try:
                if self._env.client is None:
                    self._reconnect()
                obs, info = self._env.reset()
                break
            except NET_ERRORS as e:
                self._log(f"[{self._port}] error en reset ({type(e).__name__}: {e})")
                self._reconnect()
        self._t = 0
        self._ret = 0.0
        self._last_obs = obs
        return self._pack(obs, False, False, True)

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        index = int(np.argmax(np.asarray(action)))
        try:
            obs, reward, terminated, truncated, info = self._env.step(index)
        except NET_ERRORS as e:
            self._log(f"[{self._port}] conexion perdida en step ({type(e).__name__}: {e}); episodio descartado")
            self.discarded += 1
            self._env.client = None
            info = {"discount": np.array(1.0, dtype=np.float32), "reason": "connection_lost"}
            return self._pack(self._last_obs, True, False, False), 0.0, True, info
        self._last_obs = obs
        self._t += 1
        self._ret += reward
        timeout = self._t >= self._time_limit
        done = bool(terminated or truncated or timeout)
        info = dict(info)
        if timeout and not (terminated or truncated):
            info["reason"] = "timeout"
        out = self._pack(obs, done, bool(terminated), False)
        info["discount"] = np.array(0.0 if terminated else 1.0, dtype=np.float32)
        if done and self._on_episode is not None:
            self._on_episode(info, self._ret, self._t)
        return out, float(reward), done, info

    def _pack(self, obs, is_last, is_terminal, is_first):
        return {
            "image": obs["image"],
            "vector": obs["vector"],
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def close(self):
        self._env.close()
