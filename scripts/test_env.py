import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gymnasium.utils.env_checker import check_env

from tmnf import env as envmod
from tmnf.env import TMNFEnv
from tmnf.fake_client import FakeClient

OUT = os.path.join(ROOT, "runs", "env_test")
VCP = os.path.join(ROOT, "data", "vcp", "A01.npy")
GAS_STRAIGHT = 4


class StripCounters(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        o, i = self.env.reset(seed=seed, options=options)
        return o, self._strip(i)

    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        return o, r, te, tr, self._strip(i)

    @staticmethod
    def _strip(i):
        return {k: v for k, v in i.items() if k not in ("total_env_steps", "total_sim_ms")}


def make_env(args):
    if args.fake:
        return TMNFEnv(client=FakeClient(VCP), max_time_ms=args.max_time_ms)
    return TMNFEnv(port=args.port, max_time_ms=args.max_time_ms, speed=args.speed)


def random_policy(rng):
    def act(obs):
        throttle = rng.choice(3, p=[0.1, 0.8, 0.1])
        steer = rng.integers(3)
        return throttle * 3 + steer
    return act


def heuristic_policy(args):
    def act(obs):
        v = obs["vector"]
        tgt = v[4 + 3 * args.target_k: 7 + 3 * args.target_k]
        lat = args.lat_sign * tgt[args.lat_axis]
        fwd = tgt[args.fwd_axis]
        speed = v[3] * 500.0
        thr = 1
        if speed > args.max_speed:
            thr = 0
        steer = 1
        if abs(lat) > args.deadzone * max(abs(fwd), 1e-3):
            steer = 2 if lat > 0 else 0
        return thr * 3 + steer
    return act


def run_episodes(env, policy, n, record, label):
    rows = []
    for ep in range(n):
        obs, info = env.reset(seed=ep)
        done = False
        total = 0.0
        steps = 0
        t0 = time.perf_counter()
        sim0 = env.total_sim_ms
        while not done:
            a = policy(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            steps += 1
            done = term or trunc
            if record is not None:
                if steps % 3 == 0:
                    record["imgs"].append(obs["image"])
                    record["full"].append(env.last_full_frame)
        wall = time.perf_counter() - t0
        sim = (env.total_sim_ms - sim0) / 1000.0
        rows.append(dict(steps=steps, wall=wall, sim=sim, reward=total,
                         progress_m=info["progress_m"], reason=info["reason"]))
        print(f"  [{label}] ep{ep}: pasos={steps} {steps / wall:.1f} pasos/s sim={sim:.1f}s "
              f"wall={wall:.1f}s factor={sim / wall:.1f}x R={total:.2f} "
              f"progreso={info['progress_m']:.0f} m fin={info['reason']}")
    return rows


def save_artifacts(record, tag):
    os.makedirs(OUT, exist_ok=True)
    imgs = record["imgs"]
    if imgs:
        n = min(len(imgs), 64)
        sel = [imgs[i] for i in np.linspace(0, len(imgs) - 1, n).astype(int)]
        cols = 8
        rows = (n + cols - 1) // cols
        h, w, _ = sel[0].shape
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, im in enumerate(sel):
            canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
        Image.fromarray(canvas).save(os.path.join(OUT, f"mosaic_{tag}.png"))
        Image.fromarray(record["full"][len(record["full"]) // 2]).save(os.path.join(OUT, f"full_{tag}.png"))
    full = record["full"]
    if full:
        step = max(1, len(full) // 60)
        frames = [Image.fromarray(f) for f in full[::step][:60]]
        frames[0].save(os.path.join(OUT, f"clip_{tag}.gif"), save_all=True, append_images=frames[1:],
                       duration=80, loop=0)
    print(f"  artefactos en {OUT} (tag={tag})")


def summarize(rows, label):
    steps = sum(r["steps"] for r in rows)
    wall = sum(r["wall"] for r in rows)
    sim = sum(r["sim"] for r in rows)
    reasons = [r["reason"] for r in rows]
    print(f"== {label}: {steps / wall:.1f} pasos/s, factor sim/real {sim / wall:.1f}x, "
          f"R medio {np.mean([r['reward'] for r in rows]):.2f}, "
          f"progreso medio {np.mean([r['progress_m'] for r in rows]):.0f} m, fines {reasons}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-time-ms", type=int, default=60000)
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--fwd-axis", type=int, default=2)
    ap.add_argument("--lat-axis", type=int, default=0)
    ap.add_argument("--lat-sign", type=float, default=-1.0)
    ap.add_argument("--target-k", type=int, default=2)
    ap.add_argument("--deadzone", type=float, default=0.05)
    ap.add_argument("--max-speed", type=float, default=400.0)
    ap.add_argument("--no-random", action="store_true")
    args = ap.parse_args()

    env = make_env(args)
    if args.fake and not args.skip_check:
        check_env(StripCounters(make_env(args)), skip_render_check=True)
        print("check_env OK")
    tag = "fake" if args.fake else "live"
    rng = np.random.default_rng(0)
    if not args.no_random:
        rows = run_episodes(env, random_policy(rng), args.episodes, None, "aleatoria")
        summarize(rows, "aleatoria")
    record = dict(imgs=[], full=[])
    rows = run_episodes(env, heuristic_policy(args), args.episodes, record, "heuristica")
    summarize(rows, "heuristica")
    save_artifacts(record, tag)
    print(f"total_env_steps={env.total_env_steps} total_sim_ms={env.total_sim_ms}")
    env.close()


if __name__ == "__main__":
    main()
