import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIVERGENCE_TOL = 1e-3


def main():
    ap = argparse.ArgumentParser(description="Repite una vuelta grabada en el juego y guarda cuadros HD alineados.")
    ap.add_argument("--npz", default=os.path.join("results", "mejor_vuelta_28.75s.npz"))
    ap.add_argument("--port", type=int, default=8479)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.isabs(args.npz):
        args.npz = os.path.join(ROOT, args.npz)
    stem = os.path.splitext(os.path.basename(args.npz))[0]
    out = args.out or os.path.join(ROOT, "results", f"{stem}_hd.npz")

    d = np.load(args.npz)
    actions = np.argmax(d["action"], axis=1)
    vectors = d["vector"]
    T = len(actions)

    from tmnf.env import TMNFEnv
    from tmnf.episode import prepare_client
    from tmnf.launcher import find_a01

    client, s0 = prepare_client(args.port, find_a01(), timeout_ms=900000, step_timeout_s=30, connect_timeout_s=30)
    try:
        client.sock.settimeout(120)
        env = TMNFEnv(port=args.port, speed=args.speed, fail_penalty=10.0, cp_bonus=10.0, client=client,
                      capture_size=args.size)
        env.s0 = s0
        obs, _ = env.reset()
        frames = np.zeros((T, args.size, args.size, 3), np.uint8)
        vecs = np.zeros((T, 25), np.float32)
        frames[0] = env.last_full_frame
        vecs[0] = obs["vector"]
        last_t = 0
        reason = None
        for t in range(1, T):
            obs, r, term, trunc, info = env.step(int(actions[t]))
            frames[t] = env.last_full_frame
            vecs[t] = obs["vector"]
            last_t = t
            reason = info.get("reason")
            if t % 100 == 0:
                print(f"paso {t}/{T - 1}", flush=True)
            if term or trunc:
                break
        n = last_t + 1
        diff = np.abs(vecs[:n].astype(np.float64) - vectors[:n].astype(np.float64)).max(axis=1)
        div = np.nonzero(diff > DIVERGENCE_TOL)[0]
        print(f"pasos capturados: {n} de {T}")
        print(f"max |vector replay - vector npz|: {diff.max():.3e}")
        print(f"primer paso con divergencia > {DIVERGENCE_TOL}: {int(div[0]) if len(div) else 'ninguno'}")
        print(f"razon final: {reason} (terminó con finish: {reason == 'finish'})")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, frames=frames, vector=vecs)
        print(f"guardado: {out}")
    finally:
        try:
            client.set_input()
            client.set_speed(1)
            client.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
