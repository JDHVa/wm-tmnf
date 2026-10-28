import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.episode import SEGMENT_MS, POS_TOL, prepare_client, run_scripted_segment, scripted_input, vec_diff
from tmnf.launcher import find_a01, port_is_listening


def run_mode(c, s0, name, reps, speed, frames, check):
    print(f"\n=== modo {name}: {reps} repeticiones, x{speed}, frames={'si' if frames else 'no'} ===")
    ref = None
    worst = 0.0
    for i in range(reps):
        r = run_scripted_segment(c, s0, scripted_input, SEGMENT_MS, speed=speed, frames=frames, check=check, trace_steps=6)
        f = r["final"]
        if ref is None:
            ref = f
        d = vec_diff(f.position, ref.position)
        worst = max(worst, d)
        pos = tuple(round(v, 3) for v in f.position)
        print(f"  rep {i:02d} t_fin={f.race_time} pos={pos} dif={d:.4f} m pasos={r['n_steps']} "
              f"primeros_t={r['trace']} rewind_dif={r['rewind_diff']} reintentos={r['rewind_retries']}")
    print(f"  dif max vs rep 0: {worst:.4f} m -> {'determinista' if worst <= POS_TOL else 'NO determinista'}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=20)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--frame-size", default="160x120")
    ap.add_argument("--modes", default="raw,checked")
    args = ap.parse_args()
    w, h = (int(v) for v in args.frame_size.lower().split("x"))
    frames = dict(size=(w, h), every_ms=50) if args.frames else None
    modes = [m for m in args.modes.split(",") if m]

    if args.dry:
        print("Imports OK")
        print("Puerto:", args.port, "escuchando:", port_is_listening(args.port))
        print("Modos:", modes, "reps:", args.reps, "velocidad:", args.speed, "frames:", frames)
        print("A01:", find_a01() or "no encontrado")
        return

    c, s0 = prepare_client(args.port, find_a01())
    print(f"S0: t={s0.race_time} pos={tuple(round(v, 3) for v in s0.position)} vel={tuple(round(v, 3) for v in s0.velocity)}")
    try:
        for m in modes:
            run_mode(c, s0, m, args.reps, args.speed, frames, check=(m == "checked"))
    finally:
        try:
            c.set_input()
            c.set_speed(1)
            c.unrequest_frame()
        except OSError:
            pass
        c.close()


if __name__ == "__main__":
    main()
