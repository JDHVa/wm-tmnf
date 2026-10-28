import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tmnf.game import ensure_game, listener_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    ports = args.ports or [args.port]
    for port in ports:
        c, s0 = ensure_game(port, args.map, timeout_s=args.timeout)
        pos = tuple(round(x, 2) for x in s0.position)
        print(f"[{port}] pid={listener_pid(port)} race_time={s0.race_time} pos={pos} "
              f"vel_kmh={s0.speed_kmh:.1f}")
        c.close()


if __name__ == "__main__":
    main()
