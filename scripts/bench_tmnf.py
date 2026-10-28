import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.tmi_client import TMIClient
from tmnf.episode import SEGMENT_MS, POS_TOL, reach_start, run_scripted_segment, scripted_input

OUT_DIR = os.path.join(ROOT, "runs", "bench")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
SPEEDS = [1, 2, 5, 10, 20]
FRAME_SIZES = [(256, 256), (160, 120), (64, 64)]
SAVE_SIZES = [(256, 256), (160, 120)]
SAVE_EVERY_MS = 350
TIMING_EVERY_MS = 50


def pos_diff(a, b):
    return float(np.max(np.abs(np.array(a) - np.array(b))))


def timed_segment(c, s0, speed, frame_size=None, frame_every=None, on_frame=None):
    frames = None
    if frame_size is not None:
        frames = dict(size=frame_size, every_ms=frame_every, on_frame=on_frame)
    w0 = time.perf_counter()
    r = run_scripted_segment(c, s0, scripted_input, SEGMENT_MS, speed=speed, frames=frames)
    wall = time.perf_counter() - w0
    c.set_input()
    return r["final"], wall, r["frame_ms"]


def test_rewind(c, s0):
    print("\n=== a) Prueba de rewind y determinismo ===")
    res = {"rewind": [], "final_positions": []}
    for i in range(3):
        c.rewind_to_state(s0)
        c.set_speed(1)
        c.set_input(accelerate=True)
        t = c.wait_step()
        while t - s0.race_time < 2000:
            c.set_input(accelerate=True)
            t = c.wait_step()
        reached = c.get_state()
        c.rewind_to_state(s0)
        back = c.get_state()
        dpos = pos_diff(back.position, s0.position)
        dt = back.race_time - s0.race_time
        dvel = pos_diff(back.velocity, s0.velocity)
        res["rewind"].append({"pos_diff": dpos, "time_diff_ms": dt, "vel_diff": dvel})
        res["final_positions"].append(list(reached.position))
        print(f"  intento {i + 1}: avanzo a t={reached.race_time} pos={tuple(round(v, 2) for v in reached.position)}")
        print(f"             tras rewind: dif pos={dpos:.6f} m, dif tiempo={dt} ms, dif vel={dvel:.6f}")
    ref = res["final_positions"][0]
    res["determinism_max_diff"] = max(pos_diff(p, ref) for p in res["final_positions"])
    print(f"  determinismo (misma secuencia, 3 corridas): dif max de posicion final = {res['determinism_max_diff']:.6f} m")
    return res


def test_speeds(c, s0):
    print("\n=== b) Prueba de velocidades ===")
    rows = []
    ref_pos = None
    for sp in SPEEDS:
        final, wall, _ = timed_segment(c, s0, sp)
        if ref_pos is None:
            ref_pos = final.position
        row = {
            "speed": sp,
            "race_ms": final.race_time - s0.race_time,
            "wall_ms": wall * 1000,
            "speedup": (final.race_time - s0.race_time) / (wall * 1000),
            "pos_diff_vs_1x": pos_diff(final.position, ref_pos),
            "final_pos": list(final.position),
        }
        row["deterministic"] = row["pos_diff_vs_1x"] <= POS_TOL
        rows.append(row)
        print(f"  x{sp:<3} carrera={row['race_ms']} ms  real={row['wall_ms']:.0f} ms  aceleracion={row['speedup']:.2f}x  dif pos vs x1={row['pos_diff_vs_1x']:.5f} m")
    c.set_speed(1)
    return rows


def save_frame(frame, path):
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))


def test_frames(c, s0, speed_rows):
    print("\n=== c) Prueba de capturas de pantalla ===")
    os.makedirs(FRAMES_DIR, exist_ok=True)
    good = [r["speed"] for r in speed_rows if r["deterministic"]]
    best = max(good) if good else 1
    speeds = sorted({1, best})
    print(f"  velocidades a probar: {speeds} (la mas alta determinista: x{best})")
    base_wall = {r["speed"]: r["wall_ms"] for r in speed_rows}
    out = {"best_deterministic_speed": best, "runs": []}
    for sp in speeds:
        for (w, h) in SAVE_SIZES:
            def on_frame(i, dt, frame, sp=sp, w=w, h=h):
                save_frame(frame, os.path.join(FRAMES_DIR, f"x{sp}_{w}x{h}_{i:02d}.png"))
                if (w, h) == (256, 256):
                    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                    save_frame(small, os.path.join(FRAMES_DIR, f"x{sp}_64x64_from256_{i:02d}.png"))
            _, wall, ft = timed_segment(c, s0, sp, frame_size=(w, h), frame_every=SAVE_EVERY_MS, on_frame=on_frame)
            print(f"  guardado x{sp} {w}x{h}: {len(ft)} capturas, {np.mean(ft):.1f} ms/captura")
        for (w, h) in FRAME_SIZES:
            _, wall, ft = timed_segment(c, s0, sp, frame_size=(w, h), frame_every=TIMING_EVERY_MS)
            entry = {
                "speed": sp,
                "size": [w, h],
                "frames": len(ft),
                "frame_ms_mean": float(np.mean(ft)),
                "frame_ms_p95": float(np.percentile(ft, 95)),
                "wall_ms": wall * 1000,
                "speedup": SEGMENT_MS / (wall * 1000),
                "speedup_without_frames": SEGMENT_MS / base_wall[sp] if sp in base_wall else None,
            }
            out["runs"].append(entry)
            print(f"  x{sp:<3} {w}x{h}: {entry['frame_ms_mean']:.1f} ms/captura (p95 {entry['frame_ms_p95']:.1f}), una cada {TIMING_EVERY_MS} ms de carrera -> aceleracion {entry['speedup']:.2f}x")
    c.set_speed(1)
    return out


def print_summary(summary):
    print("\n=== Resumen ===")
    print(f"{'vel':>5} {'carrera ms':>11} {'real ms':>9} {'aceleracion':>12} {'dif pos m':>10} {'determinista':>13}")
    for r in summary["speeds"]:
        print(f"{'x' + str(r['speed']):>5} {r['race_ms']:>11} {r['wall_ms']:>9.0f} {r['speedup']:>11.2f}x {r['pos_diff_vs_1x']:>10.5f} {'si' if r['deterministic'] else 'NO':>13}")
    print(f"\nRewind: dif max de posicion = {max(r['pos_diff'] for r in summary['rewind']['rewind']):.6f} m; determinismo = {summary['rewind']['determinism_max_diff']:.6f} m")
    print(f"\n{'vel':>5} {'tamano':>9} {'ms/captura':>11} {'aceleracion con capturas':>26}")
    for r in summary["frames"]["runs"]:
        print(f"{'x' + str(r['speed']):>5} {str(r['size'][0]) + 'x' + str(r['size'][1]):>9} {r['frame_ms_mean']:>11.1f} {r['speedup']:>25.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--dry", action="store_true", help="solo verifica imports, no conecta")
    args = ap.parse_args()
    if args.dry:
        print("Imports OK, velocidades:", SPEEDS, "tamanos:", FRAME_SIZES)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    c = TMIClient(args.port)
    print(f"Conectando a 127.0.0.1:{args.port} ...")
    try:
        c.connect(timeout_s=60)
    except ConnectionRefusedError:
        print("No se pudo conectar: el plugin Python_Link no esta escuchando. Revisa docs/primera_prueba.md")
        sys.exit(1)
    summary = {}
    try:
        c.set_timeout(20000)
        c.set_on_step_period(10)
        print("Esperando el inicio de la carrera (t >= 0)...")
        t0 = reach_start(c)
        s0 = c.get_state()
        print(f"Estado S0 guardado: t={s0.race_time} ms pos={tuple(round(v, 2) for v in s0.position)}")
        summary["s0"] = {"race_time": s0.race_time, "position": list(s0.position)}
        summary["rewind"] = test_rewind(c, s0)
        summary["speeds"] = test_speeds(c, s0)
        summary["frames"] = test_frames(c, s0, summary["speeds"])
        print_summary(summary)
    except socket.timeout:
        print("Timeout: el juego dejo de responder. Verifica que la carrera siga activa.")
    finally:
        try:
            c.set_input()
            c.set_speed(1)
            c.unrequest_frame()
        except OSError:
            pass
        c.close()
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Resultados en {os.path.join(OUT_DIR, 'summary.json')}")
        print("Desconectado.")


if __name__ == "__main__":
    main()
