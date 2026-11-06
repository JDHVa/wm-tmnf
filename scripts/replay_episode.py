import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_NPZ = os.path.join(
    "runs", "dreamer", "2026-10-29_ent3e-3_fp10", "train_eps",
    "20261104T224615-391900dfe5614f028c7d4536c539b61e-677.npz",
)
FINAL_ZONE = 20
DIVERGENCE_TOL = 1e-3
FIELDS = ["step", "race_time", "x", "y", "z", "speed_kmh", "last_idx", "progress_m", "lateral_m",
          "reward", "stored_reward", "vec_diff", "cp_events", "cp_passed", "cp_states",
          "cp_times_set", "reason"]


def load_npz(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def short_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("-")
    return f"{parts[0]}_{parts[-1]}" if len(parts) > 1 else stem


def default_out(npz_path, capture):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(npz_path)))
    tag = "capture" if capture else "nocapture"
    return os.path.join(run_dir, "replay", f"{short_name(npz_path)}_{tag}.csv")


def cp_info(st):
    cd = st.raw.cp_data
    n_states = int(cd.cp_states_length)
    n_times = int(cd.cp_times_length)
    states = [1 if bool(cd.cp_states[i]) else 0 for i in range(n_states)]
    set_times = 0
    for i in range(n_times):
        e = cd.cp_times[i]
        t = e.time if hasattr(e, "time") else e[0]
        if int(t) != -1:
            set_times += 1
    return states, set_times


def install_event_log(client, step_ref):
    from tmnf.tmi_client import MsgType
    events = []
    original = client._handle_other

    def wrapped(msg):
        original(msg)
        if msg in (MsgType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MsgType.SC_LAP_COUNT_CHANGED_SYNC):
            kind = "CP" if msg == MsgType.SC_CHECKPOINT_COUNT_CHANGED_SYNC else "LAP"
            events.append((kind, client.last_cp, step_ref[0]))

    client._handle_other = wrapped
    return events


def verdict(events, divergence_step, n_actions, replayed, states, total_cp):
    if any(k == "CP" and cp is not None and cp[0] == cp[1] for k, cp, _ in events):
        return "META DETECTADA"
    if divergence_step is not None and divergence_step < n_actions - FINAL_ZONE:
        return "REPLAY NO FIEL, no concluyente"
    passed = sum(states)
    if total_cp and passed < total_cp - 1:
        return "H1: faltó checkpoint"
    if total_cp and passed == total_cp - 1 and not states[-1]:
        return "H2 posible: revisar con --no-capture"
    if total_cp and passed < total_cp:
        return "H1: faltó checkpoint"
    return "sin veredicto claro: todos los CP marcados pero sin evento de meta"


def replay(env, client, s0, data, args):
    actions = np.argmax(data["action"], axis=1)
    vectors = data["vector"]
    stored_reward = data["reward"]
    T = len(actions)
    total_cp = int(s0.raw.cp_data.cp_states_length)
    step_ref = [0]
    events = install_event_log(client, step_ref)

    if args.no_capture:
        env._frame = lambda: np.zeros((env.img_size, env.img_size, 3), np.uint8)

    env.reset()
    out = args.out or default_out(args.npz, not args.no_capture)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    seq = [int(actions[t]) for t in range(1, T)]
    seq += [seq[-1]] * args.extra_steps if seq else []
    n_planned = len(seq)

    divergence_step = None
    max_div = 0.0
    prev_passed = -1
    seen_events = 0
    total_reward = 0.0
    last_info = None
    last_st = None
    finished = False
    steps_done = 0

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i, a in enumerate(seq):
            t = i + 1
            step_ref[0] = t
            obs, r, term, trunc, info = env.step(a)
            steps_done = t
            total_reward += r
            st = client.get_state()
            last_st, last_info = st, info
            states, set_times = cp_info(st)
            passed = sum(states)
            vd = ""
            sr = ""
            if t < T:
                vd_val = float(np.max(np.abs(obs["vector"].astype(np.float64) - vectors[t].astype(np.float64))))
                vd = f"{vd_val:.6f}"
                sr = f"{float(stored_reward[t]):.6f}"
                max_div = max(max_div, vd_val)
                if divergence_step is None and vd_val > DIVERGENCE_TOL:
                    divergence_step = t
            x, y, z = st.position
            w.writerow([t, st.race_time, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{st.speed_kmh:.2f}",
                        env.last_idx, f"{info['progress_m']:.2f}", f"{info['lateral_m']:.2f}",
                        f"{r:.5f}", sr, vd, client.cp_events, passed,
                        "".join(str(s) for s in states), set_times, info["reason"] or ""])
            new_events = len(events) - seen_events
            if passed != prev_passed or new_events > 0:
                print(f"  paso {t} t={st.race_time} pos=({x:.1f},{y:.1f},{z:.1f}) v={st.speed_kmh:.1f} km/h "
                      f"last_cp={client.last_cp} cp_states={''.join(str(s) for s in states)}")
                for k, cp, s in events[seen_events:]:
                    print(f"    evento {k} en paso {s}: {cp}")
                prev_passed = passed
                seen_events = len(events)
            if info["reason"] == "finish":
                finished = True
                print(f"  *** META ALCANZADA (reason=finish) en el paso {t}, t={st.race_time} ***")
                break

    states, _ = cp_info(last_st) if last_st is not None else ([], 0)
    try:
        race_finished = client.race_finished()
    except OSError:
        race_finished = None

    orig_total = float(np.sum(stored_reward[1:]))
    print("\n=== resumen ===")
    print(f"pasos repetidos: {steps_done} de {n_planned} planeados (npz: {T - 1} acciones)")
    print(f"eventos totales: {len(events)} -> {[(k, cp, s) for k, cp, s in events]}")
    print(f"ultimo last_cp: {client.last_cp}")
    print(f"cp_states finales: {''.join(str(s) for s in states)}")
    print(f"CP pasados: {sum(states)} de {total_cp}")
    print(f"race_finished(): {race_finished}")
    if last_info is not None:
        print(f"progreso final replay: {last_info['progress_m']:.1f} m, razon final: {last_info['reason']}, "
              f"velocidad: {last_info['speed_kmh']:.1f} km/h")
    print(f"recompensa total replay: {total_reward:.3f} vs original (npz): {orig_total:.3f}")
    if divergence_step is None:
        print(f"divergencia vector: max {max_div:.2e}, sin pasos con diff > {DIVERGENCE_TOL}")
    else:
        print(f"divergencia vector: primer paso con diff > {DIVERGENCE_TOL}: {divergence_step} "
              f"(max {max_div:.4f})")
    v = verdict(events, divergence_step, T - 1, steps_done, states, total_cp)
    print(f"veredicto: {v}")
    print(f"csv: {out}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--port", type=int, default=8479)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--fail-penalty", type=float, default=10.0)
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--extra-steps", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if not os.path.isabs(args.npz):
        args.npz = os.path.join(ROOT, args.npz)

    data = load_npz(args.npz)
    actions = np.argmax(data["action"], axis=1)
    print(f"npz: {args.npz}")
    print("formas:", {k: v.shape for k, v in data.items()})
    print(f"acciones a repetir: {len(actions) - 1}")
    print(f"recompensa total: {float(np.sum(data['reward'])):.3f}")
    print(f"salida csv: {args.out or default_out(args.npz, not args.no_capture)}")
    if args.dry:
        return

    from tmnf.env import TMNFEnv
    from tmnf.episode import prepare_client
    from tmnf.launcher import find_a01

    client, s0 = prepare_client(args.port, find_a01(), timeout_ms=900000, step_timeout_s=30, connect_timeout_s=30)
    try:
        client.sock.settimeout(120)
        print(f"S0: t={s0.race_time} pos={tuple(round(v, 3) for v in s0.position)} "
              f"vel={tuple(round(v, 3) for v in s0.velocity)}")
        print(f"CP totales del mapa: cp_states_length={s0.raw.cp_data.cp_states_length} "
              f"cp_times_length={s0.raw.cp_data.cp_times_length}")
        env = TMNFEnv(port=args.port, speed=args.speed, fail_penalty=args.fail_penalty, client=client)
        env.s0 = s0
        replay(env, client, s0, data, args)
    finally:
        try:
            client.set_input()
            client.set_speed(1)
            client.unrequest_frame()
        except OSError:
            pass
        client.close()


if __name__ == "__main__":
    main()
