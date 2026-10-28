import argparse
import json
import os
import subprocess
import sys
import multiprocessing as mp
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.episode import POS_TOL, SEGMENT_MS, prepare_client, run_scripted_segment, scripted_input
from tmnf.launcher import BASE_PORT, find_a01, launch_instance, port_is_listening, tm_pids, wait_for_port

try:
    import psutil
except ImportError:
    psutil = None

OUT_DIR = os.path.join(ROOT, "runs", "bench")
FRAME_EVERY_MS = 50


def summarize_positions(finals):
    if not finals:
        return dict(n_distinct=0)
    arr = np.array(finals)
    ref = arr[0]
    distinct = {tuple(np.round(p, 3)) for p in arr}
    return dict(
        n_distinct=len(distinct),
        ref=[float(v) for v in ref],
        min=[float(v) for v in arr.min(axis=0)],
        max=[float(v) for v in arr.max(axis=0)],
        max_diff_per_rep=[float(np.max(np.abs(p - ref))) for p in arr][:200],
    )


def run_reps(c, s0, speed, frames, frame_size, duration, check):
    fcfg = dict(size=frame_size, every_ms=FRAME_EVERY_MS) if frames else None
    race_ms = 0
    finals = []
    end_times = []
    frame_ms = []
    retries = 0
    err = None
    stop_at = time.perf_counter() + duration
    w0 = time.perf_counter()
    try:
        while time.perf_counter() < stop_at:
            r = run_scripted_segment(c, s0, scripted_input, SEGMENT_MS, speed=speed, frames=fcfg, check=check)
            f = r["final"]
            race_ms += f.race_time - s0.race_time
            finals.append(list(f.position))
            end_times.append(f.race_time)
            frame_ms.extend(r["frame_ms"])
            retries += r["rewind_retries"]
    except Exception as e:
        err = repr(e)
    wall = time.perf_counter() - w0
    ref = np.array(finals[0]) if finals else None
    diff = max((float(np.max(np.abs(np.array(p) - ref))) for p in finals), default=0.0)
    n_frames = len(frame_ms)
    return dict(
        reps=len(finals),
        race_ms=race_ms,
        wall_s=wall,
        speedup=(race_ms / 1000) / wall if wall > 0 else 0.0,
        max_pos_diff=diff,
        deterministic=err is None and len(finals) >= 2 and diff <= POS_TOL,
        frames=n_frames,
        frames_per_s=n_frames / wall if wall > 0 else 0.0,
        ms_per_frame=float(np.mean(frame_ms)) if frame_ms else None,
        rewind_retries=retries,
        end_race_times=sorted(set(end_times)),
        end_positions=summarize_positions(finals),
        error=err,
    )


def worker(port, map_path, cmd_q, res_q):
    c = None
    try:
        c, s0 = prepare_client(port, map_path)
        res_q.put(dict(port=port, kind="ready"))
    except Exception as e:
        res_q.put(dict(port=port, kind="ready", error=repr(e)))
        return
    try:
        while True:
            cmd = cmd_q.get()
            if cmd is None:
                break
            res = run_reps(c, s0, cmd["speed"], cmd["frames"], cmd["frame_size"], cmd["duration"], cmd["check"])
            res.update(port=port, kind="result")
            res_q.put(res)
    finally:
        try:
            c.set_input()
            c.set_speed(1)
            c.unrequest_frame()
        except Exception:
            pass
        try:
            c.close()
        except Exception:
            pass


def gpu_sample():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        u, m = out.strip().splitlines()[0].split(",")
        return float(u), float(m)
    except Exception:
        return None


class Monitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.cpu = []
        self.ram_gb = []
        self.tm_ram_gb = []
        self.gpu = []
        self.gpu_mem = []

    def run(self):
        if psutil:
            psutil.cpu_percent(None)
        while not self.stop_flag.is_set():
            time.sleep(1.0)
            if psutil:
                self.cpu.append(psutil.cpu_percent(None))
                self.ram_gb.append(psutil.virtual_memory().used / 1e9)
                rss = 0
                for pid in tm_pids():
                    try:
                        rss += psutil.Process(pid).memory_info().rss
                    except Exception:
                        pass
                self.tm_ram_gb.append(rss / 1e9)
            g = gpu_sample()
            if g:
                self.gpu.append(g[0])
                self.gpu_mem.append(g[1])

    def summary(self):
        def m(x):
            return float(np.mean(x)) if x else None

        return dict(cpu_pct=m(self.cpu), ram_gb=m(self.ram_gb), tm_ram_gb=m(self.tm_ram_gb),
                    gpu_pct=m(self.gpu), gpu_mem_mb=m(self.gpu_mem))


def ensure_instances(n, launched):
    ports = []
    for i in range(n):
        port = BASE_PORT + i
        if not port_is_listening(port):
            print(f"Lanzando instancia en puerto {port} (se abrira una ventana de TMNF)...")
            launched[port] = launch_instance(port)
            if not wait_for_port(port, 180):
                raise RuntimeError(f"El puerto {port} no abrio a tiempo")
            time.sleep(5)
        ports.append(port)
    return ports


def fmt(v, spec):
    return format(v, spec) if v is not None else "n/d"


def print_table(rows):
    print("\n=== Resumen paralelo ===")
    print(f"{'n':>2} {'total':>8} {'por inst.':>10} {'CPU %':>7} {'RAM GB':>7} {'TM GB':>6} {'GPU %':>6} {'fps cap.':>9} {'ms/cap.':>8} {'determinista':>13}")
    for r in rows:
        res = r["resources"]
        fps = [i["frames_per_s"] for i in r["instances"] if i.get("frames")]
        mspf = [i["ms_per_frame"] for i in r["instances"] if i.get("ms_per_frame")]
        print(f"{r['n']:>2} {r['total_speedup']:>7.2f}x {r['per_instance_speedup']:>9.2f}x "
              f"{fmt(res['cpu_pct'], '.1f'):>7} {fmt(res['ram_gb'], '.1f'):>7} "
              f"{fmt(res['tm_ram_gb'], '.1f'):>6} {fmt(res['gpu_pct'], '.0f'):>6} "
              f"{fmt(float(np.mean(fps)) if fps else None, '.1f'):>9} {fmt(float(np.mean(mspf)) if mspf else None, '.1f'):>8} "
              f"{'si' if r['deterministic'] else 'NO':>13}")
    for r in rows:
        for i in r["instances"]:
            ep = i.get("end_positions", {})
            print(f"  n={r['n']} [{i['port']}] reps={i.get('reps')} {i.get('speedup', 0):.2f}x dif max={i.get('max_pos_diff', 0):.4f} m "
                  f"posiciones finales distintas={ep.get('n_distinct')} t_fin={i.get('end_race_times')} reintentos rewind={i.get('rewind_retries')}")


def collect(res_q, kind, n, timeout):
    got = {}
    t_end = time.time() + timeout
    while len(got) < n:
        left = t_end - time.time()
        if left <= 0:
            raise RuntimeError(f"Tiempo agotado esperando '{kind}' ({len(got)}/{n})")
        msg = res_q.get(timeout=left)
        if msg.get("kind") == kind:
            got[msg["port"]] = msg
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-instances", type=int, default=3)
    ap.add_argument("--speed", type=float, default=20)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--frame-size", default="160x120")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    fw, fh = (int(v) for v in args.frame_size.lower().split("x"))
    frame_size = (fw, fh)

    a01 = find_a01()
    if args.dry:
        print("Imports OK")
        print("Instancias:", args.max_instances, "velocidad:", args.speed, "frames:", args.frames, "tamano:", frame_size)
        print("psutil:", "disponible" if psutil else "NO instalado (CPU/RAM no se mediran)")
        print("nvidia-smi:", gpu_sample() or "no disponible")
        print("A01:", a01 or "no encontrado")
        print("Puertos escuchando:", [BASE_PORT + i for i in range(args.max_instances) if port_is_listening(BASE_PORT + i)])
        return

    if a01 is None:
        print("No se encontro A01-Race.Challenge.Gbx")
        sys.exit(1)
    if psutil is None:
        print("Aviso: psutil no esta instalado; CPU y RAM saldran como n/d.")

    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()
    launched = {}
    procs = {}
    cmd_qs = {}
    rows = []
    try:
        for n in range(1, args.max_instances + 1):
            print(f"\n--- n = {n} ---")
            ports = ensure_instances(n, launched)
            new = [p for p in ports if p not in procs]
            for p in new:
                cmd_qs[p] = ctx.Queue()
                procs[p] = ctx.Process(target=worker, args=(p, a01, cmd_qs[p], res_q), daemon=True)
                procs[p].start()
            if new:
                ready = collect(res_q, "ready", len(new), 180)
                bad = {p: m["error"] for p, m in ready.items() if m.get("error")}
                if bad:
                    raise RuntimeError(f"Fallo al preparar instancias: {bad}")
            mon = Monitor()
            mon.start()
            cmd = dict(speed=args.speed, frames=args.frames, frame_size=frame_size,
                       duration=args.duration, check=not args.no_check)
            for p in ports:
                cmd_qs[p].put(cmd)
            results = collect(res_q, "result", len(ports), args.duration + 120)
            mon.stop_flag.set()
            mon.join(timeout=3)
            insts = [results[p] for p in ports]
            elapsed = max(r["wall_s"] for r in insts)
            total_race = sum(r["race_ms"] for r in insts) / 1000
            row = dict(
                n=n,
                speed=args.speed,
                frames=args.frames,
                frame_size=list(frame_size),
                elapsed_s=elapsed,
                total_speedup=total_race / elapsed if elapsed > 0 else 0.0,
                per_instance_speedup=float(np.mean([r["speedup"] for r in insts])),
                deterministic=all(r["deterministic"] for r in insts),
                resources=mon.summary(),
                instances=[{k: v for k, v in r.items() if k != "kind"} for r in insts],
            )
            for r in row["instances"]:
                if r["error"]:
                    print(f"  [{r['port']}] error: {r['error']}")
            rows.append(row)
            print(f"  total {row['total_speedup']:.2f}x, por instancia {row['per_instance_speedup']:.2f}x")
        print_table(rows)
    finally:
        for q in cmd_qs.values():
            q.put(None)
        for pr in procs.values():
            pr.join(timeout=15)
            if pr.is_alive():
                pr.terminate()
        with open(os.path.join(OUT_DIR, "parallel.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Resultados en {os.path.join(OUT_DIR, 'parallel.json')}")
        print("Los juegos siguen abiertos. Puertos:", sorted(procs), "PIDs TmForever:", tm_pids() or "n/d (sin psutil)")


if __name__ == "__main__":
    main()
