import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import psutil

from tmnf.episode import _prepare, prepare_client
from tmnf.launcher import find_a01
from tmnf.tmi_client import TMIClient


def count_close_wait(port):
    n = 0
    try:
        conns = psutil.net_connections(kind="tcp")
    except psutil.AccessDenied:
        return -1
    for c in conns:
        if c.status != psutil.CONN_CLOSE_WAIT:
            continue
        if (c.laddr and c.laddr.port == port) or (c.raddr and c.raddr.port == port):
            n += 1
    return n


def run_child(args):
    a01 = find_a01()
    if a01 is None:
        print("FALLO no se encontro A01", flush=True)
        return 2
    client, _ = prepare_client(args.port, a01, timeout_ms=args.timeout_ms,
                               step_timeout_s=30, connect_timeout_s=30)
    client.set_speed(1)
    for _ in range(40):
        client.set_input(accelerate=True)
        client.wait_step()
    client.set_input(accelerate=True)
    print("READY", flush=True)
    if args.shutdown:
        client.close()
        return 0
    time.sleep(3600)
    return 0


def spawn_child(args, shutdown):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--child",
           "--port", str(args.port), "--timeout-ms", str(args.timeout_ms)]
    if shutdown:
        cmd.append("--shutdown")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)


def wait_ready(proc, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return False
            continue
        line = line.strip()
        if line:
            print(f"    hijo: {line}", flush=True)
        if line == "READY":
            return True
    return False


def one_round(args, a01, index, shutdown):
    label = "shutdown limpio" if shutdown else "kill"
    print(f"Ronda {index} ({label})", flush=True)
    proc = spawn_child(args, shutdown)
    try:
        if not wait_ready(proc):
            print("  FALLO: el hijo no llego a READY", flush=True)
            return False, None
        if shutdown:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.kill()
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    t0 = time.perf_counter()
    client = TMIClient(args.port)
    try:
        client.connect(timeout_s=20)
    except Exception as e:
        print(f"  FALLO al reconectar tras {time.perf_counter() - t0:.2f} s: {type(e).__name__}: {e}", flush=True)
        client.close()
        print(f"  CLOSE_WAIT en {args.port}: {count_close_wait(args.port)}", flush=True)
        return False, None
    elapsed = time.perf_counter() - t0
    ok = False
    try:
        client.sock.settimeout(60)
        client, state = _prepare_connected(client, args, a01)
        t_start = state.race_time
        client.set_speed(1)
        t = t_start
        client.sock.settimeout(30)
        for _ in range(20):
            client.set_input(accelerate=True)
            t = client.wait_step()
        ok = t > t_start
        print(f"  race_time {t_start} -> {t}", flush=True)
    except Exception as e:
        print(f"  FALLO tras conectar: {type(e).__name__}: {e}", flush=True)
    finally:
        client.close()
    print(f"  reconexion en {elapsed:.2f} s: {'OK' if ok else 'FALLO'}", flush=True)
    time.sleep(1)
    print(f"  CLOSE_WAIT en {args.port}: {count_close_wait(args.port)}", flush=True)
    return ok, elapsed


def _prepare_connected(client, args, a01):
    return _prepare(client, args.port, a01, args.timeout_ms, print, 30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8479)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout-ms", type=int, default=900000)
    p.add_argument("--also-shutdown", action="store_true")
    p.add_argument("--child", action="store_true")
    p.add_argument("--shutdown", action="store_true")
    args = p.parse_args()
    if args.child:
        return run_child(args)

    a01 = find_a01()
    if a01 is None:
        print("No se encontro A01")
        return 1
    plan = [(i + 1, False) for i in range(args.rounds)]
    if args.also_shutdown:
        plan.append((args.rounds + 1, True))
    results = []
    for index, shutdown in plan:
        ok, elapsed = one_round(args, a01, index, shutdown)
        results.append((index, shutdown, ok, elapsed))

    print("\nResumen")
    failed = 0
    for index, shutdown, ok, elapsed in results:
        tag = "shutdown" if shutdown else "kill"
        et = f"{elapsed:.2f} s" if elapsed is not None else "-"
        print(f"  ronda {index} [{tag}]: {'OK' if ok else 'FALLO'} ({et})")
        failed += 0 if ok else 1
    print(f"CLOSE_WAIT finales en {args.port}: {count_close_wait(args.port)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
