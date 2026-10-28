import time

import numpy as np

from .launcher import load_map, restore_windows
from .tmi_client import TMIClient

POS_TOL = 1e-3
SEGMENT_MS = 3500
REWIND_TRIES = 3


def scripted_input(dt):
    if dt < 3000:
        return dict(accelerate=True)
    return dict(accelerate=True, left=True)


def vec_diff(a, b):
    return float(np.max(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float))))


def reach_start(c):
    t = c.wait_step()
    while t < 0:
        t = c.wait_step()
    return t


def prepare_client(port, map_path, timeout_ms=300000, log=print, step_timeout_s=None, connect_timeout_s=60):
    restore_windows()
    c = TMIClient(port)
    try:
        c.connect(timeout_s=connect_timeout_s)
        return _prepare(c, port, map_path, timeout_ms, log, step_timeout_s)
    except BaseException:
        c.close()
        raise


def _prepare(c, port, map_path, timeout_ms, log, step_timeout_s):
    c.set_timeout(timeout_ms)
    c.set_on_step_period(10)
    if c.is_in_menus():
        log(f"  [{port}] en menus, cargando mapa...")
        load_map(c, map_path)
    if step_timeout_s is not None:
        c.sock.settimeout(step_timeout_s)
    reach_start(c)
    if step_timeout_s is not None:
        c.sock.settimeout(None)
    c.set_input()
    return c, c.get_state()


def rewind_checked(c, s0, tries=REWIND_TRIES):
    retries = 0
    diff = None
    for _ in range(tries):
        c.rewind_to_state(s0)
        cur = c.get_state()
        diff = max(vec_diff(cur.position, s0.position), vec_diff(cur.velocity, s0.velocity),
                   float(abs(cur.race_time - s0.race_time)))
        if diff <= POS_TOL:
            return retries, diff
        retries += 1
    return retries, diff


def run_scripted_segment(c, s0, inputs_fn, duration_ms, speed=None, frames=None, check=True, trace_steps=0):
    if check:
        retries, rewind_diff = rewind_checked(c, s0)
    else:
        c.rewind_to_state(s0)
        retries, rewind_diff = 0, None
    if speed is not None:
        c.set_speed(speed)
    c.set_input(**inputs_fn(0))
    t0 = s0.race_time
    trace = []
    frame_ms = []
    next_frame = 0
    n_steps = 0
    t = c.wait_step()
    while True:
        if len(trace) < trace_steps:
            trace.append(t)
        n_steps += 1
        dt = t - t0
        if dt >= duration_ms:
            break
        if frames is not None and dt >= next_frame:
            f0 = time.perf_counter()
            frame = c.capture_frame(*frames["size"])
            frame_ms.append((time.perf_counter() - f0) * 1000)
            if frames.get("on_frame") is not None:
                frames["on_frame"](len(frame_ms) - 1, dt, frame)
            next_frame += frames["every_ms"]
        c.set_input(**inputs_fn(dt))
        t = c.wait_step()
    final = c.get_state()
    return dict(final=final, frame_ms=frame_ms, trace=trace, n_steps=n_steps,
                rewind_retries=retries, rewind_diff=rewind_diff)
