import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.episode import prepare_client, rewind_checked
from tmnf.launcher import find_a01
from tmnf.reference_line import progress, lateral_distance
from tmnf.tmi_client import MsgType

lap = sorted(glob.glob(os.path.join(ROOT, "data", "laps", "A01_*.npz")))[int(sys.argv[1])]
stride = int(sys.argv[2])
d = np.load(lap, allow_pickle=True)
P, V = d["pos"].astype(float), d["vel"].astype(float)
vcp = np.load(os.path.join(ROOT, "data", "vcp", "A01.npy")).astype(float)
print(lap, len(P))
c, s0 = prepare_client(8479, find_a01(), timeout_ms=900000, step_timeout_s=30, connect_timeout_s=30)
c.sock.settimeout(60)
ev = []
orig = c._handle_other
cur = [0]
def wrapped(m):
    orig(m)
    if m == MsgType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
        ev.append((cur[0], c.last_cp))
c._handle_other = wrapped
try:
    rewind_checked(c, s0)
    c.set_speed(5)
    c.set_input(accelerate=True)
    c.wait_step()
    last = 0
    seen = 0
    res = []
    for k in range(0, len(P), stride):
        cur[0] = k
        st = c.get_state()
        st.raw.position = [float(v) for v in P[k]]
        st.raw.dyna.current_state.linear_speed = np.array(V[k], dtype=np.float32)
        c.rewind_to_state(st)
        t = c.wait_step()
        idx, p = progress(P[k], vcp, last)
        last = idx
        if k % 500 == 0:
            g = c.get_state()
            print(k, t, "pos", tuple(round(x, 1) for x in g.position), "objetivo", tuple(round(x, 1) for x in P[k]), "v", round(g.speed_kmh))
        if len(ev) > seen:
            seen = len(ev)
            g = c.get_state()
            print("EVENTO", ev[-1], "t", t, "pos", tuple(round(x, 1) for x in g.position), "p", round(p, 2), "lat", round(lateral_distance(P[k], vcp, idx), 2))
            res.append((ev[-1][1], p))
            if ev[-1][1][0] == ev[-1][1][1]:
                break
    print("res", res)
    g = c.get_state()
    print("race_finished", c.race_finished(), "cp_states", [int(bool(g.raw.cp_data.cp_states[i])) for i in range(int(g.raw.cp_data.cp_states_length))])
    print("rewind", rewind_checked(c, s0))
    c.set_input(accelerate=True)
    t0 = c.wait_step()
    for _ in range(100):
        t = c.wait_step()
    g = c.get_state()
    print("reinicio t", t0, t, "pos", tuple(round(x, 1) for x in g.position))
finally:
    try:
        c.set_input(); c.set_speed(1); c.unrequest_frame()
    except OSError:
        pass
    c.close()
