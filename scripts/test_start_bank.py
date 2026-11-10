import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from tmnf.env import TMNFEnv
from tmnf.fake_client import FakeClient
from tmnf.start_bank import StartBank

VCP = ROOT / "data" / "vcp" / "A01.npy"


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="startbank_"))
    bank = StartBank(bucket_m=50.0, per_bucket=4, frontier=3, seed=1)
    env = TMNFEnv(client=FakeClient(str(VCP)), start_bank=bank, start_prob=0.5, snap_every_m=10.0,
                  snap_confirm_m=20.0, start_seed=3, max_lateral_m=1e9, max_time_ms=8000, cp_path="")
    gas = 4
    episodes = []
    checks = 0
    steps = 0
    obs, info = env.reset()
    ep_start = info["start_m"]
    while steps < 1200:
        obs, r, term, trunc, info = env.step(gas)
        steps += 1
        if term or trunc:
            episodes.append((ep_start, info["progress_m"], info["reason"]))
            obs, info = env.reset()
            ep_start = info["start_m"]
            if ep_start > 0:
                entry = next(e for b in bank.buckets.values() for e in b if e["progress_m"] == ep_start)
                st = env.client.get_state()
                saved = env.client.state_from_bytes(entry["data"])
                assert np.allclose(st.position, saved.position), (st.position, saved.position)
                assert env.last_idx == entry["last_idx"]
                assert abs(env.max_progress - entry["max_progress"]) < 1e-9
                assert abs(env.max_progress * env.seg_m - ep_start) < 1e-6
                assert env.t_start == saved.race_time
                checks += 1
    banked = [e for e in episodes if e[0] > 0]
    print("pasos", steps, "episodios", len(episodes), "desde banco", len(banked), "verificados", checks)
    print("banco", len(bank), bank.summary())
    print("ejemplos", episodes[:4], banked[:4])
    assert len(bank) > 0
    assert banked
    assert checks >= len(banked) > 0
    assert all(e[1] >= e[0] - 1e-6 for e in episodes)
    path = tmp / "bank.pkl"
    bank.save(path)
    assert not (tmp / "bank.pkl.tmp").exists()
    b2 = StartBank.load(path, bucket_m=50.0)
    assert b2.summary() == bank.summary()
    assert len(StartBank.load(tmp / "nada.pkl")) == 0
    e = b2.sample()
    assert b2.remove(e) and len(b2) == len(bank) - 1
    print("OK")


if __name__ == "__main__":
    main()
