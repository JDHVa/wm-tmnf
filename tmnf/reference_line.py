import numpy as np


def build_vcp(positions, spacing=10.0):
    p = np.asarray(positions, dtype=np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6])
    p = p[keep]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.floor(s[-1] / spacing)) + 1, 2)
    targets = np.arange(n) * spacing
    if s[-1] - targets[-1] > 1e-6:
        targets = np.append(targets, s[-1])
    out = np.stack([np.interp(targets, s, p[:, k]) for k in range(3)], axis=1)
    return out.astype(np.float32)


def _project(pos, vcp, i):
    a = vcp[i].astype(np.float64)
    ab = vcp[i + 1].astype(np.float64) - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-12 else min(max(float((pos - a) @ ab) / L2, 0.0), 1.0)
    return t, float(np.linalg.norm(pos - (a + t * ab)))


def progress(pos, vcp, last_idx, window=15, back=2):
    pos = np.asarray(pos, dtype=np.float64)
    lo = max(last_idx - back, 0)
    hi = min(last_idx + window, len(vcp) - 1)
    best_i, best_t, best_d = lo, 0.0, np.inf
    for i in range(lo, max(hi, lo + 1)):
        if i + 1 >= len(vcp):
            break
        t, d = _project(pos, vcp, i)
        if d < best_d:
            best_i, best_t, best_d = i, t, d
    return best_i, best_i + best_t


def lateral_distance(pos, vcp, idx):
    pos = np.asarray(pos, dtype=np.float64)
    i = int(min(max(idx, 0), len(vcp) - 2))
    return _project(pos, vcp, i)[1]
