import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmnf.env import THROTTLE_NAMES, STEER_NAMES

DEFAULT = "runs/dreamer/2026-10-29_0"
STEP_S = 0.05
SCALE = 384
PANEL_W = 300


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def act_name(a):
    if a.sum() < 0.5:
        return "-"
    i = int(np.argmax(a))
    return f"{THROTTLE_NAMES[i // 3]}/{STEER_NAMES[i % 3]}"


def render_frame(ep, t, cum):
    img = cv2.resize(ep["image"][t][:, :, ::-1], (SCALE, SCALE), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((SCALE, PANEL_W, 3), np.uint8)
    lines = [
        f"paso   {t}/{len(ep['reward']) - 1}",
        f"tiempo {t * STEP_S:5.2f} s",
        f"accion {act_name(ep['action'][t])}",
        f"km/h   {ep['vector'][t][3] * 500:6.1f}",
        f"r paso {ep['reward'][t]:+.3f}",
        f"r acum {cum[t]:+.3f}",
    ]
    for i, s in enumerate(lines):
        cv2.putText(panel, s, (12, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([img, panel])


def write_video(ep, path):
    cum = np.cumsum(ep["reward"])
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (SCALE + PANEL_W, SCALE))
    for t in range(len(ep["reward"])):
        vw.write(render_frame(ep, t, cum))
    vw.release()


def write_strip(ep, path, title):
    T = len(ep["reward"])
    n = min(T, int(3 / STEP_S))
    idx = np.unique(np.linspace(T - n, T - 1, 16).round().astype(int))
    fig, axes = plt.subplots(4, 4, figsize=(12, 13))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, t in zip(axes.ravel(), idx):
        ax.imshow(ep["image"][t], interpolation="nearest")
        ax.set_title(f"t={t * STEP_S:.2f}s {act_name(ep['action'][t])}\n{ep['vector'][t][3] * 500:.0f} km/h r={ep['reward'][t]:+.2f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def read_csv(logdir):
    p = Path(logdir) / "episodes.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def histogram(files, out):
    sel = files[-50:]
    counts = np.zeros(9)
    for f in sel:
        a = load(f)["action"]
        counts += a.sum(0)
    names = [f"{THROTTLE_NAMES[i // 3]}\n{STEER_NAMES[i % 3]}" for i in range(9)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(9), counts / counts.sum() * 100)
    ax.set_xticks(range(9))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("% de pasos")
    ax.set_title(f"Acciones de los ultimos {len(sel)} episodios")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=DEFAULT)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--best", type=int, default=0)
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    logdir = Path(a.logdir)
    allf = sorted((logdir / "train_eps").glob("*.npz"), key=lambda p: p.name)
    chosen = [Path(f) for f in a.files]
    if a.last:
        chosen += allf[-a.last:]
    if a.best:
        rs = [(float(np.load(f)["reward"].sum()), f) for f in allf]
        rs.sort(key=lambda x: -x[0])
        chosen += [f for _, f in rs[:a.best]]
    seen, uniq = set(), []
    for f in chosen:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    out = logdir / "videos"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(logdir)
    for f in uniq:
        ep = load(f)
        stem = f.stem.split("-")[0] + "_" + f.stem.split("-")[-1]
        info = ""
        try:
            n = allf.index(f)
            if n < len(rows):
                r = rows[n]
                info = f" ep{r['episode']} {r['reason']} {float(r['progress_m']):.0f}m"
        except ValueError:
            pass
        write_video(ep, out / f"{stem}.mp4")
        write_strip(ep, out / f"{stem}.png", f"{stem}{info} R={ep['reward'].sum():.2f}")
        print(out / f"{stem}.mp4", out / f"{stem}.png")
    if allf:
        histogram(allf, out / "action_hist.png")
        print(out / "action_hist.png")


if __name__ == "__main__":
    main()
