import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.reference_line import build_vcp, lateral_distance, progress


def latest_lap(map_name):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "laps", f"{map_name}_*.npz")))
    full = [f for f in files if "_incomplete" not in f]
    if full:
        return full[-1]
    if files:
        return files[-1]
    return None


def synthetic():
    th = np.linspace(0, 2 * np.pi, 800)
    x = 200 * np.cos(th) + 40 * np.cos(3 * th)
    z = 120 * np.sin(th)
    return np.stack([x, np.full_like(x, 10.0), z], axis=1)


def plot(vcp, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(vcp[:, 0], vcp[:, 2], "-", color="#888", lw=1)
    ax.scatter(vcp[:, 0], vcp[:, 2], s=6, c=np.arange(len(vcp)), cmap="viridis")
    for i in range(0, len(vcp), 20):
        ax.annotate(str(i), (vcp[i, 0], vcp[i, 2]), fontsize=8, color="red")
    ax.plot(vcp[0, 0], vcp[0, 2], "go", ms=9)
    ax.plot(vcp[-1, 0], vcp[-1, 2], "r*", ms=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lap", default=None)
    ap.add_argument("--map-name", default="A01")
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "data", "vcp")
    os.makedirs(out_dir, exist_ok=True)
    if args.dry:
        pos = synthetic()
        name = "dry"
        print("Modo dry: trayectoria sintetica")
    else:
        lap = args.lap or latest_lap(args.map_name)
        if lap is None or not os.path.exists(lap):
            print(f"No hay vueltas grabadas para {args.map_name}. Corre scripts/record_lap.py primero.")
            sys.exit(1)
        print(f"Usando {lap}")
        pos = np.load(lap)["pos"]
        name = args.map_name

    vcp = build_vcp(pos, args.spacing)
    seg = np.linalg.norm(np.diff(vcp.astype(np.float64), axis=0), axis=1)
    print(f"Puntos: {len(vcp)}  largo total: {seg.sum():.1f} m  espaciado medio: {seg.mean():.2f} m")

    idx = 0
    worst = 0.0
    for p in pos[:: max(len(pos) // 400, 1)]:
        idx, _ = progress(p, vcp, idx)
        worst = max(worst, lateral_distance(p, vcp, idx))
    print(f"Chequeo de progreso: indice final {idx}/{len(vcp) - 1}, distancia lateral maxima {worst:.2f} m")

    npy = os.path.join(out_dir, f"{name}.npy")
    png = os.path.join(out_dir, f"{name}.png")
    np.save(npy, vcp)
    plot(vcp, png, f"{name}: {len(vcp)} puntos, {seg.sum():.0f} m")
    print(f"Guardado {npy}")
    print(f"Guardado {png}")


if __name__ == "__main__":
    main()
