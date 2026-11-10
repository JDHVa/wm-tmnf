import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
f = lambda x: float(x) if x not in ("", None) else None
full = [r for r in rows if not f(r["start_m"])]
h = np.array([f(r["wall_hours"]) for r in full])
p = np.array([f(r["progress_m"]) for r in full])
fin = np.array([r["reason"] == "finish" for r in full])
lt = np.array([f(r["lap_time_s"]) if r["reason"] == "finish" else np.nan for r in full])
ep = np.array([int(r["episode"]) for r in full])

def roll(a, w):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        out[i] = np.nanmedian(a[max(0, i - w + 1):i + 1]) if a.dtype != bool else a[max(0, i - w + 1):i + 1].mean()
    return out

blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
                     "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 0.8, 1]})
changes = [(533, "recompensa por CP"), (680, "banco de estados")]
xs = [h[np.searchsorted(ep, e)] for e, _ in changes]
for a in ax:
    a.set_facecolor("#fcfcfb")
    a.grid(axis="y", color=grid, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    for x in xs:
        a.axvline(x, color=muted, lw=1, ls=":")
for x, (_, name), ha in zip(xs, changes, ("right", "left")):
    ax[0].text(x + (-0.08 if ha == "right" else 0.08), 2330, name, color=muted, fontsize=9, va="bottom", ha=ha)

ax[0].scatter(h, p, s=8, color=blue, alpha=0.25, lw=0)
ax[0].plot(h, roll(p, 25), color=blue, lw=2)
for y, name in ((1198, "CP1"), (1776, "CP2"), (2260, "meta")):
    ax[0].axhline(y, color=muted, lw=0.8, ls="--")
    ax[0].text(h[0], y + 25, name, color=muted, fontsize=9)
ax[0].set_ylim(0, 2450)
ax[0].set_ylabel("metros")
ax[0].set_title("Avance por vuelta desde la salida (puntos) y mediana de 25 vueltas (línea)")

ax[1].plot(h, 100 * roll(fin, 30), color=aqua, lw=2)
ax[1].set_ylim(-3, 103)
ax[1].set_ylabel("%")
ax[1].set_title("Vueltas terminadas (últimas 30 desde la salida)")

m = ~np.isnan(lt)
ax[2].scatter(h[m], lt[m], s=22, color=orange, lw=0)
best = np.fmin.accumulate(np.where(m, lt, np.inf))
ok = np.isfinite(best)
ax[2].step(h[ok], best[ok], where="post", color=orange, lw=2)
ax[2].axhline(28.75, color=muted, lw=1, ls="--")
ax[2].text(h[ok][0], 28.75 - 0.6, "tu mejor vuelta: 28.75 s", color=muted, fontsize=9, va="top")
ax[2].set_ylabel("segundos")
ax[2].set_title("Tiempo de vuelta de cada meta (puntos) y mejor hasta el momento (línea)")
ax[2].set_xlabel("horas de entrenamiento (reloj real, acumuladas)")
lo = np.nanmin(lt) if m.any() else 28
ax[2].set_ylim(min(lo, 28.75) - 1.5, np.nanmax(lt) + 1 if m.any() else 40)
fig.tight_layout()
fig.savefig(out, dpi=130, facecolor="#fcfcfb")
print(len(full), int(fin.sum()), np.nanmin(lt))
