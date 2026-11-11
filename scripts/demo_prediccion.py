import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from dreamer_tmnf.load import load_config
import models
import tools

W, H_PX = 1920, 1080
STEP_S = 0.05
BG = (12, 14, 22)
PANEL_BG = (20, 23, 34)
FG = (236, 239, 246)
MUTED = (146, 154, 174)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)
PURPLE = (167, 139, 250)
BLUE = (96, 165, 250)
CYAN = (56, 189, 248)
RED = (248, 113, 113)

PANEL = 680
PX_L = 120
PX_R = 1120
PY = 150
MID_X0 = PX_L + PANEL
MID_W = PX_R - MID_X0
OBS_STEPS = 20
PAUSE_STEPS = 20
TITLE_STEPS = 30
SUMMARY_STEPS = 25

FONT_DIRS = ["C:/Windows/Fonts/"]
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    f = None
    for d in FONT_DIRS:
        for n in names:
            try:
                f = ImageFont.truetype(d + n, size)
                break
            except OSError:
                continue
        if f:
            break
    if f is None:
        f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def fmt_time(t_steps):
    s = t_steps * STEP_S
    m = int(s // 60)
    return f"{m}:{s - 60 * m:05.2f}"


def build_world_model(logdir, device):
    config = load_config([])
    config.num_actions = 9
    config.device = device
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        "vector": gym.spaces.Box(-100, 100, (25,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
    })
    act_space = gym.spaces.Box(0, 1, (9,), dtype=np.float32)
    act_space.discrete = True
    wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
    ck = torch.load(os.path.join(logdir, "latest.pt"), map_location=device, weights_only=False)
    sd = {k[len("_wm."):]: v for k, v in ck["agent_state_dict"].items() if k.startswith("_wm.")}
    wm.load_state_dict(sd, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm, int(ck.get("update_count", -1))


def swap(x):
    return x.permute([1, 0] + list(range(2, len(x.shape))))


def decode(wm, states):
    out = wm.heads["decoder"](wm.dynamics.get_feat(states))
    img = out["image"].mode().clamp(0, 1)
    vec = out["vector"].mode()
    return img, vec


def imagine(wm, init, actions, sample):
    if sample:
        return wm.dynamics.imagine_with_action(actions, init)
    out = tools.static_scan(lambda prev, act: wm.dynamics.img_step(prev, act, sample=False), [swap(actions)], init)
    return {k: swap(v) for k, v in out[0].items()}


def stack_init(post, t0s):
    idx = torch.as_tensor(t0s, device=post["deter"].device)
    return {k: v[0, idx] for k, v in post.items()}


def run_imagination(wm, post, actions, t0s, horizon, sample, chunk=16):
    imgs, speeds = [], []
    for i in range(0, len(t0s), chunk):
        sub = t0s[i:i + chunk]
        init = stack_init(post, sub)
        acts = torch.stack([actions[0, t + 1:t + 1 + horizon] for t in sub])
        states = imagine(wm, init, acts, sample)
        img, vec = decode(wm, states)
        imgs.append(img.cpu().numpy())
        speeds.append(vec[..., 3].cpu().numpy() * 500.0)
    return np.concatenate(imgs), np.concatenate(speeds)


def psnr_from_mse(mse):
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))


def compute_metrics(wm, post, actions, img_f, vel_kmh, T, horizon, every, sample):
    t0s = list(range(1, T - horizon, every))
    mse_m, mse_c, vel_e = [], [], []
    for i in range(0, len(t0s), 16):
        sub = t0s[i:i + 16]
        imgs, speeds = run_imagination(wm, post, actions, sub, horizon, sample, chunk=16)
        for j, t0 in enumerate(sub):
            real = img_f[t0 + 1:t0 + 1 + horizon]
            mse_m.append(((imgs[j] - real) ** 2).mean(axis=(1, 2, 3)))
            mse_c.append(((img_f[t0][None] - real) ** 2).mean(axis=(1, 2, 3)))
            vel_e.append(np.abs(speeds[j] - vel_kmh[t0 + 1:t0 + 1 + horizon]))
    mse_m, mse_c, vel_e = np.array(mse_m), np.array(mse_c), np.array(vel_e)
    return {
        "n": len(t0s),
        "mse_modelo": mse_m.mean(0),
        "psnr_modelo": psnr_from_mse(mse_m).mean(0),
        "mse_copia": mse_c.mean(0),
        "psnr_copia": psnr_from_mse(mse_c).mean(0),
        "err_vel": vel_e.mean(0),
    }


def save_metrics(out, m, horizon):
    path = os.path.join(out, "error_por_horizonte.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "segundos", "mse_modelo", "psnr_modelo", "mse_copia", "psnr_copia", "err_vel_kmh", "n"])
        for h in range(1, horizon + 1):
            k = h - 1
            w.writerow([h, f"{h * STEP_S:.2f}", f"{m['mse_modelo'][k]:.6f}", f"{m['psnr_modelo'][k]:.3f}",
                        f"{m['mse_copia'][k]:.6f}", f"{m['psnr_copia'][k]:.3f}", f"{m['err_vel'][k]:.3f}", m["n"]])
    secs = np.arange(1, horizon + 1) * STEP_S
    plt.rcParams.update({"font.family": "Segoe UI", "text.color": "#ECEFF6", "axes.labelcolor": "#ECEFF6",
                         "xtick.color": "#929AAE", "ytick.color": "#929AAE"})
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="#0C0E16")
    for a in ax:
        a.set_facecolor("#141722")
        for s in a.spines.values():
            s.set_color("#3A4054")
        a.grid(color="#2A3042", lw=0.8)
        a.tick_params(labelsize=13)
    ax[0].plot(secs, m["psnr_modelo"], color="#A78BFA", lw=3.2, label="La IA (imaginación)")
    ax[0].plot(secs, m["psnr_copia"], color="#929AAE", lw=3.0, ls="--", label="Copiar el último cuadro")
    ax[0].set_xlabel("Segundos imaginados", fontsize=15)
    ax[0].set_ylabel("PSNR (dB, más alto es mejor)", fontsize=15)
    ax[0].set_title("Calidad de la imagen predicha", fontsize=18, color="#ECEFF6", pad=12)
    ax[0].legend(fontsize=14, facecolor="#141722", edgecolor="#3A4054", labelcolor="#ECEFF6")
    ax[1].plot(secs, m["err_vel"], color="#38BDF8", lw=3.2)
    ax[1].set_xlabel("Segundos imaginados", fontsize=15)
    ax[1].set_ylabel("Error de velocidad (km/h)", fontsize=15)
    ax[1].set_title("Error al predecir la velocidad", fontsize=18, color="#ECEFF6", pad=12)
    fig.tight_layout(pad=2.0)
    png = os.path.join(out, "error_por_horizonte.png")
    fig.savefig(png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path, png


def metrics_at(m, horizon, secs_list=(0.5, 1, 2, 3)):
    res = {}
    for s in secs_list:
        h = int(round(s / STEP_S))
        if h < 1 or h > horizon:
            continue
        k = h - 1
        res[f"{s}s"] = {
            "mse_modelo": float(m["mse_modelo"][k]), "psnr_modelo": float(m["psnr_modelo"][k]),
            "mse_copia": float(m["mse_copia"][k]), "psnr_copia": float(m["psnr_copia"][k]),
            "err_vel_kmh": float(m["err_vel"][k]),
        }
    return res


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class Frames:
    def __init__(self, real64, hd, interp):
        self.real64 = real64
        self.hd = hd
        self.resample = {"lanczos": Image.LANCZOS, "cubic": Image.BICUBIC, "nearest": Image.NEAREST}[interp]

    def real_panel(self, t, size=PANEL):
        if self.hd is not None:
            im = Image.fromarray(self.hd[t])
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            return im.resize((size, size), Image.LANCZOS)
        return Image.fromarray(self.real64[t]).resize((size, size), self.resample)

    def up(self, arr_u8, size):
        return Image.fromarray(arr_u8).resize((size, size), self.resample)


def inferno_lut():
    cm = matplotlib.colormaps["inferno"]
    return (cm(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


LUT = inferno_lut()


def err_map(imag, real, size):
    e = np.abs(imag.astype(np.float32) - real.astype(np.float32)).mean(axis=2) / 255.0
    idx = np.clip(e / 0.35, 0, 1) * 255
    return Image.fromarray(LUT[idx.astype(np.uint8)]).resize((size, size), Image.BICUBIC)


def text_center(d, cx, y, s, f, fill):
    w = d.textlength(s, font=f)
    d.text((cx - w / 2, y), s, font=f, fill=fill)


def text_right(d, x, y, s, f, fill):
    w = d.textlength(s, font=f)
    d.text((x - w, y), s, font=f, fill=fill)


def base_canvas():
    im = Image.new("RGB", (W, H_PX), BG)
    d = ImageDraw.Draw(im)
    d.text((PX_L, 20), "La IA que sueña con conducir", font=font(46, True), fill=FG)
    text_right(d, PX_R + PANEL, 28, "Momento 2 · La IA predice el futuro", font(34), MUTED)
    d.rectangle([PX_L, 86, PX_R + PANEL, 89], fill=(34, 39, 56))
    d.rectangle([PX_L, 86, PX_L + 260, 89], fill=PURPLE)
    return im


def draw_badge(d, x, y, label, color):
    f = font(30, True)
    w = d.textlength(label, font=f)
    d.rounded_rectangle([x, y, x + w + 44, y + 52], radius=26, fill=color)
    d.text((x + 22, y + 4), label, font=f, fill=(12, 14, 22))


def chart_block(d, im, x0, y0, x1, y1, seg, h_now, phase):
    t_axis = np.arange(-OBS_STEPS, seg["H"] + 1)
    real = seg["real_speed"]
    imag = seg["imag_speed"]
    ymax = max(real.max(), imag.max(), 10) * 1.12
    gx0, gy0, gx1, gy1 = x0 + 62, y0 + 40, x1 - 10, y1 - 34
    d.text((x0, y0 - 2), "Velocidad (km/h)", font=font(24, True), fill=FG)
    lx = x1 - 460
    d.line([lx, y0 + 14, lx + 36, y0 + 14], fill=CYAN, width=5)
    d.text((lx + 44, y0 - 2), "real", font=font(22), fill=MUTED)
    d.line([lx + 130, y0 + 14, lx + 166, y0 + 14], fill=PURPLE, width=5)
    d.text((lx + 174, y0 - 2), "imaginada", font=font(22), fill=MUTED)
    for frac in (0, 0.5, 1.0):
        yy = gy1 - (gy1 - gy0) * frac
        d.line([gx0, yy, gx1, yy], fill=(40, 46, 64), width=1)
        text_right(d, gx0 - 8, yy - 14, f"{ymax * frac:.0f}", font(20), MUTED)

    def px(t):
        return gx0 + (t + OBS_STEPS) / (seg["H"] + OBS_STEPS) * (gx1 - gx0)

    def py(v):
        return gy1 - v / ymax * (gy1 - gy0)

    d.line([px(0), gy0, px(0), gy1], fill=AMBER, width=2)
    d.text((px(0) + 6, gy0 - 4), "pausa", font=font(20), fill=AMBER)
    for sec in range(0, int(seg["H"] * STEP_S) + 1):
        tx = px(sec / STEP_S)
        d.line([tx, gy1, tx, gy1 + 6], fill=MUTED, width=2)
        text_center(d, tx, gy1 + 6, f"{sec}s", font(20), MUTED)
    t_now = h_now
    pts = [(px(t), py(real[t + OBS_STEPS])) for t in range(-OBS_STEPS, t_now + 1)]
    if len(pts) > 1:
        d.line(pts, fill=CYAN, width=5, joint="curve")
    if t_now >= 0:
        pts2 = [(px(t), py(imag[t])) for t in range(0, t_now + 1)]
        if len(pts2) > 1:
            d.line(pts2, fill=PURPLE, width=5, joint="curve")
        if pts2:
            cx, cy = pts2[-1]
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=PURPLE)
    cx, cy = pts[-1]
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=CYAN)


def compose(fr, seg, phase, t_real, imag_u8, h_now, hd_inset=True):
    im = base_canvas()
    d = ImageDraw.Draw(im)
    colors = {"obs": GREEN, "pause": AMBER, "dream": PURPLE, "summary": BLUE}
    labels = {"obs": "OBSERVANDO", "pause": "PAUSA", "dream": "SOÑANDO", "summary": "RESUMEN"}
    col = colors[phase]
    real_im = fr.real_panel(t_real)
    imag_im = fr.up(imag_u8, PANEL)
    d.text((PX_L, 100), "REALIDAD", font=font(34, True), fill=FG)
    right_label = "LO QUE IMAGINA LA IA" if phase in ("dream", "summary") else "LO QUE VE LA IA"
    d.text((PX_R, 100), right_label, font=font(34, True), fill=col if phase != "obs" else GREEN)
    d.rectangle([PX_L - 6, PY - 6, PX_L + PANEL + 5, PY + PANEL + 5], fill=(58, 64, 84))
    d.rectangle([PX_R - 8, PY - 8, PX_R + PANEL + 7, PY + PANEL + 7], fill=col)
    im.paste(real_im, (PX_L, PY))
    if phase == "pause":
        dark = Image.blend(imag_im, Image.new("RGB", imag_im.size, (0, 0, 0)), 0.45)
        im.paste(dark, (PX_R, PY))
    else:
        im.paste(imag_im, (PX_R, PY))
    if fr.hd is not None and hd_inset:
        inset = Image.fromarray(fr.real64[t_real]).resize((176, 176), Image.NEAREST)
        ix, iy = PX_L + 14, PY + PANEL - 176 - 14
        d.rectangle([ix - 3, iy - 3, ix + 179, iy + 179], fill=FG)
        im.paste(inset, (ix, iy))
        lab = "lo que entra a la IA (64×64)"
        f = font(20, True)
        lw = d.textlength(lab, font=f)
        d.rounded_rectangle([ix, iy - 34, ix + lw + 20, iy - 6], radius=8, fill=(12, 14, 22))
        d.text((ix + 10, iy - 34), lab, font=f, fill=FG)
    real_u8 = fr.real64[t_real]
    emap = err_map(imag_u8, real_u8, 280)
    mx = MID_X0 + (MID_W - 280) // 2
    my = PY + 150
    text_center(d, MID_X0 + MID_W / 2, my - 46, "ERROR", font(28, True), FG)
    d.rectangle([mx - 3, my - 3, mx + 283, my + 283], fill=(58, 64, 84))
    im.paste(emap, (mx, my))
    text_center(d, MID_X0 + MID_W / 2, my + 292, "claro = se equivoca", font(22), MUTED)
    cmap_bar = Image.fromarray(np.tile(LUT[None, :, :], (14, 1, 1))).resize((280, 14))
    im.paste(cmap_bar, (mx, my + 330))
    d = ImageDraw.Draw(im)
    if phase == "pause":
        ov = Image.new("RGBA", (W, H_PX), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        by = PY + PANEL // 2 - 74
        od.rectangle([PX_L, by, PX_R + PANEL, by + 148], fill=(10, 12, 20, 215))
        od.rectangle([PX_L, by, PX_R + PANEL, by + 5], fill=AMBER + (255,))
        od.rectangle([PX_L, by + 143, PX_R + PANEL, by + 148], fill=AMBER + (255,))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(im)
        text_center(d, W / 2, by + 16, f"PAUSA — la IA cierra los ojos y sueña los próximos {seg['H'] * STEP_S:.0f} s", font(46, True), AMBER)
        text_center(d, W / 2, by + 84, "solo recibe las acciones que tomó el coche, sin ver nada más", font(32), FG)
        d = ImageDraw.Draw(im)
    sy = 872
    draw_badge(d, PX_L, sy, labels[phase], col)
    horiz = seg["H"] * STEP_S
    if phase == "obs":
        k = t_real - (seg["t0"] - OBS_STEPS) + 1
        clock = f"observando {k * STEP_S:.2f} s de {OBS_STEPS * STEP_S:.2f} s"
        prog = k / OBS_STEPS
    elif phase == "pause":
        clock = "congelado"
        prog = 0.0
    else:
        clock = f"imaginando {h_now * STEP_S:.2f} s de {horiz:.2f} s"
        prog = h_now / seg["H"]
    d.text((PX_L, sy + 66), clock, font=font(32, True), fill=FG)
    d.text((PX_L, sy + 112), f"vuelta en {fmt_time(t_real)}", font=font(24), fill=MUTED)
    bx0, bx1, by0 = PX_L, PX_L + 430, sy + 152
    d.rounded_rectangle([bx0, by0, bx1, by0 + 16], radius=8, fill=(36, 41, 58))
    if prog > 0:
        d.rounded_rectangle([bx0, by0, bx0 + max(16, (bx1 - bx0) * prog), by0 + 16], radius=8, fill=col)
    h_chart = h_now if phase in ("dream", "summary") else (-1 if phase == "obs" else 0)
    if phase == "obs":
        h_chart = -(seg["t0"] - t_real)
    chart_block(d, im, 590, 880, 1360, 1060, seg, h_chart, phase)
    d = ImageDraw.Draw(im)
    rx = 1390
    if phase in ("dream", "obs", "pause"):
        mse = float(((imag_u8.astype(np.float32) - real_u8.astype(np.float32)) ** 2).mean() / 255.0 ** 2)
        d.text((rx, sy - 4), "Error del cuadro", font=font(26, True), fill=FG)
        d.text((rx, sy + 34), f"{psnr_from_mse(mse):.1f} dB", font=font(56, True), fill=col)
        d.text((rx, sy + 108), f"PSNR · MSE {mse:.4f}", font=font(24), fill=MUTED)
        if phase == "dream":
            d.text((rx, sy + 144), f"velocidad real {seg['real_speed'][h_now + OBS_STEPS]:.0f} · IA {seg['imag_speed'][h_now]:.0f} km/h", font=font(22), fill=MUTED)
    else:
        d.text((rx, sy - 4), "Resumen del segmento", font=font(26, True), fill=FG)
        d.text((rx, sy + 30), f"PSNR medio {seg['psnr_mean']:.1f} dB", font=font(34, True), fill=BLUE)
        d.text((rx, sy + 76), f"copiar cuadro {seg['psnr_copy']:.1f} dB", font=font(24), fill=MUTED)
        d.text((rx, sy + 108), f"velocidad final real {seg['real_speed'][-1]:.0f}", font=font(24), fill=CYAN)
        d.text((rx, sy + 138), f"velocidad final IA {seg['imag_speed'][-1]:.0f} km/h", font=font(24), fill=PURPLE)
    return im


def card(lines, thumb=None, extra=None):
    im = Image.new("RGB", (W, H_PX), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 8], fill=PURPLE)
    y = 330 if thumb is None else 250
    for text, size, bold, color in lines:
        text_center(d, W / 2, y, text, font(size, bold), color)
        y += int(size * 1.55)
    if thumb is not None:
        im.paste(thumb, (int(W / 2 - thumb.size[0] / 2), y + 20))
    if extra is not None:
        im.paste(extra, (int((W - extra.size[0]) / 2), 330))
    return im


class VideoWriter:
    def __init__(self, path, fps):
        self.path = path
        self.proc = None
        self.cv = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H_PX}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                stdin=subprocess.PIPE)
            self.codec = "H.264 (ffmpeg)"
        except Exception:
            import cv2
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H_PX))
            self.codec = "mp4v (cv2)"
        self.n = 0

    def write(self, im, repeat=1):
        arr = np.asarray(im.convert("RGB"))
        for _ in range(repeat):
            if self.proc is not None:
                self.proc.stdin.write(arr.tobytes())
            else:
                self.cv.write(arr[:, :, ::-1])
            self.n += 1

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def make_strips(out, fr, starts, imgs_u8, horizon, H):
    os.makedirs(os.path.join(out, "tiras"), exist_ok=True)
    cols = 8
    steps = sorted(set(int(round(x)) for x in np.linspace(horizon / cols, horizon, cols)))
    steps = [max(1, min(horizon, s)) for s in steps]
    cell, lab_w, top = 256, 190, 56
    gap = 6
    rows_names = ["Realidad", "Imaginación"]
    for si, t0 in enumerate(starts):
        w = lab_w + len(steps) * (cell + gap) + gap
        h = top + 2 * (cell + gap) + gap + 40
        im = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(im)
        d.text((16, 10), f"Segmento {si + 1} · vuelta en {fmt_time(t0)}", font=font(28, True), fill=FG)
        for r, name in enumerate(rows_names):
            y = top + gap + r * (cell + gap)
            d.text((16, y + cell / 2 - 18), name, font=font(26, True), fill=FG if r == 0 else PURPLE)
        for c, s in enumerate(steps):
            x = lab_w + gap + c * (cell + gap)
            real_im = fr.real_panel(t0 + s, cell)
            imag_im = fr.up(imgs_u8[si, s - 1], cell)
            im.paste(real_im, (x, top + gap))
            im.paste(imag_im, (x, top + gap + cell + gap))
            text_center(d, x + cell / 2, top + 2 * (cell + gap) + gap, f"+{s * STEP_S:.2f} s", font(24, True), MUTED)
        im.save(os.path.join(out, "tiras", f"segmento_{si + 1:02d}.png"))
    cols_s = [s for s in (10, 20, 40, 60) if s <= horizon] or [horizon]
    cw = 190
    pair_w = 2 * cw + 4
    lab = 150
    w = lab + len(cols_s) * (pair_w + 24) + 10
    h = 64 + len(starts) * (cw + 12) + 10
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    for c, s in enumerate(cols_s):
        x = lab + c * (pair_w + 24)
        text_center(d, x + pair_w / 2, 8, f"+{s * STEP_S:.1f} s", font(26, True), FG)
        d.text((x + 6, 40), "real", font=font(18), fill=MUTED)
        d.text((x + cw + 10, 40), "IA", font=font(18), fill=PURPLE)
    for si, t0 in enumerate(starts):
        y = 64 + si * (cw + 12)
        d.text((10, y + cw / 2 - 26), f"Seg. {si + 1}", font=font(24, True), fill=FG)
        d.text((10, y + cw / 2 + 4), fmt_time(t0), font=font(20), fill=MUTED)
        for c, s in enumerate(cols_s):
            x = lab + c * (pair_w + 24)
            im.paste(fr.real_panel(t0 + s, cw), (x, y))
            im.paste(fr.up(imgs_u8[si, s - 1], cw), (x + cw + 4, y))
    p = os.path.join(out, "tira_resumen.png")
    im.save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join("results", "mejor_vuelta_28.75s.npz"))
    ap.add_argument("--logdir", default=os.path.join("runs", "dreamer", "2026-11-09_startbank"))
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--starts", type=int, nargs="*", default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp", default="lanczos", choices=["lanczos", "cubic", "nearest"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hd", default=None)
    ap.add_argument("--metric-every", type=int, default=5)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    npz_path, logdir = absp(args.npz), absp(args.logdir)
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    out = absp(args.out) if args.out else os.path.join(ROOT, "results", f"prediccion_{stem}")
    os.makedirs(out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    horizon = args.horizon

    d = np.load(npz_path)
    data = {k: d[k] for k in d.files}
    T = len(data["image"])
    hd = None
    if args.hd:
        hd = np.load(absp(args.hd))["frames"]
        assert len(hd) >= T, "los cuadros HD no cubren el episodio"
    if args.starts:
        starts = list(args.starts)
    else:
        starts = [int(round(x)) for x in np.linspace(40, T - horizon - 1, 6)]
    for s in starts:
        assert OBS_STEPS <= s <= T - horizon - 1, f"inicio fuera de rango: {s}"
    print(f"episodio: {T} pasos ({T * STEP_S:.2f} s) · inicios: {starts} · horizonte {horizon} pasos", flush=True)

    wm, update_count = build_world_model(logdir, device)
    print(f"world model cargado (strict=True), update_count={update_count}, dispositivo {device}", flush=True)

    with torch.no_grad():
        batch = {k: data[k][None] for k in ["image", "vector", "action", "is_first", "is_terminal"]}
        batch = wm.preprocess(batch)
        embed = wm.encoder(batch)
        post, _ = wm.dynamics.observe(embed, batch["action"], batch["is_first"])
        rec_img, rec_vec = decode(wm, post)
        rec_img = rec_img[0].cpu().numpy()
        rec_speed = rec_vec[0, :, 3].cpu().numpy() * 500.0
        img_f = data["image"].astype(np.float32) / 255.0
        recon_mse = float(((rec_img - img_f) ** 2).mean())
        print(f"reconstrucción posterior: MSE {recon_mse:.5f} · PSNR {psnr_from_mse(recon_mse):.1f} dB", flush=True)
        vel_kmh = data["vector"][:, 3] * 500.0
        actions = batch["action"]
        imgs, speeds = run_imagination(wm, post, actions, starts, horizon, args.sample)
        print(f"imaginación de {len(starts)} segmentos lista ({time.time() - t_start:.1f} s)", flush=True)
        t_m = time.time()
        m = compute_metrics(wm, post, actions, img_f, vel_kmh, T, horizon, args.metric_every, args.sample)
        print(f"métricas sobre {m['n']} inicios ({time.time() - t_m:.1f} s)", flush=True)

    csv_path, png_path = save_metrics(out, m, horizon)
    at = metrics_at(m, horizon)
    print("\nhorizonte | PSNR modelo | PSNR copia | MSE modelo | MSE copia | err vel km/h")
    for k, v in at.items():
        print(f"{k:>9} | {v['psnr_modelo']:11.2f} | {v['psnr_copia']:10.2f} | {v['mse_modelo']:10.5f} | {v['mse_copia']:9.5f} | {v['err_vel_kmh']:.2f}")

    imgs_u8 = to_u8(imgs)
    rec_u8 = to_u8(rec_img)
    fr = Frames(data["image"], hd, args.interp)
    strip_path = make_strips(out, fr, starts, imgs_u8, horizon, horizon)

    video_path = None
    codec = None
    n_frames = 0
    if not args.no_video:
        video_path = os.path.join(out, "prediccion.mp4")
        vw = VideoWriter(video_path, args.fps)
        codec = vw.codec
        intro = card([
            ("La IA que sueña con conducir", 84, True, FG),
            ("Momento 2 · La IA predice el futuro", 48, False, PURPLE),
            ("Observa la vuelta, cierra los ojos y sueña los próximos segundos.", 34, False, MUTED),
            ("A la izquierda, lo que pasó de verdad. A la derecha, lo que imagina.", 34, False, MUTED),
        ])
        vw.write(intro, int(3.5 * args.fps))
        capdir = os.path.join(out, "capturas")
        os.makedirs(capdir, exist_ok=True)
        saved = set()
        for si, t0 in enumerate(starts):
            Hh = horizon
            seg = {
                "H": Hh, "t0": t0,
                "real_speed": vel_kmh[t0 - OBS_STEPS:t0 + Hh + 1],
                "imag_speed": np.concatenate([[rec_speed[t0]], speeds[si]]),
            }
            mse_seg = ((imgs[si] - img_f[t0 + 1:t0 + 1 + Hh]) ** 2).mean(axis=(1, 2, 3))
            mse_cp = ((img_f[t0][None] - img_f[t0 + 1:t0 + 1 + Hh]) ** 2).mean(axis=(1, 2, 3))
            seg["psnr_mean"] = float(psnr_from_mse(mse_seg).mean())
            seg["psnr_copy"] = float(psnr_from_mse(mse_cp).mean())
            thumb = fr.real_panel(t0, 360)
            tc = card([
                (f"Segmento {si + 1}/{len(starts)}", 80, True, FG),
                (f"{fmt_time(t0)} de la vuelta", 48, False, PURPLE),
            ], thumb=thumb)
            vw.write(tc, TITLE_STEPS * args.fps // 20)
            for t in range(t0 - OBS_STEPS + 1, t0 + 1):
                f = compose(fr, seg, "obs", t, rec_u8[t], t - t0)
                vw.write(f)
                if si == 0 and t == t0 - 8 and "obs" not in saved:
                    f.save(os.path.join(capdir, "fase_observando.png"))
                    saved.add("obs")
            for k in range(PAUSE_STEPS):
                f = compose(fr, seg, "pause", t0, rec_u8[t0], 0)
                vw.write(f)
                if si == 0 and k == PAUSE_STEPS // 2 and "pause" not in saved:
                    f.save(os.path.join(capdir, "fase_pausa.png"))
                    saved.add("pause")
            for h in range(1, Hh + 1):
                f = compose(fr, seg, "dream", t0 + h, imgs_u8[si, h - 1], h)
                vw.write(f)
                if si == 1 and h == Hh // 2 and "dream" not in saved:
                    f.save(os.path.join(capdir, "fase_sonando.png"))
                    saved.add("dream")
            f = compose(fr, seg, "summary", t0 + Hh, imgs_u8[si, Hh - 1], Hh)
            vw.write(f, SUMMARY_STEPS * args.fps // 20)
            if si == 1:
                f.save(os.path.join(capdir, "fase_resumen.png"))
            print(f"segmento {si + 1}/{len(starts)} renderizado", flush=True)
        graf = Image.open(png_path).convert("RGB")
        graf = graf.resize((1500, int(graf.size[1] * 1500 / graf.size[0])), Image.LANCZOS)
        a1 = at.get("1s")
        outro = Image.new("RGB", (W, H_PX), BG)
        od = ImageDraw.Draw(outro)
        od.rectangle([0, 0, W, 8], fill=PURPLE)
        text_center(od, W / 2, 40, "¿Qué tan bien predice?", font(60, True), FG)
        if a1:
            text_center(od, W / 2, 125, f"A 1 s: {a1['psnr_modelo']:.1f} dB contra {a1['psnr_copia']:.1f} dB de copiar el último cuadro", font(34), MUTED)
        outro.paste(graf, (int((W - graf.size[0]) / 2), 200))
        vw.write(outro, 5 * args.fps)
        vw.close()
        n_frames = vw.n
        print(f"video: {video_path} · {n_frames} cuadros · {n_frames / args.fps:.1f} s · {codec}", flush=True)

    meta = {
        "npz": npz_path, "logdir": logdir, "update_count": update_count, "horizon": horizon, "starts": starts,
        "sample": args.sample, "seed": args.seed, "interp": args.interp, "hd": args.hd,
        "metric_every": args.metric_every, "metric_n": m["n"], "recon_mse_posterior": recon_mse,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "video": {"path": video_path, "frames": n_frames, "fps": args.fps, "segundos": n_frames / args.fps if n_frames else 0, "codec": codec},
        "metricas": at,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"\nsalidas en {out}")
    print(f"  csv: {csv_path}\n  png: {png_path}\n  tira resumen: {strip_path}\n  tiras: {os.path.join(out, 'tiras')}")
    print(f"tiempo total {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
