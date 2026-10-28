import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tmnf.tmi_client import TMIClient


class Attempt:
    def __init__(self):
        self.t = []
        self.pos = []
        self.vel = []
        self.speed = []
        self.inputs = []

    def add(self, t, pos, vel, speed, inp):
        self.t.append(t)
        self.pos.append(pos)
        self.vel.append(vel)
        self.speed.append(speed)
        self.inputs.append(inp)

    def __len__(self):
        return len(self.t)

    def save(self, path, use_inputs):
        d = dict(
            t=np.array(self.t, dtype=np.int32),
            pos=np.array(self.pos, dtype=np.float32),
            vel=np.array(self.vel, dtype=np.float32),
            speed=np.array(self.speed, dtype=np.float32),
        )
        if use_inputs:
            d["inputs"] = np.array(self.inputs)
        np.savez_compressed(path, **d)


def out_path(map_name, suffix=""):
    d = os.path.join(ROOT, "data", "laps")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{map_name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.npz")


def run_dry(args):
    att = Attempt()
    for i in range(500):
        th = i / 500 * 2 * np.pi
        att.add(i * 10, (100 * np.cos(th), 10.0, 60 * np.sin(th)), (0.0, 0.0, 0.0), 100.0, "")
    p = out_path(args.map_name, "_dry")
    att.save(p, False)
    with np.load(p) as d:
        print(f"Dry OK: {p} claves={list(d.keys())} muestras={len(d['t'])}")
    os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--map-name", default="A01")
    ap.add_argument("--no-inputs", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if args.dry:
        run_dry(args)
        return

    use_inputs = not args.no_inputs
    c = TMIClient(args.port)
    print(f"Conectando a 127.0.0.1:{args.port} ...")
    try:
        c.connect(timeout_s=60)
    except OSError:
        print("No se pudo conectar: el juego o el plugin Python_Link no estan listos.")
        sys.exit(1)
    c.set_timeout(3600000)
    c.set_on_step_period(10)

    print("Listo, maneja una vuelta limpia de la pista. Yo solo observo, no toco los controles.")
    print("Si te equivocas, reinicia la carrera (Delete / Enter) y empiezo un intento nuevo. Ctrl+C para cancelar.")

    attempts = []
    cur = Attempt()
    last_t = None
    cp_seen = c.cp_events
    try:
        while True:
            t = c.wait_step()
            if last_t is not None and t < last_t:
                if len(cur) > 0:
                    attempts.append(cur)
                    print(f"Reinicio detectado (intento descartado de {last_t / 1000:.2f} s). Empieza de nuevo cuando quieras.")
                cur = Attempt()
                cp_seen = c.cp_events
            last_t = t
            if t < 0:
                continue
            s = c.get_state()
            inp = c.get_inputs() if use_inputs else ""
            cur.add(t, s.position, s.velocity, s.speed_kmh, inp)

            if c.cp_events != cp_seen:
                cp_seen = c.cp_events
                a, b = c.last_cp
                print(f"  checkpoint {a}/{b}  t={t / 1000:.2f} s")
                if a == b:
                    path = out_path(args.map_name)
                    cur.save(path, use_inputs)
                    print(f"Vuelta terminada: {t / 1000:.3f} s, {len(cur)} muestras.")
                    print(f"Guardada en {path}")
                    break
    except KeyboardInterrupt:
        attempts.append(cur)
        best = max(attempts, key=len)
        if len(best) > 0:
            p = out_path(args.map_name, "_incomplete")
            best.save(p, use_inputs)
            print(f"\nInterrumpido. Intento mas largo ({best.t[-1] / 1000:.2f} s) guardado como {p}")
        else:
            print("\nInterrumpido sin datos.")
    except ConnectionError:
        print("Se perdio la conexion con el juego.")
    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
