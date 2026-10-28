import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmnf.tmi_client import TMIClient

PHASES = [
    ("Observando (sin tocar nada)", 2000, 0, 0, 0),
    ("Acelerando", 3000, 0, 0, 1),
    ("Girando a la IZQUIERDA (acelerando)", 1000, 1, 0, 1),
    ("Girando a la DERECHA (acelerando)", 1000, 0, 1, 1),
    ("Soltando todo", 1000, 0, 0, 0),
]
PRINT_EVERY_MS = 200


def show(c, label=""):
    s = c.get_state()
    x, y, z = s.position
    print(f"[t={s.race_time/1000:6.2f}s] vel={s.speed_kmh:5.0f} km/h  pos=({x:7.1f},{y:6.1f},{z:7.1f}) {label}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8478)
    ap.add_argument("--dry", action="store_true", help="solo verifica imports, no conecta")
    args = ap.parse_args()
    if args.dry:
        print("Imports OK, fases:", [p[0] for p in PHASES])
        return

    c = TMIClient(args.port)
    print(f"Conectando a 127.0.0.1:{args.port} ...")
    try:
        c.connect(timeout_s=60)
    except ConnectionRefusedError:
        print("No se pudo conectar: el plugin Python_Link no esta escuchando. Revisa docs/primera_prueba.md")
        sys.exit(1)
    print("Conectado. Esperando que estes dentro de una carrera (mapa cargado)...")
    try:
        c.set_timeout(10000)
        c.set_on_step_period(10)
        t = c.wait_step()
        print("Carrera detectada. Cuenta atras / arranque...")
        while t < 0:
            t = c.wait_step()
        start = t
        last_print = -PRINT_EVERY_MS
        for name, dur, l, r, a in PHASES:
            print(f"--- {name} ---")
            end = t + dur
            while t < end:
                c.set_input(left=l, right=r, accelerate=a)
                if t - last_print >= PRINT_EVERY_MS:
                    show(c)
                    last_print = t
                t = c.wait_step()
        c.set_input()
        show(c, "(final)")
        print("Prueba terminada.")
    except socket.timeout:
        print("Timeout: no llego ningun paso de simulacion. Carga un mapa y pon el coche en carrera.")
    finally:
        try:
            c.set_input()
        except OSError:
            pass
        c.close()
        print("Desconectado.")


if __name__ == "__main__":
    main()
