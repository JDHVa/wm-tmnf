"""Copia tmnf/plugins/Python_Link.as a la carpeta Plugins de TMInterface (detecta OneDrive)."""
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "tmnf" / "plugins" / "Python_Link.as"
HOME = Path.home()
CANDIDATES = [
    HOME / "Documents" / "TMInterface",
    HOME / "OneDrive" / "Documents" / "TMInterface",
    HOME / "OneDrive" / "Documentos" / "TMInterface",
    HOME / "Documentos" / "TMInterface",
]


def main():
    found = [c for c in CANDIDATES if c.is_dir()]
    if not found:
        print("No encontre la carpeta TMInterface. Abre el juego una vez con TMLoader y reintenta.")
        print("Busque en:", *CANDIDATES, sep="\n  ")
        sys.exit(1)
    tmi = found[0]
    if len(found) > 1:
        print("Aviso: hay varias carpetas TMInterface:", *found, sep="\n  ")
    dst_dir = tmi / "Plugins"
    dst_dir.mkdir(exist_ok=True)
    shutil.copy2(SRC, dst_dir / SRC.name)
    print(f"Plugin copiado a: {dst_dir / SRC.name}")


if __name__ == "__main__":
    main()
