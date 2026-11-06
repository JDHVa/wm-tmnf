# Cambio a Linux

Plan para seguir el proyecto en Linux (dual boot). Windows se queda instalado. Nada de esto se ha probado todavía en Linux. Los comandos suponen Ubuntu o Debian. Más detalle del respaldo en `docs/migracion_linux.md`.

## Estado al salir de Windows (2026-11-06)

- **GitHub (`main`):** código, plugin, `data/`, resúmenes, `scripts/replay_episode.py`, resultados, videos y los episodios 468, 471 y 484.
- **Google Drive (`Mi unidad/wm-tmnf-respaldo/`):** 1 426 archivos, 3.18 GB.
  - `runs/`: copia completa, con `latest.pt`, `train_eps` y TensorBoard.
  - `locales/`: `CLAUDE_dont_read_it.md` y `virustotal.png`.
  - `claude-memory/`: memoria de Claude Code.
- **BitLocker:** `C:` y `D:` están cifrados. No hace falta tocarlos: todo se baja de GitHub y Drive. Conviene tener a mano la clave de recuperación (account.microsoft.com/devices/recoverykey).
- **Entrenamiento:** la corrida activa es `runs/dreamer/2026-11-06_cpgate`, detenida en 210 508 pasos.

## Paso 1. Preparar el sistema

- [ ] Instalar el driver de NVIDIA. La RTX 5050 es Blackwell: necesita un driver reciente.
- [ ] Comprobar con `nvidia-smi`.
- [ ] Instalar Python 3.11, git y rclone.

## Paso 2. Clonar el proyecto

```bash
git clone https://github.com/JDHVa/wm-tmnf.git
cd wm-tmnf
git clone https://github.com/NM512/dreamerv3-torch third_party/dreamerv3-torch
git -C third_party/dreamerv3-torch checkout 6ef8646
git clone https://github.com/Linesight-RL/linesight third_party/linesight
git -C third_party/linesight checkout f6e7822
```

## Paso 3. Bajar lo pesado de Drive

En `rclone config`, crear un remoto de Google Drive llamado `gdrive`.

```bash
rclone copy "gdrive:wm-tmnf-respaldo/runs" runs -P
rclone copy "gdrive:wm-tmnf-respaldo/locales" . -P
```

Comprobar que `runs/` tiene 1 421 archivos: `find runs -type f | wc -l`.

## Paso 4. Crear el entorno

```bash
python3.11 -m venv .venv
grep -v -E '^(pywin32|dxcam)==' requirements.txt > /tmp/req.txt
.venv/bin/pip install -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Paso 5. Restaurar la memoria de Claude Code

La carpeta depende de la ruta del proyecto. Abrir Claude Code una vez dentro de `wm-tmnf` y copiar ahí los archivos:

```bash
rclone copy "gdrive:wm-tmnf-respaldo/claude-memory" ~/.claude/projects/<carpeta-del-proyecto>/memory -P
```

## Paso 6. Decidir dónde corre el juego

TMNF, TMInterface y TMLoader son de Windows. En Linux corren con Wine. Linesight lo soporta, pero su documentación avisa que **el juego va mucho más lento que en Windows** a velocidades altas, y sospecha de `wineserver`. Hay que medirlo antes de mover el entrenamiento.

Prueba propuesta:

- [ ] Instalar `winehq-staging`, TMNF y el ModLoader para Linux que indica Linesight.
- [ ] Instalar OpenAL con Wine si hace falta, y `winetricks dxvk` para rendimiento.
- [ ] Probar `wine TmForever.exe` desde la carpeta del juego.
- [ ] Configurar el perfil con `wine TMLoader.exe`.
- [ ] Probar `wine <ruta>/TMLoader.exe run TmForever "default" /configstring="set custom_port 8479"`.
- [ ] Instalar el plugin `tmnf/plugins/Python_Link.as` en `Documents/TMInterface/Plugins/`, dentro del prefijo de Wine, y copiar `builtin.txt` y `config.txt` si conviene.
- [ ] Revisar que `builtin.txt` no traiga `set "speed" 0`.
- [ ] Medir pasos por segundo con `scripts/bench_tmnf.py` y compararlo con Windows (~43k pasos/h entrenando).

Referencias: `third_party/linesight/docs/source/installation.rst` y `troubleshooting.rst`.

Si Wine va demasiado lento, el plan B es arrancar en Windows para todo lo que use el juego y dejar Linux para lo demás.

## Paso 7. Adaptar el código

Sin esto no conecta con el juego en Linux: `prepare_client` llama a `restore_windows`, que importa `win32gui`.

- [ ] `tmnf/launcher.py`:
  - `restore_windows`, `focus_window` y `dismiss_offline_dialog` usan `win32*`. Sacarlos o hacerlos opcionales fuera de Windows.
  - `tm_pids` busca `tmforever.exe`; revisar el nombre del proceso con Wine.
  - `TMLOADER` y `launch_instance`: lanzar con un script de shell que reciba el puerto, como `linux_launch_game_path` de Linesight.
  - `A01_CANDIDATES` y `find_a01`: rutas del prefijo de Wine.
  - `load_map`: el juego espera una ruta de Windows (`C:\...` o `Z:\...`).
- [ ] `scripts/install_plugin.py`: agregar la ruta dentro del prefijo de Wine.
- [ ] `requirements.txt`: marcar `pywin32` y `dxcam` como solo Windows.

## Paso 8. Retomar el trabajo pendiente

La meta ya quedó resuelta (ver `docs/resumenes/2026-11-06_05.md`): el agente se saltaba el CP2. Sigue en este orden, en el sistema donde funcione el juego:

1. Ver por qué el agente se sale después del CP2, entre 1 800 y 2 100 m.
2. Reapertura automática del juego. Se cierra solo tras ~1–1.3 h.
3. `ensure_game` no carga la A01: copiar el flujo de Linesight.
4. Seguir entrenando `runs/dreamer/2026-11-06_cpgate`.

Mientras tanto, sin el juego se puede avanzar en las visualizaciones de los momentos 1 y 2 y en el renderer de difusión.

## Prompt para el chat nuevo

```text
Lee CLAUDE.md, linux.md y el último resumen en docs/resumenes/2026-11-06_05.md. Me cambié a Linux (dual boot). El proyecto ya se clonó y estoy en el paso <N> de linux.md. Continúa desde ahí.
```
