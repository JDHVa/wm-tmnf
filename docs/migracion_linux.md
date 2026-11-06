# Migración a Linux

Respaldo hecho el 2026-11-06 desde Windows 11. Nada de esta guía se ha probado en Linux todavía.

## Qué está en git
- Código, plugin, `data/` (vueltas humanas y VCP), documentación y resúmenes.
- De `runs/` (forzado con `git add -f`, porque `runs/` sigue ignorado): CSV, JSONL, configs, logs, gráficas y videos.
- Los episodios 468, 471 y 484 de `runs/dreamer/2026-10-29_ent3e-3_fp10/train_eps`, que hacen falta para `scripts/replay_episode.py`.

## Qué NO está en git
| Qué | Tamaño | Para qué sirve |
|---|---|---|
| `latest.pt` de cada corrida | 228 MB c/u | Reanudar el entrenamiento |
| `train_eps/` de cada corrida | ~1.3 GB c/u | Replay buffer para reanudar |
| `events.out.tfevents.*` | ~335 MB en la corrida 2 | Curvas de TensorBoard |
| `CLAUDE_dont_read_it.md`, `virustotal.png` | pequeños | Archivos locales ignorados |
| Memoria de Claude Code (`~/.claude/projects/.../memory/`) | pequeña | Contexto entre sesiones |

## Volver a clonar `third_party` (sin modificaciones locales)
```
git clone https://github.com/NM512/dreamerv3-torch third_party/dreamerv3-torch
git -C third_party/dreamerv3-torch checkout 6ef8646
git clone https://github.com/Linesight-RL/linesight third_party/linesight
git -C third_party/linesight checkout f6e7822
```

## Puntos que van a fallar en Linux
- `requirements.txt` fija `pywin32` y `dxcam`, que son solo de Windows.
- `torch==2.11.0+cu128` y `torchvision==0.26.0+cu128` necesitan el índice de PyTorch para CUDA 12.8.
- `tmnf/launcher.py` usa `win32gui`, `win32api`, `win32process` y la ruta de `TMLoader.exe` en `AppData`.
- TMNF, TMInterface y TMLoader son programas de Windows: en Linux hay que correrlos con Wine o Proton.
- `config.txt` y `builtin.txt` de TMInterface están en `OneDrive/Documentos/TMInterface`.
