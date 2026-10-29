# Benchmark DreamerV3 (PyTorch) en TMNF

Modelo: dyn_deter 512, dyn_hidden 512, units 512, 32x32 discreto, CNN depth 32, encoder/decoder MLP 2x256 para "vector". 19.78 M parametros totales (16.4 M world model, 1.06 M actor, 1.18 M critico). Batch 16 x 64 = 1024 pasos por gradiente.

## Paso de gradiente (sin entorno, scripts/bench_train_step.py, RTX 5050 Laptop)

| Config | grad/s | VRAM pico |
|---|---|---|
| batch 16x64, fp32 | 1.35 | 3.8 GB |
| batch 16x64, fp16 | 1.13 | 3.8 GB |
| batch 16x64, TF32 | 1.36 | 3.8 GB |
| batch 16x64, validate_args off | 1.40 | 3.8 GB |
| batch 16x32 | 2.75 | 2.1 GB |
| batch 32x32 | 1.97 | 3.8 GB |

Rendimiento ~1.4 mil pasos repetidos por segundo. El perfil muestra que casi todo es backward (autograd) y overhead de torch.distributions; torch.compile no existe en Windows.

## Entorno

~290 pasos/s a velocidad 20 sin entrenar (random prefill). Con entrenamiento, la GPU es el cuello de botella: pasos_entorno/s = grad/s * 1024 / train_ratio.

| train_ratio | pasos de entorno/s | pasos/hora |
|---|---|---|
| 512 | 2.8 | 10 k |
| 256 | 5.6 | 20 k |
| 128 | 11 | 40 k |
| 64 | 22 | 80 k |

Elegido: train_ratio 128 (1e6 pasos tomarian ~25 h; el entorno jamas es el limite).

## Pausas

Cada paso de gradiente (~0.7 s) bloquea el entorno. Con ratio 128 hay un gradiente cada 8 pasos de entorno. El pretrain (config.pretrain) bloquea el juego durante pretrain/1.4 s (100 -> ~70 s, bajado a 10 -> ~7 s). El plugin recibe set_timeout de 900 s.

## Pendiente

Medicion integrada (entorno + entrenamiento) no completada: el juego dejo de responder y se cerro tres veces justo al pasar de prefill a entrenamiento (tras 2445 pasos, durante la construccion del agente/pretrain de ~70 s sin pasos de entorno). Ver reporte.
