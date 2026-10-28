# Benchmark de instancias paralelas

Mide cuantos segundos de carrera simulada por segundo real se obtienen con 1, 2, 3... instancias de TMNF a la vez.

## Como correrlo

```
.venv\Scripts\python scripts\bench_parallel.py --dry
.venv\Scripts\python scripts\bench_parallel.py --max-instances 3 --speed 20
.venv\Scripts\python scripts\bench_parallel.py --max-instances 3 --speed 20 --frames
```

- `--dry`: solo revisa imports, psutil, nvidia-smi, ruta de A01 y puertos escuchando. No toca el juego.
- `--frames`: captura un frame de 160x120 cada 50 ms de carrera.
- `--duration`: segundos reales por etapa (default 20).

## Que hace

Para n = 1..N reutiliza las instancias que ya escuchan desde el puerto 8478 y lanza las que falten en 8479, 8480... con TMLoader. **Se abriran ventanas extra de TMNF** (una por instancia nueva, puede tardar de 30 a 60 s cada una). Si una instancia esta en menus, carga A01. Luego repite el segmento (acelerar 3000 ms + izquierda 500 ms, rewind a S0) en un hilo por instancia y mide aceleracion, determinismo, CPU, RAM y GPU.

Resultados: `runs/bench/parallel.json`. Al terminar deja entradas neutras y velocidad x1, cierra los clientes y **no cierra los juegos** (se imprimen sus puertos y PIDs).

## Notas

- `psutil` es opcional; sin el no se miden CPU ni RAM (se instala en el `.venv`).
- No minimices ni cubras las ventanas de TMNF: puede afectar la captura de frames.
