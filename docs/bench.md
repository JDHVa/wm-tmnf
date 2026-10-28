# Benchmark de TMNF (rewind, velocidad, capturas)

Mide cuatro cosas con el juego abierto en un mapa (p. ej. A01), el plugin Python_Link cargado y el coche en la salida o en la cuenta atras:

1. **Rewind y determinismo:** guarda el estado S0 al inicio de la carrera, acelera 2 s, hace rewind a S0 y compara posicion y tiempo (3 veces). La misma secuencia de inputs debe dar la misma posicion final.
2. **Velocidades** x1, x2, x5, x10, x20: ejecuta la misma secuencia (3000 ms acelerando y 500 ms girando a la izquierda), mide el tiempo real y la aceleracion efectiva. Si la posicion final difiere de la de x1 (mas de 1 mm), esa velocidad se considera no determinista.
3. **Capturas:** pide frames de 256x256, 160x120 (el tamano de Linesight) y 64x64 a x1 y a la velocidad determinista mas alta. Guarda PNG en `runs/bench/frames/` (10 por tamano guardado, mas una version 64x64 reducida desde 256x256) y mide los ms por captura y como afectan a la aceleracion (una captura cada 50 ms de carrera).
4. **Resumen:** `runs/bench/summary.json` y una tabla en pantalla.

## Uso

    .venv\Scripts\python.exe scripts\bench_tmnf.py [--port 8478] [--dry]

- `--dry` solo verifica imports.
- Deja la ventana del juego visible y no la minimices (las capturas salen de lo que renderiza el juego).
- No toques el teclado ni el control durante la prueba. Dura unos minutos.
- Si el coche se estrella durante la secuencia no pasa nada: se compara contra corridas con los mismos inputs.

## Notas del protocolo de frames

- `request_frame(w, h)` pide UNA captura: el plugin la envia en su siguiente `Render()` como `SC_REQUESTED_FRAME_SYNC` + `w*h*4` bytes y espera la respuesta del cliente.
- Formato: BGRA, uint8, filas de arriba a abajo (Linesight lo convierte con `cv2.COLOR_BGRA2GRAY` sin voltear). Si las PNG salen al reves, hay que anadir `cv2.flip`.
- `TMIClient.capture_frame(w, h)` pone la velocidad en 0, responde al paso pendiente, recibe el frame, responde y restaura la velocidad.
