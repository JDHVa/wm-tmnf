# TrackMania World Model: "La IA que sueña con conducir"

Agente de IA con **world model** que aprende a conducir en **TrackMania Nations Forever (TMNF)**. La IA aprende cómo funciona el mundo del juego, aprende a conducir dentro de su propio "sueño" y luego conduce en el juego real.

## Objetivo

Presentarlo en una **conferencia de IA en Tecmilenio** ante estudiantes, en aproximadamente un mes (fecha límite aprox.: finales de noviembre de 2026). **La prioridad es que se vea impresionante**: cada parte técnica tiene que producir algo visual que se pueda enseñar. La presentación (slides y guion) se hace al final, no ahora.

## Decisiones tomadas

- **Juego: TrackMania Nations Forever** (gratis, ligero, determinista, con editor de pistas).
  - Se descartó **Assetto Corsa**: corre a tiempo real, solo es práctica una instancia y no tiene un editor de pistas fácil.
  - Se descartó **Trackmania (2020)**: es más pesado y no se puede acelerar.
- **Integración con el juego: igual que Linesight** (la IA de referencia en TMNF, que batió 10 de 12 récords mundiales de la campaña): **TMInterface 2.1.0 + ModLoader + un plugin AngelScript** (`Python_Link.as`, en `Documents/TMInterface/Plugins/`) que habla con Python por socket. Cada instancia usa su propio puerto (8478, 8479, …). Juego en ventana, con la resolución y los gráficos al mínimo. Pide unos 20 GB de RAM y Python entre 3.10 y 3.11. Se puede tomar el código de conexión de Linesight como base.
  - El cliente viejo de Python `tminterface` (para TMI 1.x) expone `set_speed`, `set_input_state`, `get_simulation_state` / `rewind_to_state`, `respawn` y `on_run_step`. Sirve como referencia de qué permite la API. Las velocidades muy altas pueden saltarse los inputs: hay que medirlo.
- **Sueño de calidad (dos capas):** Dreamer es el "cerebro" (latente compacto, barato, para planear). Para la demo visual se entrena aparte un **renderer de difusión estilo DIAMOND** a una resolución mayor, que se usa para mostrar lo que la IA imagina y para el sueño jugable. El de 64×64 de Dreamer queda como plan B. (Como referencia, DIAMOND en CS:GO tardó 12 días en una RTX 4090: aquí se hace a mucha menor escala, con pocas pistas y estilo visual simple.)
- **Observaciones: píxeles + telemetría.** Los píxeles hacen falta para mostrar lo que la IA ve e imagina.
- **World model: DreamerV3 en PyTorch** (no el oficial en JAX, porque JAX no tiene GPU nativa en Windows).
- **Campo de pruebas: Gymnasium CarRacing**, para validar el pipeline rápido antes de conectarlo a TMNF.

## Cómo aprende a conducir

- **Recompensa:** avance a lo largo de una línea de "checkpoints virtuales" (un punto cada ~10 m), sacada de una vuelta grabada por un humano. Se penaliza quedarse atorado y se da un bono por terminar.
- **Fin del episodio:** se atora (velocidad ≈ 0 durante N s), se aleja demasiado de la línea, se acaba el tiempo o cruza la meta. El reset se hace con `rewind_to_state` (es instantáneo).
- **Observación:** imagen de 64×64 + vector (velocidad, distancia y ángulo hacia los siguientes checkpoints).
- **Acción:** se empieza con acciones discretas (dirección × acelerar/frenar). Lo analógico va después.
- **Arranque:** vueltas humanas con el control de Xbox + datos aleatorios para iniciar el world model. Después, Dreamer online con varias instancias.
- **Currículum:** una pista corta y fácil hecha en el editor → pistas de A01 → pistas difíciles o creadas por el público.

## Momentos de la demo (lo que se tiene que poder mostrar)

1. **Así ve la IA:** el juego real, la entrada de 64×64 y la reconstrucción desde el latente, lado a lado.
2. **Predice el futuro:** se pausa el juego, el modelo imagina N segundos y se compara con la realidad (se puede repetir exactamente gracias al determinismo y a guardar/cargar estados).
3. ⭐ **Sueño jugable:** una persona maneja con el control de Xbox *dentro del world model*.
4. ⭐ **Fantasmas imaginados:** los futuros que imagina la IA se dibujan como un abanico de coches fantasma, y la IA elige el mejor.
5. **Pista nueva:** alguien del público crea una pista en el editor, la IA la imagina y luego la recorre.
6. **Evolución:** timelapse del entrenamiento y carrera contra el fantasma de un humano.

## Hardware y limitaciones

- Python **3.11** (Linesight pide 3.10–3.11), venv en `.venv`.
- Laptop Lenovo LOQ: **RTX 5050 Laptop (8 GB de VRAM, Blackwell)**, i5 de 13.ª gen, 24 GB de RAM, Windows 11.
- **PyTorch tiene que estar compilado con CUDA 12.8 o más** (Blackwell no funciona con versiones anteriores).
- 8 GB de VRAM: modelos de Dreamer de unos 12 a 50 M de parámetros, con imágenes de 64×64. Nada de world models de difusión grandes.
- El sueño a 64×64 se verá borroso. Es aceptable y se puede contar como parte de la historia. Mejorar la resolución solo para mostrarlo es opcional.
- Laptop: hay *thermal throttling* en entrenamientos largos (conectada a la corriente, modo Performance en Vantage).
- Paralelismo esperado: 2 a 4 instancias de TMNF (*por medir*).

## Plan aproximado (4 semanas)

| Semana | Objetivo |
|---|---|
| 1 | Setup (PyTorch + TMNF + TMInterface), conexión en Python (captura, telemetría, inputs), entorno Gymnasium, prototipo en CarRacing |
| 2 | Recolección de datos + entrenar el world model → momentos 1 y 2 |
| 3 | Sueño jugable + entrenar el agente → momentos 3 y 4 |
| 4 | Afinar el agente, pistas nuevas, pulir el dashboard → momentos 5 y 6 |

Orden pensado para reducir riesgo: aunque el agente no quede bien, las demos del world model (1 a 3) ya funcionan solas.

## Stack

Python 3.11 · PyTorch (CUDA 12.8+) · DreamerV3 (port en PyTorch: base `NM512/dreamerv3-torch`) · Gymnasium · TMInterface · captura de pantalla (dxcam o similar) · FastAPI + WebSocket para el dashboard · TensorBoard / W&B.

## Paper (después de la conferencia)

Meta: preprint en arXiv y después un workshop (NeurIPS/ICML/ICLR) o un congreso (MICAI, COMIA). El autor es estudiante de la **UANL** (la conferencia es en Tecmilenio, pero la afiliación es UANL). Asesor y coautor: un profesor (doctor) de la UANL, que no tiene publicaciones en arXiv. arXiv es opcional: se puede publicar directamente en un congreso o workshop.

**Preguntas de investigación:**
1. **Eficiencia de muestras:** ¿un agente con world model (DreamerV3) aprende a conducir en TMNF con menos horas de juego que un agente sin world model (Linesight / IQN)? Métrica: tiempo de vuelta en función de los pasos y las horas de interacción.
2. **Generalización:** ¿el agente y su world model se transfieren a pistas nuevas que nunca vio (creadas en el editor)? Métricas: tasa de terminar la pista, tiempo de vuelta y error de predicción del world model en pistas vistas vs. no vistas.
- Bonus: visualizaciones del sueño y error de predicción a N pasos (se puede medir con precisión gracias al determinismo de TMNF).

**Reglas para que el código sirva al paper desde el inicio:**
- Registrar siempre: pasos del entorno, horas de juego simuladas y reales, pasos de gradiente, tiempos de vuelta, checkpoints alcanzados y error de predicción del world model.
- Toda corrida lleva su semilla y su config guardadas junto a los resultados (reproducible).
- Mínimo 3 semillas por experimento final.
- **Separación fija de pistas:** pistas de entrenamiento y pistas de prueba (nunca vistas), definidas antes de entrenar y que no se tocan después.
- Línea base: Linesight entrenado en las mismas pistas, con el mismo presupuesto de interacción.
- Guardar los modelos entrenados y los videos de cada experimento.

## Reparto de trabajo entre modelos

- **Sonnet 5.5** (`claude-sonnet-5-5`) **escribe el código** (se lanza como subagente con `model: "sonnet"`).
- **Opus 5.5** (`claude-opus-5-5`) **revisa** el código que escribe Sonnet, toma las decisiones de arquitectura y coordina.
- **Haiku 5.5** (`claude-haiku-5-5`) **instala dependencias** (venv, pip, herramientas; se lanza como subagente con `model: "haiku"`).
- Todas las dependencias van en el venv del proyecto (`.venv`), nunca globales.

## Notas

- `CLAUDE_dont_read_it.md` es de otro contexto: no leerlo.
- Idioma de trabajo con el usuario: español.
- No agregues comentarios en el codigo solo codigo
- **Cada 20 mensajes del usuario**, escribir un resumen de la conversación en `docs/resumenes/AAAA-MM-DD_NN.md`, para poder empezar un chat nuevo de Claude Code sin perder contexto. Debe incluir: qué se decidió, qué se hizo (archivos), resultados de pruebas, problemas abiertos y siguiente paso. Avisar al usuario cuando se escriba y **darle un prompt listo para copiar y pegar** en el chat nuevo. El prompt debe pedir leer `CLAUDE.md` y el último resumen, y decir en qué paso exacto se continúa.