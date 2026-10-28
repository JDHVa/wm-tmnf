# Avisos de terceros

## Linesight

- Proyecto: https://github.com/Linesight-RL/linesight
- Autores: Linesight-RL (el plugin `Python_Link.as` es de "Agade")
- Licencia: MIT, segun lo declarado en su `setup.py` (el repositorio no incluye archivo LICENSE).

Se usa como base:

- `tmnf/plugins/Python_Link.as`: copia del plugin de Linesight. Unico cambio: si `custom_port` no esta definido (0), escucha en el puerto 8478 por defecto. El protocolo no cambia.
- `tmnf/tmi_client.py`: cliente de Python adaptado del protocolo de `trackmania_rl/tmi_interaction/tminterface2.py` de Linesight (mismos tipos de mensaje y enteros int32 little-endian nativos).
