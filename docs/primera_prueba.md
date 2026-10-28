# Primera prueba: Python maneja el coche

## Antes de empezar (una sola vez)
El plugin ya esta copiado en `C:\Users\jesus\OneDrive\Documentos\TMInterface\Plugins\Python_Link.as`
(tu carpeta Documentos esta en OneDrive). Si lo modificas en `tmnf/plugins/`, vuelve a copiarlo con:

    .venv\Scripts\python.exe scripts\install_plugin.py

## Pasos
1. Abre el juego con TMLoader (perfil `default`). Debe aparecer la consola de TMInterface.
   Como el plugin se acaba de copiar, hay que cargarlo: **cierra y vuelve a abrir el juego**
   (los plugins de la carpeta `Plugins` se cargan al iniciar). Si el juego ya estaba abierto,
   en el menu superior de TMInterface busca la seccion **Plugins** y recarga ("Reload plugins");
   el nombre exacto del boton puede variar en tu version.
2. Comprueba en la consola/log de TMInterface el mensaje
   `Python_Link listening on port 8478`. Si sale, el plugin esta activo.
3. Inicia sesion con tu perfil y carga una pista cualquiera (Solo > Local > cualquier mapa, p. ej. A01).
   Deja la ventana del juego visible y con el foco.
4. Con la pista en pantalla (coche en la salida, o en la cuenta atras), ejecuta en otra terminal:

       .venv\Scripts\python.exe scripts\test_connection.py

5. Veras ~5 lineas por segundo con velocidad, posicion y tiempo; luego el coche acelera 3 s,
   gira a la izquierda 1 s, a la derecha 1 s y suelta todo. Al final dice "Desconectado".

El script se puede lanzar antes de cargar la pista: espera hasta 60 s a que empiece la carrera.
Para probar sin juego: `scripts\test_connection.py --dry`.

## Problemas comunes
- "No se pudo conectar": el plugin no esta cargado (paso 1-2) o el puerto es otro (`--port`).
- Tras ejecutar, el juego queda esperando si el script se corta mal; el plugin nos desconecta solo tras 10 s.
- Pausa el juego solo en cada paso de simulacion: eso es normal (es lo que permite controlarlo de forma exacta).
