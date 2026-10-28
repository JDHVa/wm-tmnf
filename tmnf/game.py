import socket
import time

from .episode import prepare_client
from .launcher import find_a01, launch_instance, port_is_listening, restore_windows, wait_for_port

STEP_TIMEOUT_S = 30
MAX_RETRIES = 2


def listener_pid(port):
    import psutil
    for c in psutil.net_connections(kind="tcp4"):
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
            return c.pid
    return None


def kill_listener(port, log=print):
    import psutil
    pid = listener_pid(port)
    if pid is None:
        return None
    log(f"  [{port}] matando proceso colgado pid={pid}")
    try:
        p = psutil.Process(pid)
        p.kill()
        p.wait(timeout=15)
    except psutil.NoSuchProcess:
        pass
    t0 = time.time()
    while time.time() - t0 < 20 and port_is_listening(port):
        time.sleep(0.5)
    return pid


def start_if_needed(port, timeout_s, log=print):
    if port_is_listening(port):
        return
    log(f"  [{port}] puerto cerrado, lanzando instancia...")
    launch_instance(port)
    if not wait_for_port(port, timeout=timeout_s):
        raise TimeoutError(f"El puerto {port} no abrio en {timeout_s}s")
    time.sleep(3.0)


def ensure_game(port, map_path=None, timeout_s=180, log=print):
    map_path = map_path or find_a01()
    t0 = time.time()
    last = None
    for attempt in range(MAX_RETRIES + 1):
        remaining = max(30.0, timeout_s - (time.time() - t0))
        start_if_needed(port, remaining, log)
        restore_windows()
        try:
            return prepare_client(port, map_path, log=log, step_timeout_s=STEP_TIMEOUT_S,
                                  connect_timeout_s=STEP_TIMEOUT_S)
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            last = e
            log(f"  [{port}] intento {attempt + 1} fallo: {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                break
            kill_listener(port, log)
    raise RuntimeError(f"No se pudo dejar listo el juego en el puerto {port}") from last
