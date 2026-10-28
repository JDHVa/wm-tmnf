import os
import subprocess
import time

TMLOADER = os.path.join(os.path.expanduser("~"), "AppData", "Local", "TMLoader", "TMLoader.exe")
PROFILE = "default"
BASE_PORT = 8478

A01_CANDIDATES = [
    r"C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\White\A01-Race.Challenge.Gbx",
    r"C:\Program Files (x86)\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\White\A01-Race.Challenge.Gbx",
    r"C:\Program Files\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\White\A01-Race.Challenge.Gbx",
]


def find_a01():
    for p in A01_CANDIDATES:
        if os.path.isfile(p):
            return p
    for root in (r"C:\Program Files (x86)", r"C:\Program Files"):
        for dirpath, _, files in os.walk(root):
            if "A01-Race.Challenge.Gbx" in files:
                return os.path.join(dirpath, "A01-Race.Challenge.Gbx")
    return None


def port_is_listening(port, host="127.0.0.1"):
    import psutil
    return any(
        c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
        for c in psutil.net_connections(kind="tcp4")
    )


def launch_instance(port, tmloader=TMLOADER, profile=PROFILE):
    if not os.path.isfile(tmloader):
        raise FileNotFoundError(f"No se encontro TMLoader en {tmloader}")
    cmd = f'"{tmloader}" run TmForever "{profile}" /configstring="set custom_port {port}"'
    return subprocess.Popen(cmd, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))


def wait_for_port(port, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_is_listening(port):
            return True
        time.sleep(1.0)
    return False


def load_map(client, map_path):
    path = map_path.replace("/", "\\")
    if " " in path and not path.startswith('"'):
        path = f'"{path}"'
    client.execute_command(f"map {path}")


def restore_windows(pids=None):
    import win32con
    import win32gui
    import win32process
    pids = set(pids or tm_pids())
    restored = []

    def cb(h, _):
        if win32process.GetWindowThreadProcessId(h)[1] in pids and win32gui.IsIconic(h):
            win32gui.ShowWindow(h, win32con.SW_SHOWNOACTIVATE)
            restored.append(h)

    win32gui.EnumWindows(cb, None)
    if restored:
        time.sleep(1.0)
    return len(restored)


def tm_pids():
    try:
        import psutil
    except ImportError:
        return []
    return [p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "tmforever.exe"]
