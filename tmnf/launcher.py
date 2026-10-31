import os
import subprocess
import time

TMLOADER = os.path.join(os.path.expanduser("~"), "AppData", "Local", "TMLoader", "TMLoader.exe")
PROFILE = "default"
BASE_PORT = 8478
GAME_SETTINGS = ("set autologin jesus", "set skip_map_load_screens true", "set unfocused_fps_limit false", "set autorewind false")

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


def _game_window(pids):
    import win32gui
    import win32process
    found = []

    def cb(h, _):
        if win32process.GetWindowThreadProcessId(h)[1] in pids and win32gui.IsWindowVisible(h):
            if win32gui.GetClassName(h) != "ConsoleWindowClass":
                found.append(h)

    win32gui.EnumWindows(cb, None)
    return found[0] if found else None


def dismiss_offline_dialog(pids=None, attempts=1):
    import ctypes
    import win32api
    import win32con
    import win32gui
    from PIL import ImageGrab
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    h = _game_window(set(pids or tm_pids()))
    if h is None:
        return False
    if win32gui.IsIconic(h):
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
    win32api.keybd_event(0x12, 0, 0, 0)
    win32api.keybd_event(0x12, 0, 2, 0)
    try:
        win32gui.SetForegroundWindow(h)
    except Exception:
        pass
    time.sleep(0.8)
    if win32gui.GetForegroundWindow() != h:
        return False
    left, top = win32gui.ClientToScreen(h, (0, 0))
    _, _, cw, ch = win32gui.GetClientRect(h)
    bx, by = int(cw * 0.5), int(ch * 0.573)

    def move(x, y):
        win32api.SetCursorPos((left + x, top + y))
        time.sleep(0.03)

    def clickable():
        px = ImageGrab.grab((left + bx - 40, top + by - 3, left + bx + 40, top + by + 3)).convert("L")
        return sum(1 for v in px.getdata() if v > 225) / (80 * 6) > 0.25

    gx, gy = int(cw * 0.906), int(ch * 0.442)

    def drag(dx, dy, x0, y0):
        move(x0, y0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.2)
        for i in range(1, 41):
            move(x0 + dx * i // 40, y0 + dy * i // 40)
        time.sleep(0.2)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(0.8)

    for _ in range(attempts):
        dragged = False
        if not clickable():
            drag(240, 80, gx, gy)
            dragged = True
        if clickable():
            for _ in range(4):
                move(bx, by)
                time.sleep(0.2)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                time.sleep(0.1)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
                time.sleep(1.5)
                if not clickable():
                    break
            else:
                continue
            time.sleep(3.0)
            move(int(cw * 0.52), int(ch * 0.33))
            time.sleep(0.2)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            if dragged:
                time.sleep(0.5)
                drag(-240, -80, gx + 240, gy + 80)
            return True
    return False


def focus_window(pids=None):
    import win32api
    import win32gui
    h = _game_window(set(pids or tm_pids()))
    if h is None:
        return False
    win32api.keybd_event(0x12, 0, 0, 0)
    win32api.keybd_event(0x12, 0, 2, 0)
    try:
        win32gui.SetForegroundWindow(h)
    except Exception:
        return False
    return True
