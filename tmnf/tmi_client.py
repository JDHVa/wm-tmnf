import socket
import struct
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from tminterface.structs import CheckpointData, SimStateData

HOST = "127.0.0.1"


class MsgType(IntEnum):
    SC_RUN_STEP_SYNC = 1
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()


@dataclass
class CarState:
    race_time: int
    position: tuple
    velocity: tuple
    speed_kmh: float
    raw: SimStateData


class TMIClient:
    def __init__(self, port: int = 8478):
        self.port = port
        self.sock = None
        self._pending = None
        self.speed = 1.0
        self.frame_wh = None
        self.last_frame = None
        self.last_cp = None
        self.cp_events = 0


    def connect(self, timeout_s: float = 60.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout_s)
        self.sock.connect((HOST, self.port))
        msg = self._read_int()
        if msg != MsgType.SC_ON_CONNECT_SYNC:
            raise RuntimeError(f"Se esperaba SC_ON_CONNECT_SYNC y llego {msg}")
        self._pending = msg

    def close(self):
        if self.sock is None:
            return
        try:
            self.sock.sendall(struct.pack("i", MsgType.C_SHUTDOWN))
            self._pending = None
        except OSError:
            pass
        self.sock.close()
        self.sock = None


    def wait_step(self) -> int:
        self._ack()
        while True:
            msg = self._read_int()
            if msg == MsgType.SC_RUN_STEP_SYNC:
                t = self._read_int()
                self._pending = msg
                return t
            self._handle_other(msg)

    def _handle_other(self, msg):
        if msg in (MsgType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MsgType.SC_LAP_COUNT_CHANGED_SYNC):
            a = self._read_int()
            b = self._read_int()
            if msg == MsgType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                self.last_cp = (a, b)
                self.cp_events += 1
            self._respond(msg)
        elif msg == MsgType.SC_ON_CONNECT_SYNC:
            self._respond(msg)
        elif msg == MsgType.SC_REQUESTED_FRAME_SYNC:
            self.last_frame = self.get_frame(*self.frame_wh)
            self._respond(msg)
        else:
            raise RuntimeError(f"Mensaje inesperado del plugin: {msg}")

    def _ack(self):
        if self._pending is not None:
            self._respond(self._pending)
            self._pending = None

    def _respond(self, msg):
        self.sock.sendall(struct.pack("i", int(msg)))


    def set_input(self, left=False, right=False, accelerate=False, brake=False):
        self.sock.sendall(struct.pack("iBBBB", MsgType.C_SET_INPUT_STATE, left, right, accelerate, brake))

    def set_speed(self, speed: float):
        self.sock.sendall(struct.pack("if", MsgType.C_SET_SPEED, speed))
        self.speed = speed

    def set_timeout(self, ms: int):
        self.sock.sendall(struct.pack("iI", MsgType.C_SET_TIMEOUT, ms))

    def set_on_step_period(self, ms: int):
        self.sock.sendall(struct.pack("ii", MsgType.C_SET_ON_STEP_PERIOD, ms))

    def execute_command(self, cmd: str):
        data = cmd.encode()
        self.sock.sendall(struct.pack("ii", MsgType.C_EXECUTE_COMMAND, len(data)) + data)

    def give_up(self):
        self.sock.sendall(struct.pack("i", MsgType.C_GIVE_UP))

    def get_state(self) -> CarState:
        self.sock.sendall(struct.pack("i", MsgType.C_GET_SIMULATION_STATE))
        n = self._read_int()
        s = SimStateData(self._read_exact(n))
        s.cp_data.resize(CheckpointData.cp_states_field, s.cp_data.cp_states_length)
        s.cp_data.resize(CheckpointData.cp_times_field, s.cp_data.cp_times_length)
        return CarState(s.race_time, tuple(s.position), tuple(s.velocity), s.display_speed, s)

    def rewind_to_state(self, state: CarState):
        data = state.raw.data
        self.sock.sendall(struct.pack("ii", MsgType.C_REWIND_TO_STATE, len(data)) + bytes(data))

    def rewind_to_current_state(self):
        self.sock.sendall(struct.pack("i", MsgType.C_REWIND_TO_CURRENT_STATE))

    def prevent_simulation_finish(self):
        self.sock.sendall(struct.pack("i", MsgType.C_PREVENT_SIMULATION_FINISH))

    def race_finished(self) -> bool:
        self.sock.sendall(struct.pack("i", MsgType.C_RACE_FINISHED))
        return self._read_int() > 0

    def toggle_interface(self, visible: bool):
        self.sock.sendall(struct.pack("ii", MsgType.C_TOGGLE_INTERFACE, int(visible)))

    def get_inputs(self) -> str:
        self.sock.sendall(struct.pack("i", MsgType.C_GET_INPUTS))
        n = self._read_int()
        return self._read_exact(n).decode("utf-8") if n > 0 else ""

    def request_frame(self, w: int, h: int):
        self.frame_wh = (w, h)
        self.sock.sendall(struct.pack("iii", MsgType.C_REQUEST_FRAME, w, h))

    def unrequest_frame(self):
        self.sock.sendall(struct.pack("i", MsgType.C_UNREQUEST_FRAME))

    def get_frame(self, w: int, h: int) -> np.ndarray:
        data = self._read_exact(w * h * 4)
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))

    def capture_frame(self, w: int, h: int) -> np.ndarray:
        prev_speed = self.speed
        self.rewind_to_current_state()
        self.request_frame(w, h)
        self.set_speed(0)
        self._ack()
        while True:
            msg = self._read_int()
            if msg == MsgType.SC_REQUESTED_FRAME_SYNC:
                frame = self.get_frame(w, h)
                self.set_speed(prev_speed)
                self._respond(msg)
                break
            if msg == MsgType.SC_RUN_STEP_SYNC:
                self._read_int()
                self._respond(msg)
            else:
                self._handle_other(msg)
        self.last_frame = frame
        return frame

    def is_in_menus(self) -> bool:
        self.sock.sendall(struct.pack("i", MsgType.C_IS_IN_MENUS))
        return self._read_int() > 0


    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("El plugin cerro la conexion")
            buf += chunk
        return bytes(buf)

    def _read_int(self) -> int:
        return struct.unpack("i", self._read_exact(4))[0]
