from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple


class MOTION_STATE(ctypes.Structure):
    """JoyShockLibrary 返回的姿态结构体。"""

    _fields_ = [
        ("quatW", ctypes.c_float),
        ("quatX", ctypes.c_float),
        ("quatY", ctypes.c_float),
        ("quatZ", ctypes.c_float),
        ("accelX", ctypes.c_float),
        ("accelY", ctypes.c_float),
        ("accelZ", ctypes.c_float),
        ("gravX", ctypes.c_float),
        ("gravY", ctypes.c_float),
        ("gravZ", ctypes.c_float),
    ]


class JOY_SHOCK_STATE(ctypes.Structure):
    """JoyShockLibrary 的简化按键/摇杆状态。"""

    _fields_ = [
        ("buttons", ctypes.c_int),
        ("lTrigger", ctypes.c_float),
        ("rTrigger", ctypes.c_float),
        ("stickLX", ctypes.c_float),
        ("stickLY", ctypes.c_float),
        ("stickRX", ctypes.c_float),
        ("stickRY", ctypes.c_float),
    ]


class DualSenseButtons:
    L1 = 0x00100
    R1 = 0x00200
    L2 = 0x00400
    R2 = 0x00800


def resolve_default_dll_path() -> Path:
    return Path(__file__).resolve().parent / "JoyShockLibrary.dll"


def load_library(dll_path: Optional[str | Path] = None):
    library = ctypes.CDLL(str(dll_path or resolve_default_dll_path()))
    library.JslGetMotionState.restype = MOTION_STATE
    if hasattr(library, "JslGetSimpleState"):
        library.JslGetSimpleState.restype = JOY_SHOCK_STATE
    return library


def quat_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """四元数转欧拉角，返回值单位为度。"""

    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class JoyShockDevice:
    """JoyShockLibrary 的轻量封装。

    关键点是导入模块时不自动连接设备，只有显式调用 connect_first_device()
    或 read_*() 时才会真正访问手柄。
    """

    def __init__(self, dll_path: Optional[str | Path] = None, library=None):
        self.dll_path = Path(dll_path) if dll_path else resolve_default_dll_path()
        self._library = library
        self.device_id: Optional[int] = None

    @property
    def library(self):
        if self._library is None:
            self._library = load_library(self.dll_path)
        return self._library

    def connect_first_device(self, warmup_seconds: float = 3.0) -> int:
        num_devices = self.library.JslConnectDevices()
        if num_devices == 0:
            raise RuntimeError("没有检测到手柄，请确保已通过蓝牙或 USB 连接。")

        device_handles = (ctypes.c_int * 4)()
        self.library.JslGetConnectedDeviceHandles(device_handles, 4)
        self.device_id = int(device_handles[0])

        if warmup_seconds > 0:
            time.sleep(warmup_seconds)
        return self.device_id

    def ensure_connected(self) -> int:
        if self.device_id is None:
            return self.connect_first_device(warmup_seconds=0.0)
        return self.device_id

    def read_motion_state(self) -> MOTION_STATE:
        device_id = self.ensure_connected()
        return self.library.JslGetMotionState(device_id)

    def read_simple_state(self) -> JOY_SHOCK_STATE:
        if not hasattr(self.library, "JslGetSimpleState"):
            raise RuntimeError("当前 JoyShockLibrary 未导出 JslGetSimpleState，无法读取 L1/L2。")
        device_id = self.ensure_connected()
        return self.library.JslGetSimpleState(device_id)

    def read_euler_deg(self) -> Tuple[float, float, float]:
        motion = self.read_motion_state()
        return quat_to_euler(
            motion.quatW,
            motion.quatX,
            motion.quatY,
            motion.quatZ,
        )

    def read_quaternion_wxyz(self) -> Tuple[float, float, float, float]:
        motion = self.read_motion_state()
        return (motion.quatW, motion.quatX, motion.quatY, motion.quatZ)

    def read_quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        motion = self.read_motion_state()
        return (motion.quatX, motion.quatY, motion.quatZ, motion.quatW)

    def read_input_frame(self, l2_threshold: float = 0.2) -> dict:
        motion = self.read_motion_state()
        simple = self.read_simple_state()

        l1_pressed = bool(simple.buttons & DualSenseButtons.L1)
        l2_button = bool(simple.buttons & DualSenseButtons.L2)
        l2_value = float(simple.lTrigger)
        l2_pressed = l2_value > l2_threshold or l2_button

        roll, pitch, yaw = quat_to_euler(
            motion.quatW,
            motion.quatX,
            motion.quatY,
            motion.quatZ,
        )

        return {
            "quaternion_wxyz": (
                float(motion.quatW),
                float(motion.quatX),
                float(motion.quatY),
                float(motion.quatZ),
            ),
            "quaternion_xyzw": (
                float(motion.quatX),
                float(motion.quatY),
                float(motion.quatZ),
                float(motion.quatW),
            ),
            "euler_deg": (roll, pitch, yaw),
            "buttons": {
                "l1": l1_pressed,
                "l2_button": l2_button,
                "l2_pressed": l2_pressed,
            },
            "analogs": {
                "l2_value": l2_value,
                "r2_value": float(simple.rTrigger),
                "stick_lx": float(simple.stickLX),
                "stick_ly": float(simple.stickLY),
                "stick_rx": float(simple.stickRX),
                "stick_ry": float(simple.stickRY),
            },
        }

    def stream_euler(self, interval_seconds: float = 0.01) -> Iterator[Tuple[float, float, float]]:
        while True:
            yield self.read_euler_deg()
            time.sleep(interval_seconds)

    def close(self) -> None:
        if self._library is not None:
            self._library.JslDisconnectAndDisposeAll()
        self.device_id = None

    def __enter__(self) -> "JoyShockDevice":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    with JoyShockDevice() as device:
        print("正在连接手柄，请将其平放几秒钟让底层自动校准...")
        device.connect_first_device(warmup_seconds=3.0)
        print("手柄连接成功。")

        try:
            while True:
                roll, pitch, yaw = device.read_euler_deg()
                print(
                    f"末端映射 -> Rx: {roll:6.1f}度, Ry: {pitch:6.1f}度, Rz: {yaw:6.1f}度    ",
                    end="\r",
                )
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n已断开连接。")


if __name__ == "__main__":
    main()
