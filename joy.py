from __future__ import annotations

import argparse
import ctypes
import math
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ahrs import AHRS_ALGORITHMS, create_ahrs, jsl_convention  # noqa: E402

# 默认姿态解算算法：jsl / jsl_fixed / ekf / eskf / madgwick / mahony (见 ahrs.py)
DEFAULT_AHRS = "ekf"


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


class IMU_STATE(ctypes.Structure):
    """JoyShockLibrary 返回的原始 IMU 数据：加速度 (g)、已标定角速度 (deg/s)。"""

    _fields_ = [
        ("accelX", ctypes.c_float),
        ("accelY", ctypes.c_float),
        ("accelZ", ctypes.c_float),
        ("gyroX", ctypes.c_float),
        ("gyroY", ctypes.c_float),
        ("gyroZ", ctypes.c_float),
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


# JslSetCallback 回调：设备每上报一帧 (DualSense 约 250 Hz) 调用一次
# (deviceId, 当前按键状态, 上一帧按键状态, 当前 IMU, 上一帧 IMU, 距上一帧的时间 s)
JSL_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_int, JOY_SHOCK_STATE, JOY_SHOCK_STATE, IMU_STATE, IMU_STATE, ctypes.c_float
)


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
    if hasattr(library, "JslGetIMUState"):
        library.JslGetIMUState.restype = IMU_STATE
    if hasattr(library, "JslGetAndFlushAccumulatedGyro"):
        float_ptr = ctypes.POINTER(ctypes.c_float)
        library.JslGetAndFlushAccumulatedGyro.argtypes = [ctypes.c_int, float_ptr, float_ptr, float_ptr]
        library.JslGetAndFlushAccumulatedGyro.restype = None
    if hasattr(library, "JslSetCallback"):
        library.JslSetCallback.argtypes = [JSL_CALLBACK]
        library.JslSetCallback.restype = None
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

    ahrs 参数选择姿态解算算法 (见 ahrs.py)：

    - ``"jsl"``      直接使用 DLL 内置融合输出的四元数 (原方案)
    - ``"jsl_fixed"`` JSL 同一算法的 Python 移植，套用上游修复后的重力修正
    - ``"ekf"``      航姿扩展卡尔曼滤波，估计陀螺零偏，抗平移加速度干扰 (默认)
    - ``"eskf"``     误差状态卡尔曼滤波 (ESKF / MEKF)，与 ekf 同样的噪声与门限设置
    - ``"madgwick"`` Madgwick 梯度下降滤波
    - ``"mahony"``   Mahony 显式互补滤波 (PI 修正)

    除 jsl 外的算法默认通过 JslSetCallback 在 DLL 的输入线程里逐帧 (约 250 Hz) 更新，
    积分步长用手柄上报的帧间隔，和主循环读取频率无关 (立方体示例只有 10 Hz 也不受影响)。
    剧烈晃动时如果按 10~100 Hz 用平均角速度积分，三轴转动的不可交换性和帧计数误差会
    累积成几度到几十度的姿态误差，所以必须逐帧积分。
    DLL 不支持回调或 imu_callback=False 时，退回到 JslGetAndFlushAccumulatedGyro 轮询。

    输出四元数与 JSL 内置四元数采用相同约定 (见 ahrs.jsl_convention)，
    切换算法后 servo_control.py 的 R_map 无需修改。
    注意 JslSetCallback 是全局的，同一进程只应创建一个启用回调的 JoyShockDevice。
    """

    def __init__(
        self,
        dll_path: Optional[str | Path] = None,
        library=None,
        ahrs: str = DEFAULT_AHRS,
        ahrs_options: Optional[dict] = None,
        imu_callback: bool = True,
    ):
        self.dll_path = Path(dll_path) if dll_path else resolve_default_dll_path()
        self._library = library
        self.device_id: Optional[int] = None
        self.ahrs_name = ahrs.lower()
        self.ahrs_filter = create_ahrs(self.ahrs_name, **(ahrs_options or {}))
        self._last_ahrs_time: Optional[float] = None
        self.imu_callback = imu_callback
        self._callback = None
        self._lock = threading.Lock()
        self._latest_quat: Optional[Tuple[float, float, float, float]] = None
        # 逐帧 IMU 监听器：fn(gyro_dps, accel_g, dt)，在 DLL 输入线程里调用 (录制数据用)
        self.imu_listeners: list = []

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
        self.reset_ahrs()
        self._install_callback()

        if warmup_seconds > 0:
            time.sleep(warmup_seconds)
        if self.callback_active and self.ahrs_filter is not None:
            # 等第一帧 IMU 到达，保证滤波器已经用重力完成初始对准
            deadline = time.perf_counter() + 1.0
            while self._latest_quat is None and time.perf_counter() < deadline:
                time.sleep(0.005)
        return self.device_id

    @property
    def callback_active(self) -> bool:
        return self._callback is not None

    def _install_callback(self) -> None:
        if not self.imu_callback or self._callback is not None or not hasattr(self.library, "JslSetCallback"):
            return

        def on_report(device_id, state, last_state, imu, last_imu, delta_time):
            if device_id != self.device_id:
                return
            gyro = (imu.gyroX, imu.gyroY, imu.gyroZ)
            accel = (imu.accelX, imu.accelY, imu.accelZ)
            dt = float(delta_time)
            if not 0.0 < dt < 0.1:  # 首帧或断线重连，只做重力对准不积分
                dt = 0.0
            if self.ahrs_filter is not None:
                with self._lock:
                    q = self.ahrs_filter.update(gyro, accel, dt)
                    if self.ahrs_filter.initialized:
                        self._latest_quat = tuple(float(v) for v in jsl_convention(q))
            for listener in self.imu_listeners:
                listener(gyro, accel, dt)

        self._callback = JSL_CALLBACK(on_report)  # 必须保持引用，防止被垃圾回收
        self.library.JslSetCallback(self._callback)

    def _remove_callback(self) -> None:
        if self._callback is not None:
            self.library.JslSetCallback(JSL_CALLBACK())  # 空函数指针
            self._callback = None

    def reset_ahrs(self) -> None:
        """重新初始化姿态滤波器 (下一帧用当前加速度重新对准重力)。"""
        with self._lock:
            if self.ahrs_filter is not None:
                self.ahrs_filter.reset()
            self._last_ahrs_time = None
            self._latest_quat = None

    def ensure_connected(self) -> int:
        if self.device_id is None:
            return self.connect_first_device(warmup_seconds=0.0)
        return self.device_id

    def read_motion_state(self) -> MOTION_STATE:
        device_id = self.ensure_connected()
        return self.library.JslGetMotionState(device_id)

    def read_imu_state(self) -> IMU_STATE:
        if not hasattr(self.library, "JslGetIMUState"):
            raise RuntimeError("当前 JoyShockLibrary 未导出 JslGetIMUState，只能使用 --ahrs jsl。")
        device_id = self.ensure_connected()
        return self.library.JslGetIMUState(device_id)

    def read_accumulated_gyro_dps(self) -> Tuple[float, float, float]:
        """读取自上次调用以来的平均角速度 (deg/s)，不会漏掉两次轮询之间的陀螺数据。"""
        if not hasattr(self.library, "JslGetAndFlushAccumulatedGyro"):
            imu = self.read_imu_state()
            return (float(imu.gyroX), float(imu.gyroY), float(imu.gyroZ))
        device_id = self.ensure_connected()
        gx, gy, gz = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
        self.library.JslGetAndFlushAccumulatedGyro(device_id, ctypes.byref(gx), ctypes.byref(gy), ctypes.byref(gz))
        return (gx.value, gy.value, gz.value)

    def _update_ahrs(self) -> Tuple[float, float, float, float]:
        now = time.perf_counter()
        gyro = self.read_accumulated_gyro_dps()
        imu = self.read_imu_state()
        accel = (float(imu.accelX), float(imu.accelY), float(imu.accelZ))
        # 第一帧只用加速度对准重力，并丢弃之前累积的陀螺数据
        dt = 0.0 if self._last_ahrs_time is None else now - self._last_ahrs_time
        self._last_ahrs_time = now
        q = self.ahrs_filter.update(gyro, accel, dt)
        return tuple(float(v) for v in jsl_convention(q))

    def read_simple_state(self) -> JOY_SHOCK_STATE:
        if not hasattr(self.library, "JslGetSimpleState"):
            raise RuntimeError("当前 JoyShockLibrary 未导出 JslGetSimpleState，无法读取 L1/L2。")
        device_id = self.ensure_connected()
        return self.library.JslGetSimpleState(device_id)

    def read_euler_deg(self) -> Tuple[float, float, float]:
        return quat_to_euler(*self.read_quaternion_wxyz())

    def read_quaternion_wxyz(self) -> Tuple[float, float, float, float]:
        if self.ahrs_filter is not None:
            if self.callback_active:
                with self._lock:
                    q = self._latest_quat
                if q is not None:
                    return q
                return (1.0, 0.0, 0.0, 0.0)
            return self._update_ahrs()
        motion = self.read_motion_state()
        return (float(motion.quatW), float(motion.quatX), float(motion.quatY), float(motion.quatZ))

    def read_quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        w, x, y, z = self.read_quaternion_wxyz()
        return (x, y, z, w)

    def read_input_frame(self, l2_threshold: float = 0.2) -> dict:
        quat_w, quat_x, quat_y, quat_z = self.read_quaternion_wxyz()
        simple = self.read_simple_state()

        l1_pressed = bool(simple.buttons & DualSenseButtons.L1)
        l2_button = bool(simple.buttons & DualSenseButtons.L2)
        l2_value = float(simple.lTrigger)
        l2_pressed = l2_value > l2_threshold or l2_button

        roll, pitch, yaw = quat_to_euler(quat_w, quat_x, quat_y, quat_z)

        return {
            "ahrs": self.ahrs_name,
            "quaternion_wxyz": (quat_w, quat_x, quat_y, quat_z),
            "quaternion_xyzw": (quat_x, quat_y, quat_z, quat_w),
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
            self._remove_callback()
            self._library.JslDisconnectAndDisposeAll()
        self.device_id = None

    def __enter__(self) -> "JoyShockDevice":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def add_ahrs_argument(parser: argparse.ArgumentParser, default: str = DEFAULT_AHRS) -> None:
    parser.add_argument(
        "--ahrs",
        choices=AHRS_ALGORITHMS,
        default=default,
        help=(
            "姿态解算算法: jsl=DLL 内置融合(原方案), jsl_fixed=JSL 上游修复版, ekf=扩展卡尔曼滤波, "
            "eskf=误差状态卡尔曼滤波, madgwick=Madgwick 滤波, mahony=Mahony 互补滤波 (默认: %(default)s)"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 DualSense 姿态并打印欧拉角")
    add_ahrs_argument(parser)
    args = parser.parse_args()

    with JoyShockDevice(ahrs=args.ahrs) as device:
        print(f"姿态解算算法: {args.ahrs}")
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
