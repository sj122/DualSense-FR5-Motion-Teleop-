from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from joy import JoyShockDevice

from fairino import Robot

IP='192.168.40.129'
robot=Robot.RPC(IP)

def init():
    home_pose=(0,-90,90,0,90,0)
    robot.SetSpeed(50)
    robot.MoveJ(home_pose, 0, 0)
    return True
# ==========================================
# 调试与控制参数
# ==========================================
READ_HZ = 100
SEND_HZ = 20
SEND_INTERVAL = 1.0 / SEND_HZ

# 这里的 alpha 现在作用在 100Hz 内部姿态平滑上，不再是 20Hz 下发前的欧拉角平均。
FILTER_ALPHA = 0.15
L2_THRESHOLD = 0.2


def remap_rpy(rx, ry, rz):
    """
    将手柄欧拉角 (rx, ry, rz) 映射到机械臂坐标系。

    手柄坐标系 (JoyShockLibrary, DualSense 平放时屏幕朝上):
      X: 指向手柄右侧
      Y: 指向手柄前方（远离用户）
      Z: 指向手柄上方（屏幕方向）

    机械臂坐标系 (Fairino TCP 末端):
      X: 工具平面内
      Y: 工具平面内
      Z: 工具方向（通常朝下）

    映射逻辑（手柄竖直握持如方向盘时）:
      手柄 yaw(绕Z旋转)   → 机械臂 rx(绕X旋转)
      手柄 roll(绕X旋转)  → 机械臂 ry(绕Y旋转,取反)
      手柄 pitch(绕Y旋转) → 机械臂 rz(绕Z旋转,取反)

    返回: (robot_rx, robot_ry, robot_rz) 单位：度
    """
    robot_rx = rx
    robot_ry = -rz
    robot_rz = -ry
    return robot_rx, robot_ry, robot_rz
# ==========================================
# 机械臂 API 接入点
# ==========================================
def get_robot_current_pose() -> R:
    """
    获取机械臂当前真实姿态。

    请在这里替换成你的机械臂 API。
    """
    _,pos=robot.GetActualTCPPose()
    current_rx, current_ry, current_rz = pos[3:]
    return R.from_euler("xyz", [current_rx, current_ry, current_rz], degrees=True)


def send_robot_target_pose(
    target_rotation: R,
    target_euler_deg: np.ndarray | None = None,
) -> None:
    """
    将目标姿态下发给机械臂。

    如果你的机械臂接口直接吃欧拉角，优先使用 `target_euler_deg`，
    它已经做过连续化处理，能避免 `±180°` 附近的指令跳变。
    """
    if target_euler_deg is None:
        target_rx, target_ry, target_rz = target_rotation.as_euler("xyz", degrees=True)
    else:
        target_rx, target_ry, target_rz = np.asarray(target_euler_deg, dtype=float)

    # 将手柄坐标系下的欧拉角映射到机械臂坐标系
    target_rx, target_ry, target_rz = remap_rpy(target_rx, target_ry, target_rz)
    pos=[-495,-130,475]
    robot.MoveL([pos,target_rx,target_ry,target_rz],0,0)
    print(
        f"下发控制指令 [{SEND_HZ}Hz] -> "
        f"Rx: {target_rx:6.1f}°, Ry: {target_ry:6.1f}°, Rz: {target_rz:6.1f}°\033[K",
        end="\r",
    )


# ==========================================
# 数学工具
# ==========================================
def rotation_from_quaternion_wxyz(quaternion_wxyz) -> R:
    w, x, y, z = quaternion_wxyz
    return R.from_quat([x, y, z, w])


def unwrap_euler_degrees(previous_euler_deg, current_euler_deg) -> np.ndarray:
    """
    将当前欧拉角映射到离上一帧最近的等效表达，减少 ±180° 分支切换带来的跳变。
    """
    previous = np.asarray(previous_euler_deg, dtype=float)
    current = np.asarray(current_euler_deg, dtype=float)
    wrapped_delta = (current - previous + 180.0) % 360.0 - 180.0
    return previous + wrapped_delta


class RotationLowPassFilter:
    """
    在旋转空间做一阶低通，而不是直接对欧拉角逐轴求平均。
    """

    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._filtered_rotation: R | None = None

    def reset(self) -> None:
        self._filtered_rotation = None

    def update(self, target_rotation: R) -> R:
        if self._filtered_rotation is None or self.alpha >= 1.0:
            self._filtered_rotation = target_rotation
            return self._filtered_rotation

        key_rotations = R.from_quat(
            np.vstack(
                [
                    self._filtered_rotation.as_quat(),
                    target_rotation.as_quat(),
                ]
            )
        )
        slerp = Slerp([0.0, 1.0], key_rotations)
        self._filtered_rotation = slerp([self.alpha])[0]
        return self._filtered_rotation


# ==========================================
# 遥操作控制器
# ==========================================
class RobotTeleopController:
    def __init__(self, clutch_threshold: float = 0.2):
        self.clutch_threshold = clutch_threshold
        self.reference_hand_rotation: R | None = None
        self.base_robot_rotation: R = R.identity()
        self._clutch_active = False
        self._last_l1_pressed = False

    def update(self, quaternion_wxyz, l2_value: float, l1_pressed: bool) -> tuple[bool, R | None]:
        current_hand_rotation = rotation_from_quaternion_wxyz(quaternion_wxyz)
        clutch_active = l2_value > self.clutch_threshold
        recalibrate_pressed = l1_pressed and not self._last_l1_pressed
        target_robot_rotation = None

        if recalibrate_pressed or self.reference_hand_rotation is None:
            self.reference_hand_rotation = current_hand_rotation
            self.base_robot_rotation = get_robot_current_pose()
            print("\n[L1 触发] 已重新对齐手柄参考系与机械臂当前姿态。\033[K")

        if clutch_active and not self._clutch_active:
            self.reference_hand_rotation = current_hand_rotation
            self.base_robot_rotation = get_robot_current_pose()
            rx, ry, rz = self.base_robot_rotation.as_euler("xyz", degrees=True)
            print(
                "\n[L2 按下] 锁定当前机械臂姿态作为参考原点 -> "
                f"Rx={rx:.1f}, Ry={ry:.1f}, Rz={rz:.1f}\033[K"
            )
        elif clutch_active:
            delta_rotation = self.reference_hand_rotation.inv() * current_hand_rotation
            target_robot_rotation = self.base_robot_rotation * delta_rotation
        elif not clutch_active and self._clutch_active:
            print("\n[L2 松开] 停止下发姿态指令，机械臂保持当前位置。\033[K")

        self._clutch_active = clutch_active
        self._last_l1_pressed = l1_pressed
        return clutch_active, target_robot_rotation


# ==========================================
# 主程序
# ==========================================
def main() -> None:
    print("=" * 50)
    print(" 机械臂遥操作节点启动（高频内部姿态平滑 + 低频下发）")
    print(f" [配置] 读取频率: {READ_HZ}Hz | 下发频率: {SEND_HZ}Hz")
    print(f" [配置] 内部滤波系数: {FILTER_ALPHA}")
    print("=" * 50)
    print("正在连接手柄，请将其平放数秒完成底层校准...")

    with JoyShockDevice() as device:
        device.connect_first_device(warmup_seconds=3.0)
        controller = RobotTeleopController(clutch_threshold=L2_THRESHOLD)
        rotation_filter = RotationLowPassFilter(alpha=FILTER_ALPHA)

        last_sent_euler = None
        last_send_time = time.time()
        read_interval = 1.0 / READ_HZ

        try:
            while True:
                loop_start_time = time.time()

                frame = device.read_input_frame(l2_threshold=L2_THRESHOLD)
                effective_l2_value = max(
                    frame["analogs"]["l2_value"],
                    1.0 if frame["buttons"]["l2_pressed"] else 0.0,
                )

                is_active, target_rotation = controller.update(
                    quaternion_wxyz=frame["quaternion_wxyz"],
                    l2_value=effective_l2_value,
                    l1_pressed=frame["buttons"]["l1"],
                )

                current_time = time.time()
                if is_active and target_rotation is not None:
                    smooth_rotation = rotation_filter.update(target_rotation)

                    if current_time - last_send_time >= SEND_INTERVAL:
                        command_euler = np.array(
                            smooth_rotation.as_euler("xyz", degrees=True)
                        )
                        if last_sent_euler is not None:
                            command_euler = unwrap_euler_degrees(
                                last_sent_euler,
                                command_euler,
                            )

                        send_robot_target_pose(
                            smooth_rotation,
                            target_euler_deg=command_euler,
                        )

                        last_sent_euler = command_euler
                        last_send_time = current_time
                else:
                    last_sent_euler = None
                    rotation_filter.reset()

                elapsed = time.time() - loop_start_time
                sleep_time = read_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n已收到中断信号，遥操作节点安全关闭。")


if __name__ == "__main__":
    init()
    main()
