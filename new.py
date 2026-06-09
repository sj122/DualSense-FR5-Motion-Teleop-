from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from joy import JoyShockDevice  # 你的手柄封装库


# ==========================================
# 1. 机械臂 API 接口层 (请替换为 Fairino SDK 的实际调用)
# ==========================================
def get_robot_current_pose() -> R:
    """
    【API 接入点】获取机械臂末端当前的真实姿态。
    你需要在这里调用 Fairino 的 get_global_waypoint() 或类似接口。
    """
    # 假设从 API 获取到的当前 Rx, Ry, Rz (单位: 度) 为 180, 0, 0
    current_rx, current_ry, current_rz = 180.0, 0.0, 0.0 
    
    # 将其转为 SciPy 的 Rotation 对象返回
    return R.from_euler("xyz", [current_rx, current_ry, current_rz], degrees=True)

def send_robot_target_pose(target_rotation: R) -> None:
    """
    【API 接入点】将解算后的目标姿态下发给机械臂。
    你需要在这里调用 Fairino 的 MoveCart() 或 ServoCart() 接口。
    """
    target_rx, target_ry, target_rz = target_rotation.as_euler("xyz", degrees=True)
    
    # 打印到控制台以供调试
    print(f"下发控制指令 -> Rx: {target_rx:6.1f}°, Ry: {target_ry:6.1f}°, Rz: {target_rz:6.1f}°     ", end='\r')


# ==========================================
# 2. 核心数学转换工具
# ==========================================
def rotation_from_quaternion_wxyz(quaternion_wxyz) -> R:
    w, x, y, z = quaternion_wxyz
    return R.from_quat([x, y, z, w])


# ==========================================
# 3. 机械臂遥操核心控制器 (离合 + 基准对齐)
# ==========================================
class RobotTeleopController:
    """处理手柄与机械臂末端的相对姿态映射。"""

    def __init__(self, clutch_threshold: float = 0.2):
        self.clutch_threshold = clutch_threshold
        
        # 手柄的基准姿态 (按下离合瞬间的手柄姿态)
        self.reference_hand_rotation: R | None = None
        # 机械臂的基准姿态 (按下离合瞬间的物理机械臂姿态)
        self.base_robot_rotation: R = R.identity()
        
        self._clutch_active = False
        self._last_l1_pressed = False

    def update(self, quaternion_wxyz, l2_value: float, l1_pressed: bool) -> tuple[bool, R | None]:
        current_hand_rotation = rotation_from_quaternion_wxyz(quaternion_wxyz)
        
        clutch_active = l2_value > self.clutch_threshold
        recalibrate_pressed = l1_pressed and not self._last_l1_pressed
        
        target_robot_rotation = None

        # [逻辑 A]: L1 强制校准/对齐
        # 如果按下 L1，或者程序刚启动还没有基准，则强制对齐当前的坐标系
        if recalibrate_pressed or self.reference_hand_rotation is None:
            self.reference_hand_rotation = current_hand_rotation
            self.base_robot_rotation = get_robot_current_pose()
            print("\n[L1 触发] 已重置手柄内部坐标系与机械臂对齐。")

        # [逻辑 B]: L2 离合控制
        if clutch_active and not self._clutch_active:
            # 刚按下离合的瞬间：刷新基准
            self.reference_hand_rotation = current_hand_rotation
            self.base_robot_rotation = get_robot_current_pose()
            rx, ry, rz = self.base_robot_rotation.as_euler("xyz", degrees=True)
            print(f"\n[离合激活] 锁定当前机械臂姿态作为偏移原点: Rx={rx:.1f}, Ry={ry:.1f}, Rz={rz:.1f}")

        elif clutch_active:
            # 离合持续按下时：计算增量并叠加
            # 增量 = 基准手柄姿态的逆 * 当前手柄姿态
            delta_rotation = self.reference_hand_rotation.inv() * current_hand_rotation
            
            # 目标姿态 = 机械臂初始姿态 * 增量
            target_robot_rotation = self.base_robot_rotation * delta_rotation

        elif not clutch_active and self._clutch_active:
            # 刚松开离合的瞬间
            print("\n[离合松开] 停止下发姿态指令，机械臂悬停。")

        # 更新状态
        self._clutch_active = clutch_active
        self._last_l1_pressed = l1_pressed
        
        return clutch_active, target_robot_rotation


# ==========================================
# 4. 主程序入口
# ==========================================
def main() -> None:
    print("==================================================")
    print("机械臂遥操节点启动")
    print("正在连接手柄，请将其平放几秒钟完成底层校准...")
    print(" - 按住 L2 : 激活姿态映射")
    print(" - 按下 L1 : 强制重置相对原点 (建议手柄拿歪时按一下)")
    print(" - 按 Ctrl+C : 安全退出")
    print("==================================================\n")

    with JoyShockDevice() as device:
        # 给到底层滤波器 3 秒时间收敛
        device.connect_first_device(warmup_seconds=3.0)
        
        controller = RobotTeleopController(clutch_threshold=0.2)

        try:
            while True:
                # 为了机械臂的控制平顺度，通常维持 50Hz 或 100Hz 的下发频率
                time.sleep(0.01) 

                frame = device.read_input_frame(l2_threshold=0.2)
                
                # 兼容模拟扳机和数字按键
                effective_l2_value = max(
                    frame["analogs"]["l2_value"],
                    1.0 if frame["buttons"]["l2_pressed"] else 0.0,
                )
                
                # 更新控制器逻辑
                is_active, target_rotation = controller.update(
                    quaternion_wxyz=frame["quaternion_wxyz"],
                    l2_value=effective_l2_value,
                    l1_pressed=frame["buttons"]["l1"],
                )
                
                # 如果离合处于激活状态，下发数据到机械臂
                if is_active and target_rotation is not None:
                    send_robot_target_pose(target_rotation)

        except KeyboardInterrupt:
            print("\n\n已收到中断信号，遥操节点安全关闭。")


if __name__ == "__main__":
    main()