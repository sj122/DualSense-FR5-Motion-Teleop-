from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# 请确保 joy 库在你的环境里正常工作
from joy import JoyShockDevice
from fairino import Robot

# ==========================================
# 全局配置与初始化
# ==========================================
#IP = '192.168.40.129'
IP = '192.168.58.2'
robot = Robot.RPC(IP)

READ_HZ = 100
SEND_HZ = 30
SEND_INTERVAL = 1.0 / SEND_HZ

# 低通滤波系数 (作用于位置和平滑，越小越平滑但延迟越大，1.0为无滤波)
FILTER_ALPHA = 0.15
L2_THRESHOLD = 0.2

def init():
    home_pose_sim = (0, -90, 90, 0, 90, 0)
    home_pose_real=(-45,-113,-72,8,90,0)
    robot.SetSpeed(50)
    robot.MoveJ(home_pose_real, 0, 0)
    # 等待机械臂到位
    time.sleep(2)
    return True

# ==========================================
# 数学工具与滤波器
# ==========================================
def rotation_from_quaternion_wxyz(quaternion_wxyz) -> R:
    w, x, y, z = quaternion_wxyz
    return R.from_quat([x, y, z, w])

def unwrap_euler_degrees(previous_euler_deg, current_euler_deg) -> np.ndarray:
    """将当前欧拉角映射到离上一帧最近的等效表达，减少 ±180° 分支切换带来的跳变。"""
    previous = np.asarray(previous_euler_deg, dtype=float)
    current = np.asarray(current_euler_deg, dtype=float)
    wrapped_delta = (current - previous + 180.0) % 360.0 - 180.0
    return previous + wrapped_delta

class RotationLowPassFilter:
    """在旋转空间做一阶低通(Slerp)，而不是直接对欧拉角逐轴求平均"""
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
            np.vstack([self._filtered_rotation.as_quat(), target_rotation.as_quat()])
        )
        slerp = Slerp([0.0, 1.0], key_rotations)
        self._filtered_rotation = slerp([self.alpha])[0]
        return self._filtered_rotation

class PositionLowPassFilter:
    """对 XYZ 平移目标点进行一阶低通滤波"""
    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._filtered_pos: np.ndarray | None = None

    def reset(self) -> None:
        self._filtered_pos = None

    def update(self, target_pos: list[float]) -> list[float]:
        target = np.array(target_pos, dtype=float)
        if self._filtered_pos is None or self.alpha >= 1.0:
            self._filtered_pos = target
            return self._filtered_pos.tolist()

        self._filtered_pos = self.alpha * target + (1.0 - self.alpha) * self._filtered_pos
        return self._filtered_pos.tolist()

# ==========================================
# 机械臂 API 接入点
# ==========================================
def get_robot_current_pose() -> tuple[list[float], R]:
    """获取机械臂当前真实位置和姿态"""
    err, pos = robot.GetActualTCPPose()
    current_pos = [pos[0], pos[1], pos[2]]
    current_rot = R.from_euler("xyz", [pos[3], pos[4], pos[5]], degrees=True)
    return current_pos, current_rot

# def send_robot_target_pose(
#     target_rotation: R,
#     locked_position: list[float],
#     target_euler_deg: np.ndarray | None = None,
# ) -> None:
#     """下发目标位置与姿态给机械臂的底层伺服"""
#     if target_euler_deg is None:
#         target_rx, target_ry, target_rz = target_rotation.as_euler("xyz", degrees=True)
#     else:
#         target_rx, target_ry, target_rz = np.asarray(target_euler_deg, dtype=float)

#     # 发送指令：使用锁定的/摇杆积分的位置，加上计算好的绝对姿态
#     robot.MoveCart( [
#         locked_position[0], 
#         locked_position[1], 
#         locked_position[2], 
#         target_rx, 
#         target_ry, 
#         target_rz
#     ],0,00)
    
#     print(
#         f"下发指令 [{SEND_HZ}Hz] -> "
#         f"P:[{locked_position[0]:.1f}, {locked_position[1]:.1f}, {locked_position[2]:.1f}] "
#         f"R:[{target_rx:6.1f}°, {target_ry:6.1f}°, {target_rz:6.1f}°]\033[K",
#         end="\r",
#     )
def send_robot_target_pose(
    target_rotation: R,
    locked_position: list[float],
    target_euler_deg: np.ndarray | None = None,
) -> None:
    """下发目标位置与姿态给机械臂的底层伺服"""
    if target_euler_deg is None:
        target_rx, target_ry, target_rz = target_rotation.as_euler("xyz", degrees=True)
    else:
        target_rx, target_ry, target_rz = np.asarray(target_euler_deg, dtype=float)

    # 强制转换为 Python 原生浮点数 (避免 NumPy float64 序列化报错)
    desc_pos = [
        float(locked_position[0]), 
        float(locked_position[1]), 
        float(locked_position[2]), 
        float(target_rx), 
        float(target_ry), 
        float(target_rz)
    ]
    
    # 外部轴 (原生 float)
    exaxis = [0.0, 0.0, 0.0, 0.0]

    try:
        # 【终极修复】：严格匹配旧版固件底层的 8 参数签名
        # 真实参数顺序: mode, desc_pos, exaxis, acc, vel, cmdT, filterT, gain
        robot.robot.ServoCart(
            0,                    # 1. mode: 0 (绝对运动)
            desc_pos,             # 2. desc_pos: ARRAY (目标位姿)
            exaxis,               # 3. exaxis: ARRAY (外部轴)
            0.0,                  # 4. acc: DOUBLE (加速度)
            0.0,                  # 5. vel: DOUBLE (速度)
            float(SEND_INTERVAL), # 6. cmdT: DOUBLE (控制周期)
            0.0,                  # 7. filterT: DOUBLE (滤波时间)
            0.0                   # 8. gain: DOUBLE (增益)
        )
    except Exception as e:
        print(f"\n[警告] 底层 RPC 发送异常: {e}")

    print(
        f"下发指令 [{SEND_HZ}Hz] -> "
        f"P:[{locked_position[0]:.1f}, {locked_position[1]:.1f}, {locked_position[2]:.1f}] "
        f"R:[{target_rx:6.1f}°, {target_ry:6.1f}°, {target_rz:6.1f}°]\033[K",
        end="\r",
    )
# ==========================================
# 核心：遥操作控制器
# ==========================================
class RobotTeleopController:
    def __init__(self, clutch_threshold: float = 0.2):
        self.clutch_threshold = clutch_threshold
        
        # 姿态参考
        self.reference_hand_rotation: R | None = None
        self.base_robot_rotation: R = R.identity()
        
        # 位置参考 (-495, -130, 475 为默认安全点，L2按下时会自动更新为当前真实值)370, -554, 615
        self.current_target_position: list[float] = [370, -554, 615]
        
        self._clutch_active = False
        self._last_l1_pressed = False

        # --- 摇杆平移控制参数 ---
        self.max_translation_speed = 150.0  # 最大平移速度 (毫米/秒)
        self.stick_deadzone = 0.15          # 摇杆死区
        self.dt = 1.0 / READ_HZ             # 积分时间步长

        # 相似变换矩阵: 手柄坐标系 -> 机械臂基坐标系映射
        # (手柄Z->机械臂X, 手柄X->机械臂-Y, 手柄Y->机械臂-Z)
        M = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
        self.R_map = R.from_matrix(M)

    def update(self, frame: dict) -> tuple[bool, R | None, list[float] | None]:
        quaternion_wxyz = frame["quaternion_wxyz"]
        current_hand_rotation = rotation_from_quaternion_wxyz(quaternion_wxyz)
        
        # 提取按键与摇杆
        l2_value = max(frame["analogs"]["l2_value"], 1.0 if frame["buttons"]["l2_pressed"] else 0.0)
        clutch_active = l2_value > self.clutch_threshold
        l1_pressed = frame["buttons"]["l1"]
        recalibrate_pressed = l1_pressed and not self._last_l1_pressed
        
        target_robot_rotation = None
        target_robot_position = None

        if recalibrate_pressed or self.reference_hand_rotation is None:
            self.reference_hand_rotation = current_hand_rotation
            self._lock_robot_base_state()
            print("\n[L1 触发] 已重新对齐手柄参考系与机械臂当前状态。\033[K")

        if clutch_active and not self._clutch_active:
            self.reference_hand_rotation = current_hand_rotation
            self._lock_robot_base_state()
            rx, ry, rz = self.base_robot_rotation.as_euler("xyz", degrees=True)
            print(
                f"\n[L2 按下] 锁定机械臂状态作为参考原点 -> "
                f"Pos={self.current_target_position}, Rx={rx:.1f}, Ry={ry:.1f}, Rz={rz:.1f}\033[K"
            )
            
        elif clutch_active:
            # === 1. 姿态计算 (相似变换映射) ===
            delta_hand = self.reference_hand_rotation.inv() * current_hand_rotation
            delta_robot = self.R_map * delta_hand * self.R_map.inv()
            target_robot_rotation = delta_robot * self.base_robot_rotation

            # === 2. 平移计算 (积分器) ===
            # 注：不同手柄底层库对于摇杆的键名可能不同 (比如 l_x 或是 left_x)，请根据实际修改
            lx = frame["analogs"].get("l_x", 0.0)
            ly = frame["analogs"].get("l_y", 0.0)
            ry = frame["analogs"].get("r_y", 0.0)

            # 死区过滤
            lx = 0.0 if abs(lx) < self.stick_deadzone else lx
            ly = 0.0 if abs(ly) < self.stick_deadzone else ly
            ry = 0.0 if abs(ry) < self.stick_deadzone else ry

            # 速度映射 (此处极性可根据你的观察习惯反转正负号)
            dx = ly * self.max_translation_speed * self.dt   # 左摇杆推前 -> X 增加
            dy = -lx * self.max_translation_speed * self.dt  # 左摇杆推右 -> Y 减小
            dz = ry * self.max_translation_speed * self.dt   # 右摇杆推上 -> Z 增加

            self.current_target_position[0] += dx
            self.current_target_position[1] += dy
            self.current_target_position[2] += dz
            
            target_robot_position = self.current_target_position

        elif not clutch_active and self._clutch_active:
            print("\n[L2 松开] 停止下发指令，等待下次接管。\033[K")

        self._clutch_active = clutch_active
        self._last_l1_pressed = l1_pressed
        
        return clutch_active, target_robot_rotation, target_robot_position

    def _lock_robot_base_state(self):
        """重新获取机械臂当前真实状态，消除累计误差"""
        current_pos, current_rot = get_robot_current_pose()
        self.current_target_position = current_pos
        self.base_robot_rotation = current_rot

# ==========================================
# 主程序循环
# ==========================================
def main() -> None:
    print("=" * 50)
    print(" 机械臂遥操作节点启动 (平移积分 + 姿态映射 + 联合平滑)")
    print(f" [配置] 读取频率: {READ_HZ}Hz | 下发频率: {SEND_HZ}Hz")
    print(" [按键] L1: 重新校准中心点 | L2: 踩下离合开始控制")
    print(" [摇杆] 左摇杆控制 XY，右摇杆上下控制 Z")
    print("=" * 50)
    print("正在连接手柄，请将其平放数秒完成底层校准...")

    with JoyShockDevice() as device:
        device.connect_first_device(warmup_seconds=3.0)
        
        controller = RobotTeleopController(clutch_threshold=L2_THRESHOLD)
        rotation_filter = RotationLowPassFilter(alpha=FILTER_ALPHA)
        position_filter = PositionLowPassFilter(alpha=FILTER_ALPHA)

        last_sent_euler = None
        last_send_time = time.time()
        read_interval = 1.0 / READ_HZ

        try:
            while True:
                loop_start_time = time.time()

                # 读取手柄帧
                frame = device.read_input_frame(l2_threshold=L2_THRESHOLD)
                
                # 更新控制器逻辑
                is_active, target_rotation, target_position = controller.update(frame=frame)

                current_time = time.time()
                
                # 如果踩下离合，且有目标输出
                if is_active and target_rotation is not None and target_position is not None:
                    # 联合平滑滤波
                    smooth_rotation = rotation_filter.update(target_rotation)
                    smooth_position = position_filter.update(target_position)

                    # 按照 sendhz频率下发给机械臂
                    if current_time - last_send_time >= SEND_INTERVAL:
                        command_euler = np.array(smooth_rotation.as_euler("xyz", degrees=True))
                        
                        # 欧拉角连续化，防止死锁跳变
                        if last_sent_euler is not None:
                            command_euler = unwrap_euler_degrees(last_sent_euler, command_euler)

                        # 发送位置与姿态
                        send_robot_target_pose(
                            target_rotation=smooth_rotation,
                            locked_position=smooth_position,
                            target_euler_deg=command_euler,
                        )

                        last_sent_euler = command_euler
                        last_send_time = current_time
                else:
                    # 松开离合时，重置所有滤波器和状态
                    last_sent_euler = None
                    rotation_filter.reset()
                    position_filter.reset()

                # 精确控制 100Hz 读取循环
                elapsed = time.time() - loop_start_time
                sleep_time = read_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n已收到中断信号，遥操作节点安全关闭。")

if __name__ == "__main__":
    init()
    main()