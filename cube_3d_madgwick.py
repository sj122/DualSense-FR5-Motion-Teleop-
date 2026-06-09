"""
PS5 DualSense 3D 立方体姿态显示 - 使用 JoyShockLibrary

基于 joy.py，使用底层库的融合四元数数据驱动 3D 立方体旋转。

运行:
    python cube_3d_madgwick.py

按 Ctrl+C 退出。
"""

import ctypes
import time
import math
import numpy as np
from vpython import scene, box, vector, rate, color, norm, cylinder


# 1. 定义 C++ 中的 MOTION_STATE 结构体
class MOTION_STATE(ctypes.Structure):
    _fields_ = [
        ("quatW", ctypes.c_float), ("quatX", ctypes.c_float),
        ("quatY", ctypes.c_float), ("quatZ", ctypes.c_float),
        ("accelX", ctypes.c_float), ("accelY", ctypes.c_float), ("accelZ", ctypes.c_float),
        ("gravX", ctypes.c_float), ("gravY", ctypes.c_float), ("gravZ", ctypes.c_float)
    ]


# 2. 加载 DLL 文件
jsl = ctypes.CDLL('./JoyShockLibrary.dll')
jsl.JslGetMotionState.restype = MOTION_STATE


def connect_controller():
    """连接手柄"""
    num_devices = jsl.JslConnectDevices()
    if num_devices == 0:
        print("没有检测到手柄！请确保已通过蓝牙或USB连接。")
        exit()
    
    device_handles = (ctypes.c_int * 4)()
    jsl.JslGetConnectedDeviceHandles(device_handles, 4)
    device_id = device_handles[0]
    
    print("手柄连接成功！")
    time.sleep(2)
    return device_id


def quat_to_euler(w, x, y, z):
    """四元数转欧拉角 (Roll, Pitch, Yaw)"""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角 (deg) 转换为旋转矩阵 (ZYX 顺序)"""
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    cos_r, sin_r = math.cos(r), math.sin(r)
    cos_p, sin_p = math.cos(p), math.sin(p)
    cos_y, sin_y = math.cos(y), math.sin(y)

    Rz = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])

    R_matrix = Rz @ Ry @ Rx
    return R_matrix


def apply_rotation_to_box(box_obj, rotation_matrix):
    """根据旋转矩阵更新立方体的方向"""
    axis = vector(*rotation_matrix[:, 0])
    up = vector(*rotation_matrix[:, 2])
    box_obj.axis = norm(axis)
    box_obj.up = norm(up)


def main():
    # 连接控制器
    device_id = connect_controller()

    # 配置 vpython 场景
    scene.width = 900
    scene.height = 700
    scene.background = vector(0.2, 0.2, 0.2)
    scene.forward = vector(0, 0, -1)
    scene.range = 2

    # 创建立方体
    cube = box(pos=vector(0, 0, 0), length=1, height=1, width=1,
               color=color.blue, opacity=0.9)

    # 添加辅助坐标轴
    axis_scale = 1.2
    axis_x = cylinder(pos=vector(0, 0, 0), axis=vector(axis_scale, 0, 0), radius=0.05, color=color.red)
    axis_y = cylinder(pos=vector(0, 0, 0), axis=vector(0, axis_scale, 0), radius=0.05, color=color.green)
    axis_z = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, axis_scale), radius=0.05, color=color.blue)

    print("\n3D 立方体已创建。请摇晃手柄观察立方体旋转。")
    print("按 Ctrl+C 停止...\n")

    try:
        frame_count = 0
        last_caption_update = time.time()

        while True:
            rate(100)  # 100 Hz 更新率

            # 读取融合的四元数数据
            motion = jsl.JslGetMotionState(device_id)

            # 转换为欧拉角
            roll, pitch, yaw = quat_to_euler(motion.quatW, motion.quatX, motion.quatY, motion.quatZ)

            # 计算旋转矩阵
            rot_matrix = euler_to_rotation_matrix(roll, pitch, yaw)

            # 应用到立方体
            apply_rotation_to_box(cube, rot_matrix)

            # 旋转坐标轴
            axis_x.axis = vector(*rot_matrix[:, 0]) * axis_scale
            axis_y.axis = vector(*rot_matrix[:, 1]) * axis_scale
            axis_z.axis = vector(*rot_matrix[:, 2]) * axis_scale

            frame_count += 1
            # 每 20 帧更新一次标题
            now = time.time()
            if now - last_caption_update > 0.2:
                scene.caption = f"Rx: {roll:6.1f}° | Ry: {pitch:6.1f}° | Rz: {yaw:6.1f}°"
                last_caption_update = now
                if frame_count % 100 == 0:
                    print(f"末端映射 -> Rx(Roll): {roll:6.1f}°, Ry(Pitch): {pitch:6.1f}°, Rz(Yaw): {yaw:6.1f}°")

    except KeyboardInterrupt:
        print("\n停止...")
        jsl.JslDisconnectAndDisposeAll()
        print("已断开连接。")


if __name__ == '__main__':
    main()
