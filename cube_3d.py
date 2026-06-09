"""
PS5 DualSense 3D 立方体姿态显示

使用 DualSense 的陀螺仪和加速度计数据通过互补滤波计算 RPY，
用 vpython 渲染一个立方体，根据手柄姿态旋转立方体。

运行:
    pip install -r requirements.txt vpython
    python cube_3d.py

按 Ctrl+C 退出。
"""

import time
import threading
import math
from collections import deque

import numpy as np
from vpython import scene, box, vector, rate, color, norm

from pydualsense import pydualsense

# 陀螺仪缩放因子（LSB to deg/s）
GYRO_SCALE = 1.0 / 16.4
GYRO_SCALE_Z = 0.1


class ComplementaryFilter:
    """互补滤波器：融合加速度计和陀螺仪"""
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro_bias = (0.0, 0.0, 0.0)
        self.accel_ref = (0.0, 0.0, 0.0)
        self.runtime_bias = [0.0, 0.0, 0.0]
        self.bias_adapt_rate = 0.01

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """
        更新滤波器状态
        gx, gy, gz: deg/s
        ax, ay, az: g
        dt: 时间差
        返回: (roll, pitch, yaw) in degrees
        """
        # 加速度计计算 roll/pitch（不受陀螺积分漂移影响）
        accel_roll = math.degrees(math.atan2(ay, az))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

        # 互补滤波：陀螺积分 + 加速度修正
        self.roll = self.alpha * (self.roll + gx * dt) + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * accel_pitch
        # yaw 只能通过陀螺积分（没有地磁计）
        self.yaw += gz * dt
        # 包装 yaw 到 [-180, 180]
        self.yaw = (self.yaw + 180.0) % 360.0 - 180.0

        return self.roll, self.pitch, self.yaw


def read_sensor_state(ds):
    """读取传感器状态"""
    g = getattr(ds.state, 'gyro', None)
    a = getattr(ds.state, 'accelerometer', None)

    gx = gy = gz = ax = ay = az = 0.0

    if g is not None:
        if hasattr(g, 'Pitch'):
            # 轴映射：Pitch->gx, Roll->gy, Yaw->gz
            gx = float(getattr(g, 'Pitch', 0.0))
            gy = float(getattr(g, 'Roll', 0.0))
            gz = float(getattr(g, 'Yaw', 0.0))
        else:
            gx = float(getattr(g, 'x', 0.0))
            gy = float(getattr(g, 'y', 0.0))
            gz = float(getattr(g, 'z', 0.0))

    if a is not None:
        if hasattr(a, 'X'):
            ax = float(getattr(a, 'X', 0.0))
            ay = float(getattr(a, 'Y', 0.0))
            az = float(getattr(a, 'Z', 0.0))
        else:
            ax = float(getattr(a, 'x', 0.0))
            ay = float(getattr(a, 'y', 0.0))
            az = float(getattr(a, 'z', 0.0))

    # 应用陀螺仪缩放
    gx *= GYRO_SCALE
    gy *= GYRO_SCALE
    gz *= GYRO_SCALE * GYRO_SCALE_Z

    return gx, gy, gz, ax, ay, az


def calibrate(ds, duration=3.0):
    """校准：读取 N 秒数据计算偏置"""
    print(f'开始校准：请将手柄平放并保持稳定 {duration} 秒...')
    start = time.time()
    gx_list, gy_list, gz_list = [], [], []
    ax_list, ay_list, az_list = [], [], []

    while time.time() - start < duration:
        gx, gy, gz, ax, ay, az = read_sensor_state(ds)
        gx_list.append(gx)
        gy_list.append(gy)
        gz_list.append(gz)
        ax_list.append(ax)
        ay_list.append(ay)
        az_list.append(az)
        time.sleep(0.01)

    gyro_bias = (np.mean(gx_list), np.mean(gy_list), np.mean(gz_list))
    accel_ref = (np.mean(ax_list), np.mean(ay_list), np.mean(az_list))

    print(f'校准完成：gyro_bias={gyro_bias}')
    return gyro_bias, accel_ref


def sensor_thread(ds, filt, state_dict, stop_event, sample_hz=100):
    """后台传感器读取线程"""
    dt_target = 1.0 / sample_hz
    last = time.time()
    frame_count = 0
    stationary_count = 0  # 连续静止帧计数

    while not stop_event.is_set():
        gx, gy, gz, ax, ay, az = read_sensor_state(ds)

        # 加强的自适应运行偏置：需要连续多帧都静止才更新
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        motion_thresh = 5.0  # 更严格的阈值
        is_stationary = accel_norm > 0.95 and accel_norm < 1.05 and max(abs(gx), abs(gy), abs(gz)) < motion_thresh
        
        if is_stationary:
            stationary_count += 1
            # 只在连续 20 帧都静止后才更新
            if stationary_count > 20:
                filt.runtime_bias[0] = (1 - 0.02) * filt.runtime_bias[0] + 0.02 * gx
                filt.runtime_bias[1] = (1 - 0.02) * filt.runtime_bias[1] + 0.02 * gy
                filt.runtime_bias[2] = (1 - 0.02) * filt.runtime_bias[2] + 0.02 * gz
        else:
            stationary_count = 0

        # 减去偏置
        gbx, gby, gbz = filt.gyro_bias
        rbx, rby, rbz = filt.runtime_bias
        gx -= (gbx + rbx)
        gy -= (gby + rby)
        gz -= (gbz + rbz)

        # 更新滤波器
        now = time.time()
        dt = now - last if last is not None else dt_target
        last = now

        r, p, y = filt.update(gx, gy, gz, ax, ay, az, dt)

        # 每 100 帧打印一次
        frame_count += 1
        if frame_count % 100 == 0:
            motion_status = "MOVING" if is_stationary == False else "STATIC"
            print(f'[Frame {frame_count}] RPY: R={r:.2f}°, P={p:.2f}°, Y={y:.2f}° | Raw: gx={gx:.2f}, gy={gy:.2f}, gz={gz:.2f} | Accel: {accel_norm:.3f}g | {motion_status}')

        # 更新全局状态
        state_dict['roll'] = r
        state_dict['pitch'] = p
        state_dict['yaw'] = y
        state_dict['accel_norm'] = accel_norm
        state_dict['motion'] = not is_stationary

        # 保持采样率
        elapsed = time.time() - now
        to_sleep = max(0.0, dt_target - elapsed)
        time.sleep(to_sleep)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角 (deg) 转换为旋转矩阵 (ZYX 顺序：Yaw-Pitch-Roll)
    """
    # 转换为弧度
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    # 旋转矩阵
    cos_r, sin_r = math.cos(r), math.sin(r)
    cos_p, sin_p = math.cos(p), math.sin(p)
    cos_y, sin_y = math.cos(y), math.sin(y)

    # ZYX 旋转矩阵
    # 先绕 Z 旋转 (yaw)，再绕 Y (pitch)，最后绕 X (roll)
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

    # 组合：R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def apply_rotation_to_box(box_obj, rotation_matrix):
    """
    根据旋转矩阵更新立方体的方向
    vpython 中的 axis/up 向量表示方向
    """
    # 提取主方向向量（旋转矩阵的列向量）
    axis = vector(*rotation_matrix[:, 0])  # 第一列
    up = vector(*rotation_matrix[:, 2])    # 第三列

    box_obj.axis = norm(axis)
    box_obj.up = norm(up)


def main():
    # 初始化 DualSense
    ds = pydualsense()
    ds.init()

    # 校准
    gyro_bias, accel_ref = calibrate(ds, duration=3.0)

    # 创建滤波器（使用更保守的 alpha，更依赖加速度计）
    filt = ComplementaryFilter(alpha=0.92)
    filt.gyro_bias = gyro_bias
    filt.accel_ref = accel_ref

    # 初始化 roll/pitch
    aax, aay, aaz = accel_ref
    # 计算重力方向的欧拉角
    init_roll = math.degrees(math.atan2(aay, aaz))
    init_pitch = math.degrees(math.atan2(-aax, math.sqrt(aay * aay + aaz * aaz)))
    filt.roll = init_roll
    filt.pitch = init_pitch
    print(f'初始 RPY: R={init_roll:.2f}°, P={init_pitch:.2f}°, Y=0.00°')
    print(f'加速度基准: ax={aax:.3f}, ay={aay:.3f}, az={aaz:.3f} (范数={math.sqrt(aax**2+aay**2+aaz**2):.3f}g)')

    # 启动传感器线程
    state_dict = {
        'roll': init_roll, 
        'pitch': init_pitch, 
        'yaw': 0.0,
        'accel_norm': 1.0,
        'motion': False
    }
    stop_event = threading.Event()
    th = threading.Thread(target=sensor_thread, args=(ds, filt, state_dict, stop_event), daemon=True)
    th.start()

    # 配置 vpython 场景
    scene.width = 800
    scene.height = 600
    # vpython 需要用 vector 而不是 tuple
    scene.background = vector(0.9, 0.9, 0.9)
    scene.forward = vector(0, 0, -1)
    scene.range = 3

    # 创建立方体
    cube = box(pos=vector(0, 0, 0), length=1, height=1, width=1,
               color=color.blue, opacity=0.8)

    # 添加文本提示（虽然 vpython 文本支持有限）
    print("\n3D 立方体已创建。请摇晃手柄观察立方体旋转。")
    print("按 Ctrl+C 停止...\n")

    try:
        frame_count = 0
        while True:
            rate(100)  # 100 Hz 更新率

            # 获取最新的 RPY
            roll = state_dict['roll']
            pitch = state_dict['pitch']
            yaw = state_dict['yaw']
            accel_norm = state_dict.get('accel_norm', 1.0)
            is_motion = state_dict.get('motion', False)

            # 计算旋转矩阵
            rot_matrix = euler_to_rotation_matrix(roll, pitch, yaw)

            # 应用到立方体
            apply_rotation_to_box(cube, rot_matrix)

            frame_count += 1
            if frame_count % 50 == 0:
                status = "Moving" if is_motion else "Stationary"
                scene.caption = f"Roll: {roll:.1f}° | Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}° | Accel: {accel_norm:.2f}g | {status}"

    except KeyboardInterrupt:
        print("\n停止...")
        stop_event.set()
        th.join(timeout=1.0)
        try:
            ds.close()
        except:
            pass
        print("已退出。")


if __name__ == '__main__':
    main()
