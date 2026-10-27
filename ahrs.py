"""手柄姿态解算 (AHRS) 算法集合。

可选算法：

| 名称 | 说明 |
|---|---|
| ``jsl``       | JoyShockLibrary 内置融合 (GamepadMotion 互补滤波)，即仓库原来的做法 (DLL 计算) |
| ``jsl_fixed`` | 同一算法的 Python 移植，套用上游 GamepadMotionHelpers 修复后的重力修正 |
| ``ekf``       | 航姿扩展卡尔曼滤波 (EKF-AHRS)，状态 = 四元数 + 陀螺零偏 (7 维，加性更新) |
| ``eskf``      | 误差状态卡尔曼滤波 (ESKF / MEKF)，误差状态 = 姿态误差角 + 零偏误差 (6 维) |
| ``madgwick``  | Madgwick 梯度下降互补滤波 (IMU 版本) |
| ``mahony``    | Mahony 显式互补滤波 (PI 修正，积分项估计零偏) |

约定：

- 四元数统一用 ``(w, x, y, z)``，Hamilton 乘法，表示 "机体 -> 世界" 的旋转。
- 机体坐标系与 JoyShockLibrary 的 IMU 坐标系一致 (X 右, Y 上, Z 朝向玩家)。
- 陀螺仪输入单位 deg/s，加速度计输入单位 g；静止时加速度计读数指向 "上"。
- ``ekf`` / ``eskf`` / ``madgwick`` / ``mahony`` 在内部以 Z 轴朝上的世界系计算，输出时统一
  转换到 JoyShockLibrary 的世界系 (Y 轴朝上)，所有算法的输出可以直接互换。

DualSense 没有磁力计，所以所有算法的航向 (yaw) 都只能靠陀螺积分。
EKF / ESKF 通过在线估计陀螺零偏 + 静止零角速度修正 (ZARU) 来抑制航向漂移。
"""

from __future__ import annotations

import math

import numpy as np

AHRS_ALGORITHMS = ("jsl", "jsl_fixed", "ekf", "eskf", "madgwick", "mahony")

DEG2RAD = math.pi / 180.0


# ==========================================
# 四元数工具 (w, x, y, z)
# ==========================================
def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle
    return np.concatenate(([math.cos(half)], axis / norm * math.sin(half)))


def quat_from_rotvec(rotvec) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    return quat_from_axis_angle(rotvec, float(np.linalg.norm(rotvec)))


def quat_rotate(q: np.ndarray, v) -> np.ndarray:
    """用 q 旋转向量 v，即 q * v * q^-1。"""
    return quat_mul(quat_mul(q, np.concatenate(([0.0], v))), quat_conj(q))[1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_from_two_vectors(u, v) -> np.ndarray:
    """最短弧旋转：把单位向量 u 转到单位向量 v。"""
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    d = float(np.dot(u, v))
    if d < -0.999999:
        axis = np.cross([1.0, 0.0, 0.0], u)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], u)
        return quat_from_axis_angle(axis, math.pi)
    return quat_normalize(np.concatenate(([1.0 + d], np.cross(u, v))))


# 内部 Z-up 世界系 -> JoyShockLibrary 的 Y-up 世界系 (绕 X 轴 -90°，把 +Z 转到 +Y)
ZUP_TO_JSL_WORLD = quat_from_axis_angle([1.0, 0.0, 0.0], -math.pi / 2)


def jsl_convention(q_wxyz) -> np.ndarray:
    """真实姿态 <-> JSL 内置四元数约定 的互相转换 (自逆)。

    JoyShockLibrary 内置的 GamepadMotionHelpers 有一个已知 bug：重力修正时把
    "机体->世界" 的四元数当成了 "世界->机体" 来用，导致输出的 pitch/roll 方向是反的
    (上游 GamepadMotionHelpers 已在 commit a061205 "Fix GetOrientation() returning
    inverted pitch/roll" 中修复，但 JoyShockLibrary 打包的旧版本没有同步)。
    在水平参考附近，其相对旋转约等于真实相对旋转绕 Y 轴 180° 的相似变换，
    即四元数 (w, x, y, z) -> (w, -x, y, -z)。

    servo_control.py 里的 R_map 是基于 JSL 输出调好的，所以 EKF / Madgwick
    输出时也转换到这个约定，切换算法后机械臂的运动方向保持不变。
    """
    w, x, y, z = q_wxyz
    return np.array([w, -x, y, -z], dtype=float)


def _right_mul_matrix(p: np.ndarray) -> np.ndarray:
    """返回矩阵 M，使得 q ⊗ p = M @ q。"""
    pw, px, py, pz = p
    return np.array([
        [pw, -px, -py, -pz],
        [px, pw, pz, -py],
        [py, -pz, pw, px],
        [pz, py, -px, pw],
    ])


def _xi_matrix(q: np.ndarray) -> np.ndarray:
    """q ⊗ [0, ω] = Ξ(q) @ ω。"""
    w, x, y, z = q
    return np.array([
        [-x, -y, -z],
        [w, -z, y],
        [z, w, -x],
        [-y, x, w],
    ])


# ==========================================
# 基类
# ==========================================
class AHRSFilter:
    """统一接口：update(gyro_dps, accel_g, dt) -> 机体->JSL世界 四元数 (w, x, y, z)。"""

    name = "base"

    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 机体 -> Z-up 世界
        self.initialized = False

    def reset(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.initialized = False

    def _initialize(self, accel: np.ndarray) -> None:
        """用第一帧加速度对齐重力方向 (俯仰/横滚)，航向取 0。"""
        if np.linalg.norm(accel) < 1e-6:
            return
        self.q = quat_from_two_vectors(accel, [0.0, 0.0, 1.0])
        self.initialized = True

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        q = quat_mul(ZUP_TO_JSL_WORLD, self.q)
        if q[0] < 0:
            q = -q
        return tuple(float(v) for v in q)

    def update(self, gyro_dps, accel_g, dt: float) -> tuple[float, float, float, float]:
        gyro = np.asarray(gyro_dps, dtype=float) * DEG2RAD
        accel = np.asarray(accel_g, dtype=float)
        if not self.initialized:
            self._initialize(accel)
            if not self.initialized:
                return self.quaternion_wxyz
        if dt > 0.0:
            self._step(gyro, accel, float(dt))
        return self.quaternion_wxyz

    def _step(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        raise NotImplementedError


# ==========================================
# EKF / ESKF 共用部分
# ==========================================
class _GravityAidedKalman(AHRSFilter):
    """EKF 与 ESKF 共用的噪声参数、晃动度、加速度门限和静止检测。

    两者只在状态表示和线性化方式上不同，其余处理完全一致，便于公平对比。

    初始姿态协方差要取小 (默认 1°)：没有磁力计时航向不可观，过大的初始方差不会被观测
    收敛，随机体转动混入机体系各轴后，会通过互协方差让重力修正 "漏" 进航向。仿真中把
    初始方差放到约 11° 时，ESKF 的遥操作相对姿态误差会从 0.4° 恶化到约 3°。
    子类实现 _reset_covariance / _predict / _update_accel / _update_zaru。
    """

    def __init__(
        self,
        gyro_noise_dps: float = 0.3,        # 陀螺角速度白噪声 (deg/s, 1σ)
        gyro_bias_walk_dps: float = 0.02,   # 零偏随机游走 (deg/s/√s)
        accel_noise_g: float = 0.03,        # 加速度方向观测噪声 (g)
        accel_adaptive_gain: float = 40.0,  # |a|-1 偏差对观测噪声的放大系数
        accel_reject_g: float = 0.25,       # |a| 偏离 1g 超过该值则跳过重力修正
        shake_ref_g: float = 0.05,          # 晃动度对观测噪声的放大参考值
        shake_reject_g: float = 0.3,        # 晃动度超过该值则跳过重力修正
        gyro_reject_dps: float = 400.0,     # 角速度超过该值则跳过重力修正
        initial_attitude_std_deg: float = 1.0,  # 初始姿态不确定度 (加速度对准后的倾角误差，航向按定义为 0)
        initial_bias_std_dps: float = 1.0,
        zaru: bool = True,
        still_gyro_dps: float = 2.0,        # 静止判定：角速度阈值
        still_accel_g: float = 0.04,        # 静止判定：晃动度阈值
        still_time_s: float = 0.4,          # 持续多久才认为静止
    ):
        super().__init__()
        self.sigma_g = gyro_noise_dps * DEG2RAD
        self.sigma_bw = gyro_bias_walk_dps * DEG2RAD
        self.sigma_a = accel_noise_g
        self.accel_adaptive_gain = accel_adaptive_gain
        self.accel_reject_g = accel_reject_g
        self.shake_ref = shake_ref_g
        self.shake_reject = shake_reject_g
        self.gyro_reject = gyro_reject_dps * DEG2RAD
        self.initial_attitude_std = initial_attitude_std_deg * DEG2RAD
        self.initial_bias_std = initial_bias_std_dps * DEG2RAD
        self.zaru = zaru
        self.still_gyro = still_gyro_dps * DEG2RAD
        self.still_accel = still_accel_g
        self.still_time = still_time_s
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.bias = np.zeros(3)
        self._reset_covariance()
        self._still_timer = 0.0
        self._smooth_accel: np.ndarray | None = None
        self.shakiness = 0.0
        self.is_still = False

    @property
    def gyro_bias_dps(self) -> np.ndarray:
        return self.bias / DEG2RAD

    def _step(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        dq = self._predict(gyro, dt)
        self._track_shakiness(accel, dt, dq)
        if np.linalg.norm(gyro) < self.gyro_reject and self.shakiness < self.shake_reject:
            norm = float(np.linalg.norm(accel))
            deviation = abs(norm - 1.0)
            if norm > 1e-6 and deviation <= self.accel_reject_g:
                sigma2 = (
                    self.sigma_a ** 2
                    * (1.0 + (self.accel_adaptive_gain * deviation) ** 2)
                    * (1.0 + (self.shakiness / self.shake_ref) ** 2)
                )
                self._update_accel(accel / norm, sigma2)
        if self.zaru and self._detect_still(gyro, accel, dt):
            # 零角速度修正：静止时 陀螺读数 = 零偏 + 噪声
            self._update_zaru(gyro)
        self.q = quat_normalize(self.q)
        if self.q[0] < 0:
            self.q = -self.q

    def _track_shakiness(self, accel: np.ndarray, dt: float, dq: np.ndarray) -> None:
        """晃动度：加速度偏离其平滑值的幅度，峰值保持并以 0.25 s 半衰期衰减。"""
        if self._smooth_accel is None:
            self._smooth_accel = accel.copy()
        else:
            # 平滑值跟随机体一起转动，单纯的转动不会被误判为晃动
            self._smooth_accel = quat_rotate(quat_conj(dq), self._smooth_accel)
        decay = 2.0 ** (-dt / 0.25)
        self.shakiness = max(self.shakiness * decay, float(np.linalg.norm(accel - self._smooth_accel)))
        self._smooth_accel = accel + (self._smooth_accel - accel) * decay

    def _detect_still(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> bool:
        still_now = (
            np.linalg.norm(gyro) < self.still_gyro
            and self.shakiness < self.still_accel
            and abs(np.linalg.norm(accel) - 1.0) < 0.1
        )
        self._still_timer = self._still_timer + dt if still_now else 0.0
        self.is_still = self._still_timer >= self.still_time
        return self.is_still

    def _reset_covariance(self) -> None:
        raise NotImplementedError

    def _predict(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError

    def _update_accel(self, accel_dir: np.ndarray, sigma2: float) -> None:
        raise NotImplementedError

    def _update_zaru(self, gyro: np.ndarray) -> None:
        raise NotImplementedError


# ==========================================
# 1. 航姿扩展卡尔曼滤波 EKF-AHRS
# ==========================================
class EKFAHRS(_GravityAidedKalman):
    """四元数 + 陀螺零偏的 7 维 EKF (四元数各分量直接作为状态，加性更新)。

    参考 waihekor/EKF_AHRS (MATLAB) 与 agilexrobotics Piper 手机遥操作 (main.py)，
    在其基础上做了以下改进：

    1. 预测用精确的四元数指数积分，状态转移雅可比包含 "四元数-零偏" 耦合项
       (参考代码里 A_12 = 0，零偏只能靠观测间接修正)。
    2. 过程噪声由陀螺噪声密度推导，并随 dt 缩放，读取频率变化时不需要重新调参。
    3. 加速度观测噪声自适应：|a| 偏离 1g 越多、"晃动度" (加速度相对其平滑值的波动，
       参考 JSL 的 Shakiness) 越大、角速度越快，越不信任加速度计；超过阈值直接跳过
       本帧重力修正。剧烈乱晃时主要靠陀螺积分，避免平移/离心加速度把姿态 "拽歪"，
       并通过协方差耦合把错误带进航向和零偏。
    4. 静止检测 + 零角速度修正 (ZARU)：手柄静止时把陀螺读数当作零偏的直接观测，
       使包括 Z 轴在内的三轴零偏都可观，显著降低无磁力计时的航向漂移。
    5. Joseph 形式协方差更新 + 对称化，数值更稳定。
    """

    name = "ekf"

    def _reset_covariance(self) -> None:
        # 小角度下四元数虚部 ≈ 误差角 / 2
        self.P = np.diag([(0.5 * self.initial_attitude_std) ** 2] * 4 + [self.initial_bias_std ** 2] * 3)

    def _predict(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.q
        omega = gyro - self.bias
        dq = quat_from_rotvec(omega * dt)

        F = np.eye(7)
        F[:4, :4] = _right_mul_matrix(dq)
        xi = _xi_matrix(q)
        F[:4, 4:] = -0.5 * dt * xi

        Q = np.zeros((7, 7))
        Q[:4, :4] = (0.5 * dt * self.sigma_g) ** 2 * (xi @ xi.T)
        Q[4:, 4:] = np.eye(3) * (self.sigma_bw ** 2 * dt)

        self.q = quat_normalize(quat_mul(q, dq))
        self.P = F @ self.P @ F.T + Q
        return dq

    def _measurement_update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        P = self.P
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        dx = K @ residual
        self.q = self.q + dx[:4]
        self.bias = self.bias + dx[4:]
        I_KH = np.eye(7) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        self.q = quat_normalize(self.q)

    def _update_accel(self, accel_dir: np.ndarray, sigma2: float) -> None:
        w, x, y, zq = self.q
        # 观测模型：机体系下的 "上" 方向 = R(q)^T [0, 0, 1]
        h = np.array([
            2.0 * (x * zq - w * y),
            2.0 * (w * x + y * zq),
            w * w - x * x - y * y + zq * zq,
        ])
        H = np.zeros((3, 7))
        H[:, :4] = 2.0 * np.array([
            [-y, zq, -w, x],
            [x, w, zq, y],
            [w, -x, -y, zq],
        ])
        self._measurement_update(accel_dir - h, H, np.eye(3) * sigma2)

    def _update_zaru(self, gyro: np.ndarray) -> None:
        H = np.zeros((3, 7))
        H[:, 4:] = np.eye(3)
        self._measurement_update(gyro - self.bias, H, np.eye(3) * self.sigma_g ** 2)


# ==========================================
# 2. 误差状态卡尔曼滤波 ESKF (乘性 EKF / MEKF)
# ==========================================
class ESKFAHRS(_GravityAidedKalman):
    """误差状态卡尔曼滤波：名义状态 (q, b) 用非线性方程传播，
    滤波器只估计小量误差 δx = [δθ (机体系姿态误差角), δb]，6 维。

    相比 7 维加性 EKF：
    - 姿态误差用 3 维旋转向量表示，没有四元数归一化约束带来的冗余维度和奇异协方差；
    - 修正以乘性方式注入 q ← q ⊗ exp(δθ)，始终保持单位四元数；
    - 误差状态始终在零附近，线性化更准确。
    参考 J. Solà, "Quaternion kinematics for the error-state Kalman filter", 2017。
    噪声参数、晃动门限、ZARU 与 EKF 完全相同。
    """

    name = "eskf"

    def _reset_covariance(self) -> None:
        self.P = np.diag([self.initial_attitude_std ** 2] * 3 + [self.initial_bias_std ** 2] * 3)

    def _predict(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        omega = gyro - self.bias
        dq = quat_from_rotvec(omega * dt)

        F = np.eye(6)
        F[:3, :3] = quat_to_matrix(dq).T     # δθ_k+1 = R(ω dt)^T δθ_k - δb dt
        F[:3, 3:] = -np.eye(3) * dt
        Q = np.diag([(self.sigma_g * dt) ** 2] * 3 + [self.sigma_bw ** 2 * dt] * 3)

        self.q = quat_normalize(quat_mul(self.q, dq))
        self.P = F @ self.P @ F.T + Q
        return dq

    def _measurement_update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        P = self.P
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        dx = K @ residual
        I_KH = np.eye(6) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        # 注入误差并重置：q ← q ⊗ exp(δθ)，b ← b + δb，P ← G P Gᵀ
        self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(dx[:3])))
        self.bias = self.bias + dx[3:]
        G = np.eye(6)
        G[:3, :3] -= 0.5 * skew(dx[:3])
        P = G @ P @ G.T
        self.P = 0.5 * (P + P.T)

    def _update_accel(self, accel_dir: np.ndarray, sigma2: float) -> None:
        # h = R(q)^T e_z；真实姿态 q ⊗ exp(δθ) 下 h ≈ h + [h]× δθ
        h = quat_to_matrix(self.q)[2]
        H = np.zeros((3, 6))
        H[:, :3] = skew(h)
        self._measurement_update(accel_dir - h, H, np.eye(3) * sigma2)

    def _update_zaru(self, gyro: np.ndarray) -> None:
        H = np.zeros((3, 6))
        H[:, 3:] = np.eye(3)
        self._measurement_update(gyro - self.bias, H, np.eye(3) * self.sigma_g ** 2)


# ==========================================
# 3. Madgwick 梯度下降 AHRS (IMU 版本)
# ==========================================
class MadgwickAHRS(AHRSFilter):
    """S. Madgwick, "An efficient orientation filter for inertial and
    inertial/magnetic sensor arrays", 2010。

    q̇ = 0.5 q ⊗ ω - β · ∇f / |∇f|，用梯度下降把估计的重力方向拉向加速度计方向。
    β 越大越信任加速度计 (收敛快、易受平移加速度干扰)，越小越信任陀螺。
    """

    name = "madgwick"

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def _step(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        q_dot = 0.5 * np.array([
            -q1 * gx - q2 * gy - q3 * gz,
            q0 * gx + q2 * gz - q3 * gy,
            q0 * gy - q1 * gz + q3 * gx,
            q0 * gz + q1 * gy - q2 * gx,
        ])

        norm = float(np.linalg.norm(accel))
        if norm > 1e-6:
            ax, ay, az = accel / norm
            # f = R(q)^T [0,0,1] - a，J = ∂f/∂q；step = J^T f
            f = np.array([
                2.0 * (q1 * q3 - q0 * q2) - ax,
                2.0 * (q0 * q1 + q2 * q3) - ay,
                2.0 * (0.5 - q1 * q1 - q2 * q2) - az,
            ])
            J = np.array([
                [-2.0 * q2, 2.0 * q3, -2.0 * q0, 2.0 * q1],
                [2.0 * q1, 2.0 * q0, 2.0 * q3, 2.0 * q2],
                [0.0, -4.0 * q1, -4.0 * q2, 0.0],
            ])
            step = J.T @ f
            step_norm = float(np.linalg.norm(step))
            if step_norm > 1e-12:
                q_dot -= self.beta * step / step_norm

        self.q = quat_normalize(self.q + q_dot * dt)


# ==========================================
# 4. Mahony 显式互补滤波
# ==========================================
class MahonyAHRS(AHRSFilter):
    """R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary Filters on the
    Special Orthogonal Group", IEEE TAC, 2008 (IMU 版本)。

    误差 e = a × v (测量重力方向 × 估计重力方向)，ω' = ω + Kp·e + Ki·∫e，
    比例项修正姿态，积分项相当于在线估计陀螺零偏 (只对水平两轴可观，航向零偏不可观)。
    """

    name = "mahony"

    def __init__(self, kp: float = 0.5, ki: float = 0.02):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.integral = np.zeros(3)

    @property
    def gyro_bias_dps(self) -> np.ndarray:
        return -self.integral / DEG2RAD

    def _step(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        omega = gyro.copy()
        norm = float(np.linalg.norm(accel))
        if norm > 1e-6:
            v = quat_to_matrix(self.q)[2]   # 估计的机体系 "上" 方向
            e = np.cross(accel / norm, v)
            if self.ki > 0.0:
                self.integral += self.ki * e * dt
            omega = omega + self.kp * e + self.integral
        self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(omega * dt)))


# ==========================================
# 5. JoyShockLibrary 内置融合的 Python 移植 (原版 / 上游修复版)
# ==========================================
class JSLGamepadMotion:
    """GamepadMotionHelpers::Motion::Update 的逐行移植 (JoyShockLibrary 内置算法)。

    - ``fixed=False``：与 JoyShockLibrary 打包的旧版一致 (pitch/roll 取反的 bug)。
      实机上 ``jsl`` 选项直接读取 DLL 里的四元数，这个版本只用于离线对比。
    - ``fixed=True``：套用上游 GamepadMotionHelpers commit a061205 的修复 (重力修正的
      三行改动)，即 ``jsl_fixed`` 选项，可以直接在 joy.py 里使用。
    输入：陀螺 deg/s (已标定)，加速度 g；输出：机体 -> JSL 世界 (Y 轴朝上)。
    """

    SHAKINESS_MIN = 0.01
    SHAKINESS_MAX = 0.4
    STILL_SPEED = 1.0
    SHAKY_SPEED = 0.1
    GYRO_FACTOR = 0.1
    GYRO_MIN = 0.05
    GYRO_MAX = 0.25
    MIN_SPEED = 0.01
    SHORT_STEADINESS_HALF_TIME = 0.25

    def __init__(self, fixed: bool = False):
        self.fixed = fixed
        self.name = "jsl_fixed" if fixed else "jsl"
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.grav = np.zeros(3)
        self.smooth_accel = np.zeros(3)
        self.shakiness = 0.0

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        q = self.q if self.q[0] >= 0 else -self.q
        return tuple(float(v) for v in q)

    def update(self, gyro_dps, accel_g, dt: float) -> tuple[float, float, float, float]:
        axis = np.asarray(gyro_dps, dtype=float)
        accel = np.asarray(accel_g, dtype=float)
        angle_speed = float(np.linalg.norm(axis)) * DEG2RAD
        rotation = quat_from_axis_angle(axis, angle_speed * dt)
        rotation_inv = quat_conj(rotation)
        self.q = quat_mul(self.q, rotation)

        accel_mag = float(np.linalg.norm(accel))
        if accel_mag > 0.0:
            accel_norm = accel / accel_mag
            self.smooth_accel = quat_rotate(rotation_inv, self.smooth_accel)
            smooth_factor = 2.0 ** (-dt / self.SHORT_STEADINESS_HALF_TIME)
            self.shakiness *= smooth_factor
            self.shakiness = max(self.shakiness, float(np.linalg.norm(accel - self.smooth_accel)))
            self.smooth_accel = accel + (self.smooth_accel - accel) * smooth_factor

            self.grav = quat_rotate(rotation_inv, self.grav)
            grav_to_accel = accel_norm * -accel_mag - self.grav
            grav_to_accel_len = float(np.linalg.norm(grav_to_accel))
            grav_to_accel_dir = grav_to_accel / grav_to_accel_len if grav_to_accel_len > 0 else grav_to_accel

            t = np.clip((self.shakiness - self.SHAKINESS_MIN) / (self.SHAKINESS_MAX - self.SHAKINESS_MIN), 0.0, 1.0)
            correction_speed = self.STILL_SPEED + (self.SHAKY_SPEED - self.STILL_SPEED) * t
            gyro_limit = max(angle_speed * self.GYRO_FACTOR, self.MIN_SPEED)
            if correction_speed > gyro_limit:
                close_enough = np.clip((grav_to_accel_len - self.GYRO_MIN) / (self.GYRO_MAX - self.GYRO_MIN), 0.0, 1.0)
                correction_speed = gyro_limit + (correction_speed - gyro_limit) * close_enough
            delta = grav_to_accel_dir * correction_speed * dt
            if np.dot(delta, delta) < grav_to_accel_len ** 2:
                self.grav = self.grav + delta
            else:
                self.grav = accel_norm * -accel_mag

            grav_len = float(np.linalg.norm(self.grav))
            grav_dir = self.grav / grav_len if grav_len > 0 else self.grav
            down = np.array([0.0, -1.0, 0.0])
            if self.fixed:
                # 上游修复版：Vec * Quat == q * v * q^-1 (世界系重力方向)，在世界系左乘修正
                gravity_direction = quat_rotate(self.q, grav_dir)
                error_angle = math.acos(float(np.clip(np.dot(down, gravity_direction), -1.0, 1.0)))
                flattened = np.cross(gravity_direction, down)
                self.q = quat_mul(quat_from_axis_angle(flattened, error_angle), self.q)
            else:
                # JoyShockLibrary 打包的旧版：Vec * Quat.Inverse() == q^-1 * v * q
                gravity_direction = quat_rotate(quat_conj(self.q), grav_dir)
                error_angle = math.acos(float(np.clip(np.dot(down, gravity_direction), -1.0, 1.0)))
                flattened = np.cross(down, gravity_direction)
                self.q = quat_mul(self.q, quat_from_axis_angle(flattened, error_angle))
        else:
            self.grav = quat_rotate(rotation_inv, self.grav)

        self.q = quat_normalize(self.q)
        self.initialized = True
        return self.quaternion_wxyz


def create_ahrs(name: str, **kwargs) -> AHRSFilter | JSLGamepadMotion | None:
    """根据名称创建滤波器；``jsl`` 返回 None，表示直接使用 DLL 内置四元数。"""
    name = name.lower()
    if name == "jsl":
        return None
    if name == "jsl_fixed":
        return JSLGamepadMotion(fixed=True)
    if name == "ekf":
        return EKFAHRS(**kwargs)
    if name == "eskf":
        return ESKFAHRS(**kwargs)
    if name == "madgwick":
        return MadgwickAHRS(**kwargs)
    if name == "mahony":
        return MahonyAHRS(**kwargs)
    raise ValueError(f"未知的姿态解算算法: {name!r}，可选: {', '.join(AHRS_ALGORITHMS)}")
