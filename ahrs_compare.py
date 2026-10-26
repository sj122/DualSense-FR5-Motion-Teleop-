"""三种姿态解算算法对比：JSL 内置融合 / EKF-AHRS / Madgwick。

用法：

    python ahrs_compare.py                    # 仿真对比，生成 docs/ahrs_comparison.png
    python ahrs_compare.py --runs 50          # 蒙特卡洛次数
    python ahrs_compare.py --record 30 --out imu.csv   # 用真实手柄录 30 秒数据 (Windows + DLL)
    python ahrs_compare.py --csv imu.csv      # 用录好的真实数据回放对比 (无真值，只画曲线)

仿真流程尽量贴近实机：

- 真值轨迹按 "桌面静置 -> 拿起 -> 慢速遥操作 -> 快速甩动 -> 平移加速度干扰 -> 大角度 -> 放下静置"
  分段生成，手持阶段叠加 8~12 Hz 生理震颤。
- 手柄 IMU 以 250 Hz 出数，陀螺带白噪声 + 残余零偏 (JSL 自动标定后剩下的部分) + 零偏随机游走，
  加速度计带白噪声，并包含手的平移加速度。
- JSL 内置算法按 250 Hz 逐帧运行 (与 DLL 内一致)；EKF / Madgwick 按 100 Hz 运行，
  输入是 JslGetAndFlushAccumulatedGyro 式的平均角速度 + 最新加速度 (与 joy.py 实机读取方式一致)。

评价指标 (都在 "真实物理约定" 下计算，JSL 输出先做 pitch/roll 镜像还原，见 joy.py 说明)：

- 倾角误差：估计的重力方向与真实重力方向的夹角 (roll/pitch 精度，与航向无关)。
- 遥操作相对姿态误差：以 t=5 s (按下 L2 的时刻) 为参考，ref⁻¹·q 与真值 ref⁻¹·q 的夹角，
  这正是 servo_control.py 下发给机械臂的量。
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ahrs import EKFAHRS, JSLGamepadMotion, MadgwickAHRS, jsl_convention  # noqa: E402

DEVICE_HZ = 250
APP_HZ = 100
DURATION = 60.0
CLUTCH_TIME = 5.0      # 遥操作参考时刻 (模拟按下 L2)
ALIGN_TIME = 3.0       # 航向对齐时刻 (各算法初始航向不同，只比较相对量)

ALGORITHMS = ("jsl", "ekf", "madgwick")
LABELS = {
    "truth": "真值",
    "jsl": "JSL 内置 (原方案)",
    "ekf": "EKF-AHRS (新)",
    "madgwick": "Madgwick",
}
# 分类色板 (前三个槽位在明/暗背景下都通过了色盲两两区分校验)
COLORS = {
    "jsl": "#2a78d6",
    "ekf": "#eb6834",
    "madgwick": "#1baf7a",
    "truth": "#0b0b0b",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

SEGMENTS = [
    (0.0, 3.0, "桌面静置"),
    (3.0, 5.0, "拿起"),
    (5.0, 23.0, "慢速遥操作"),
    (23.0, 31.0, "快速甩动"),
    (31.0, 39.0, "平移加速度干扰"),
    (39.0, 46.0, "大角度 75°"),
    (46.0, 48.0, "放下"),
    (48.0, 60.0, "桌面静置"),
]


# ==========================================
# 仿真数据
# ==========================================
def _min_jerk(t: np.ndarray, t0: float, duration: float) -> np.ndarray:
    s = np.clip((t - t0) / duration, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def _min_jerk_acc(t: np.ndarray, t0: float, duration: float) -> np.ndarray:
    s = np.clip((t - t0) / duration, 0.0, 1.0)
    inside = (t > t0) & (t < t0 + duration)
    return np.where(inside, (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2, 0.0)


def _band_noise(rng, t: np.ndarray, amp: float, f_lo: float = 8.0, f_hi: float = 12.0, n: int = 6) -> np.ndarray:
    out = np.zeros_like(t)
    for _ in range(n):
        f = rng.uniform(f_lo, f_hi)
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return amp * out / np.sqrt(n / 2)


def truth_trajectory(t: np.ndarray, rng) -> tuple[R, np.ndarray]:
    """返回 (机体->JSL世界 的真值姿态, 世界系平移加速度 m/s²)。

    欧拉角约定 (JSL 机体系 X右/Y上/Z朝向玩家，世界 Y 轴朝上)：
    yaw 绕世界 Y，pitch 绕机体 X (抬头为正)，roll 绕机体 Z，内旋顺序 Y-X-Z。
    """
    yaw = np.zeros_like(t)
    pitch = np.zeros_like(t)
    roll = np.zeros_like(t)

    def move(arr, t0, dur, delta):
        arr += delta * _min_jerk(t, t0, dur)

    # 拿起：微抬 15°
    move(pitch, 3.0, 1.5, 15)
    # 慢速遥操作 (单轴)
    move(pitch, 5.5, 1.5, 30); move(pitch, 7.5, 1.5, -30)
    move(roll, 9.0, 1.5, -35); move(roll, 11.0, 1.5, 35)
    move(yaw, 13.0, 2.0, 45); move(yaw, 15.5, 2.0, -45)
    # 慢速遥操作 (多轴组合)
    move(yaw, 18.0, 2.0, -30); move(pitch, 18.0, 2.0, 25); move(roll, 18.5, 2.0, 20)
    move(yaw, 20.5, 2.0, 30); move(pitch, 20.5, 2.0, -25); move(roll, 20.8, 2.0, -20)
    # 快速甩动 (峰值角速度 ~150-200°/s)
    fast = _min_jerk(t, 23.0, 0.5) * (1 - _min_jerk(t, 30.0, 0.8))
    pitch += fast * 25 * np.sin(2 * np.pi * 1.2 * (t - 23))
    roll += fast * 20 * np.sin(2 * np.pi * 1.5 * (t - 23) + 1.0)
    yaw += fast * 30 * np.sin(2 * np.pi * 0.9 * (t - 23) + 2.0)
    # 平移加速度干扰阶段：保持抬头 20°
    move(pitch, 31.0, 1.0, 20); move(pitch, 38.0, 1.0, -20)
    # 大角度
    move(pitch, 39.5, 2.5, 60); move(pitch, 43.5, 2.5, -60)
    # 放下
    move(pitch, 46.0, 1.5, -15)

    held = _min_jerk(t, 3.0, 0.5) * (1 - _min_jerk(t, 46.5, 1.0))
    yaw += held * _band_noise(rng, t, 0.12)
    pitch += held * _band_noise(rng, t, 0.15)
    roll += held * _band_noise(rng, t, 0.12)

    rot = R.from_euler("YXZ", np.column_stack([yaw, pitch, roll]), degrees=True)

    # 平移加速度：手快速推拉 15~20 cm，峰值约 0.5~0.7 g
    acc = np.zeros((len(t), 3))
    bursts = [
        (31.8, 0.4, [0.18, 0.0, 0.0]), (32.6, 0.4, [-0.18, 0.0, 0.0]),
        (33.6, 0.35, [0.0, 0.0, -0.15]), (34.3, 0.35, [0.0, 0.0, 0.15]),
        (35.2, 0.4, [0.0, 0.2, 0.0]), (35.9, 0.4, [0.0, -0.2, 0.0]),
        (36.7, 0.4, [0.12, 0.0, 0.12]), (37.3, 0.4, [-0.12, 0.0, -0.12]),
    ]
    for t0, dur, d in bursts:
        acc += np.outer(_min_jerk_acc(t, t0, dur), d)
    # 手持阶段的小幅平移抖动
    for axis in range(3):
        acc[:, axis] += held * _band_noise(rng, t, 0.15, 2.0, 6.0)
    return rot, acc


def simulate(seed: int, duration: float = DURATION) -> dict:
    rng = np.random.default_rng(seed)
    dt_dev = 1.0 / DEVICE_HZ
    t_dev = np.arange(0.0, duration + 1e-9, dt_dev)
    rot, acc_world = truth_trajectory(t_dev, rng)

    # 陀螺：相邻两帧之间的平均角速度 (机体系, deg/s)
    gyro = np.zeros((len(t_dev), 3))
    gyro[1:] = np.degrees((rot[:-1].inv() * rot[1:]).as_rotvec()) / dt_dev
    bias0 = rng.uniform(-0.3, 0.3, 3)
    bias_walk = np.cumsum(rng.normal(0.0, 0.003 * np.sqrt(dt_dev), (len(t_dev), 3)), axis=0)
    gyro_meas = gyro + bias0 + bias_walk + rng.normal(0.0, 0.15, gyro.shape)

    # 加速度计：比力 = 平移加速度 + "上" (单位 g)，转到机体系
    specific_force = acc_world / 9.81 + np.array([0.0, 1.0, 0.0])
    accel_meas = rot.inv().apply(specific_force) + rng.normal(0.0, 0.005, (len(t_dev), 3))

    t_app = np.arange(0.0, duration + 1e-9, 1.0 / APP_HZ)
    return {
        "t_dev": t_dev,
        "gyro_dev": gyro_meas,
        "accel_dev": accel_meas,
        "t_app": t_app,
        "truth_app": rot[np.searchsorted(t_dev, t_app - 1e-9).clip(0, len(t_dev) - 1)],
        "bias0": bias0,
    }


# ==========================================
# 运行各算法
# ==========================================
def run_filters(data: dict, ekf_kwargs: dict | None = None, madgwick_kwargs: dict | None = None) -> dict:
    t_dev, gyro_dev, accel_dev, t_app = data["t_dev"], data["gyro_dev"], data["accel_dev"], data["t_app"]
    dt_dev = t_dev[1] - t_dev[0]
    n_app = len(t_app)

    jsl = JSLGamepadMotion()
    ekf = EKFAHRS(**(ekf_kwargs or {}))
    mad = MadgwickAHRS(**(madgwick_kwargs or {}))
    out = {name: np.zeros((n_app, 4)) for name in ALGORITHMS}
    cost = {name: 0.0 for name in ALGORITHMS}

    j = 0
    gyro_sum = np.zeros(3)
    gyro_count = 0
    last_app_time = None
    for k, ta in enumerate(t_app):
        # 设备帧 -> JSL (250 Hz) + 累积陀螺
        while j < len(t_dev) and t_dev[j] <= ta + 1e-9:
            if j > 0:
                tic = time.perf_counter()
                jsl.update(gyro_dev[j], accel_dev[j], dt_dev)
                cost["jsl"] += time.perf_counter() - tic
            gyro_sum += gyro_dev[j]
            gyro_count += 1
            j += 1
        out["jsl"][k] = jsl.quaternion_wxyz

        # 应用层 100 Hz：平均角速度 + 最新加速度
        gyro_avg = gyro_sum / max(gyro_count, 1)
        gyro_sum[:] = 0.0
        gyro_count = 0
        accel = accel_dev[j - 1]
        dt = 0.0 if last_app_time is None else ta - last_app_time
        last_app_time = ta
        for name, filt in (("ekf", ekf), ("madgwick", mad)):
            tic = time.perf_counter()
            out[name][k] = filt.update(gyro_avg, accel, dt)
            cost[name] += time.perf_counter() - tic

    cost_us = {
        "jsl": cost["jsl"] / max(len(t_dev) - 1, 1) * 1e6,
        "ekf": cost["ekf"] / n_app * 1e6,
        "madgwick": cost["madgwick"] / n_app * 1e6,
    }
    return {"quats": out, "cost_us": cost_us}


def to_rotation(quats_wxyz: np.ndarray, from_jsl: bool) -> R:
    """转成真实物理约定下的 Rotation。JSL 的 pitch/roll 是镜像的，需要先还原。"""
    q = np.asarray(quats_wxyz)
    if from_jsl:
        q = np.array([jsl_convention(row) for row in q])
    return R.from_quat(q[:, [1, 2, 3, 0]])


def yaw_align(est: R, truth: R, t: np.ndarray, t_align: float) -> R:
    k = int(np.searchsorted(t, t_align))
    d_yaw = truth[k].as_euler("YXZ")[0] - est[k].as_euler("YXZ")[0]
    return R.from_euler("Y", d_yaw) * est


def evaluate(data: dict, result: dict) -> dict:
    t = data["t_app"]
    truth = data["truth_app"]
    up = np.array([0.0, 1.0, 0.0])
    g_true = truth.inv().apply(up)
    k_ref = int(np.searchsorted(t, CLUTCH_TIME))
    rel_true = truth[k_ref].inv() * truth
    mask = t >= CLUTCH_TIME

    metrics, series = {}, {}
    for name in ALGORITHMS:
        est = to_rotation(result["quats"][name], from_jsl=(name == "jsl"))
        est = yaw_align(est, truth, t, ALIGN_TIME)
        g_est = est.inv().apply(up)
        tilt_err = np.degrees(np.arccos(np.clip(np.sum(g_est * g_true, axis=1), -1.0, 1.0)))
        rel_est = est[k_ref].inv() * est
        rel_err = np.degrees((rel_true.inv() * rel_est).magnitude())
        series[name] = {"rot": est, "tilt_err": tilt_err, "rel_err": rel_err}
        settled = t >= 1.0
        metrics[name] = {
            "tilt_rmse": float(np.sqrt(np.mean(tilt_err[settled] ** 2))),
            "tilt_max": float(np.max(tilt_err[settled])),
            "rel_rmse": float(np.sqrt(np.mean(rel_err[mask] ** 2))),
            "rel_max": float(np.max(rel_err[mask])),
            "rel_final": float(rel_err[-1]),
            "cost_us": result["cost_us"][name],
        }
    return {"metrics": metrics, "series": series}


# ==========================================
# 画图
# ==========================================
def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    })
    return plt


def _shade_segments(ax, label: bool = False):
    for i, (a, b, name) in enumerate(SEGMENTS):
        if i % 2 == 1:
            ax.axvspan(a, b, color="#efeee9", lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=INK_2)


def plot_comparison(data: dict, ev: dict, mc: dict | None, path: Path) -> None:
    plt = setup_matplotlib()
    from matplotlib.gridspec import GridSpec

    t = data["t_app"]
    truth_euler = data["truth_app"].as_euler("YXZ", degrees=True)
    fig = plt.figure(figsize=(15, 13.5))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.1], hspace=0.6, wspace=0.25,
                  top=0.92, bottom=0.06)
    fig.suptitle("DualSense 姿态解算算法对比：JSL 内置融合 vs EKF-AHRS vs Madgwick",
                 fontsize=15, fontweight="bold", color=INK, y=0.985)

    # 第 1 行：roll / pitch / yaw 曲线 (真实物理约定)
    for col, (idx, title) in enumerate([(2, "Roll 横滚角"), (1, "Pitch 俯仰角"), (0, "Yaw 航向角")]):
        ax = fig.add_subplot(gs[0, col])
        _shade_segments(ax)
        for name in ALGORITHMS:
            e = ev["series"][name]["rot"].as_euler("YXZ", degrees=True)[:, idx]
            ax.plot(t, e, color=COLORS[name], lw=1.4, label=LABELS[name])
        ax.plot(t, truth_euler[:, idx], color=COLORS["truth"], lw=1.2, ls="--", label=LABELS["truth"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("角度 (°)")
        ax.set_xlim(t[0], t[-1])
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965), fontsize=10)

    # 第 2 行：倾角误差
    ax = fig.add_subplot(gs[1, :])
    _shade_segments(ax, label=True)
    for name in ALGORITHMS:
        ax.plot(t, ev["series"][name]["tilt_err"], color=COLORS[name], lw=1.3, label=LABELS[name])
    ax.set_title("倾角误差 (roll/pitch 合成误差，越小越好)", loc="left", pad=16)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("误差 (°)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

    # 第 3 行：遥操作相对姿态误差
    ax = fig.add_subplot(gs[2, :])
    _shade_segments(ax, label=True)
    ax.axvline(CLUTCH_TIME, color=INK_2, lw=1, ls=":")
    ax.text(CLUTCH_TIME + 0.3, 0.6, "按下 L2\n(参考时刻)", transform=ax.get_xaxis_transform(), fontsize=8, color=INK_2)
    for name in ALGORITHMS:
        ax.plot(t, ev["series"][name]["rel_err"], color=COLORS[name], lw=1.3, label=LABELS[name])
    ax.set_title("遥操作相对姿态误差 (下发给机械臂的 ref⁻¹·q，含航向漂移)", loc="left", pad=16)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("误差 (°)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

    # 第 4 行：统计柱状图
    stats = mc if mc is not None else {n: {k: (v, 0.0) for k, v in ev["metrics"][n].items()} for n in ALGORITHMS}
    groups = [("tilt_rmse", "倾角 RMSE"), ("tilt_max", "倾角最大误差"), ("rel_rmse", "相对姿态 RMSE"), ("rel_final", "60 s 末航向漂移")]
    ax = fig.add_subplot(gs[3, :2])
    width = 0.26
    x = np.arange(len(groups))
    for i, name in enumerate(ALGORITHMS):
        means = [stats[name][key][0] for key, _ in groups]
        stds = [stats[name][key][1] for key, _ in groups]
        bars = ax.bar(x + (i - 1) * width, means, width * 0.92, yerr=stds if mc else None,
                      color=COLORS[name], label=LABELS[name], error_kw={"ecolor": INK_2, "lw": 1, "capsize": 2})
        for rect, m, sd in zip(bars, means, stds):
            ax.text(rect.get_x() + rect.get_width() / 2, m + (sd if mc else 0.0), f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [g[1] for g in groups])
    ax.set_ylabel("误差 (°)")
    runs_note = f"（{mc['_runs']} 次蒙特卡洛平均 ± 标准差）" if mc else ""
    ax.set_title(f"误差统计{runs_note}", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[3, 2])
    names = ["ekf", "madgwick"]
    vals = [stats[n]["cost_us"][0] for n in names]
    bars = ax.barh([LABELS[n] for n in names], vals, color=[COLORS[n] for n in names], height=0.5)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, f" {v:.0f} µs", va="center", fontsize=8, color=INK)
    ax.set_title("单次更新耗时 (Python, 100 Hz 预算 10000 µs)", loc="left")
    ax.set_xlabel("µs / 次")
    ax.set_xlim(0, max(vals) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    fig.text(0.01, 0.003,
             "仿真条件：IMU 250 Hz，陀螺噪声 0.15°/s + 残余零偏 ±0.3°/s + 零偏随机游走，加速度噪声 0.005 g；"
             "EKF/Madgwick 以 100 Hz 运行 (与 joy.py 一致)，JSL 以 250 Hz 运行。"
             "JSL 输出已做 pitch/roll 镜像还原后再评估；各算法航向在 t=3 s 对齐。",
             fontsize=8, color=INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_recorded(rows: dict, path: Path) -> None:
    plt = setup_matplotlib()
    t = rows["t"] - rows["t"][0]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("真实手柄数据回放：三种算法姿态对比 (无真值)", fontsize=14, fontweight="bold", color=INK)
    rots = {name: to_rotation(rows[name], from_jsl=(name == "jsl")) for name in ALGORITHMS}
    k = int(np.searchsorted(t, min(ALIGN_TIME, t[-1])))
    ref = rots["ekf"]
    for name in ALGORITHMS:
        rots[name] = yaw_align(rots[name], ref, t, t[k])
    for ax, (idx, title) in zip(axes, [(2, "Roll 横滚角"), (1, "Pitch 俯仰角"), (0, "Yaw 航向角")]):
        for name in ALGORITHMS:
            ax.plot(t, rots[name].as_euler("YXZ", degrees=True)[:, idx], color=COLORS[name], lw=1.3, label=LABELS[name])
        ax.set_title(title, loc="left")
        ax.set_ylabel("角度 (°)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("时间 (s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 真实数据录制 / 回放
# ==========================================
CSV_FIELDS = ["t", "dt", "gx", "gy", "gz", "ax", "ay", "az", "jsl_w", "jsl_x", "jsl_y", "jsl_z"]


def record(seconds: float, out_path: Path) -> None:
    from joy import JoyShockDevice

    print("正在连接手柄，请先平放数秒...")
    with JoyShockDevice(ahrs="jsl") as device:
        device.connect_first_device(warmup_seconds=3.0)
        device.read_accumulated_gyro_dps()  # 清空预热期间累积的陀螺
        print(f"开始录制 {seconds:.0f} 秒，请按 静置 -> 慢速转动 -> 快速甩动 -> 平移 -> 放下 的顺序操作手柄。")
        last = time.perf_counter()
        start = last
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)
            while time.perf_counter() - start < seconds:
                time.sleep(max(0.0, 1.0 / APP_HZ - (time.perf_counter() - last)))
                now = time.perf_counter()
                gyro = device.read_accumulated_gyro_dps()
                imu = device.read_imu_state()
                motion = device.read_motion_state()
                writer.writerow([now - start, now - last, *gyro, imu.accelX, imu.accelY, imu.accelZ,
                                 motion.quatW, motion.quatX, motion.quatY, motion.quatZ])
                last = now
    print(f"已保存到 {out_path}")


def replay(csv_path: Path, fig_path: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    ekf, mad = EKFAHRS(), MadgwickAHRS()
    rows = {"t": data["t"], "jsl": np.column_stack([data["jsl_w"], data["jsl_x"], data["jsl_y"], data["jsl_z"]]),
            "ekf": np.zeros((len(data), 4)), "madgwick": np.zeros((len(data), 4))}
    for i, row in enumerate(data):
        gyro = (row["gx"], row["gy"], row["gz"])
        accel = (row["ax"], row["ay"], row["az"])
        dt = 0.0 if i == 0 else row["dt"]
        rows["ekf"][i] = ekf.update(gyro, accel, dt)
        rows["madgwick"][i] = mad.update(gyro, accel, dt)
    plot_recorded(rows, fig_path)
    print(f"EKF 估计的陀螺残余零偏 (deg/s): {np.round(ekf.gyro_bias_dps, 3)}")
    print(f"图已保存到 {fig_path}")


# ==========================================
# 主程序
# ==========================================
def monte_carlo(runs: int, seed: int) -> dict:
    keys = ["tilt_rmse", "tilt_max", "rel_rmse", "rel_max", "rel_final", "cost_us"]
    acc = {name: {k: [] for k in keys} for name in ALGORITHMS}
    for r in range(runs):
        data = simulate(seed + r)
        ev = evaluate(data, run_filters(data))
        for name in ALGORITHMS:
            for k in keys:
                acc[name][k].append(ev["metrics"][name][k])
    stats = {name: {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc[name].items()} for name in ALGORITHMS}
    stats["_runs"] = runs
    return stats


def print_table(stats: dict) -> None:
    header = f"| 指标 (°) | {' | '.join(LABELS[n] for n in ALGORITHMS)} |"
    print(header)
    print("|" + "---|" * (len(ALGORITHMS) + 1))
    rows = [("tilt_rmse", "倾角 RMSE"), ("tilt_max", "倾角最大误差"), ("rel_rmse", "遥操作相对姿态 RMSE"),
            ("rel_max", "遥操作相对姿态最大误差"), ("rel_final", "60 s 末航向漂移"), ("cost_us", "单次更新耗时 (µs, Python)")]
    for key, label in rows:
        cells = [f"{stats[n][key][0]:.2f} ± {stats[n][key][1]:.2f}" for n in ALGORITHMS]
        print(f"| {label} | {' | '.join(cells)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="JSL / EKF / Madgwick 姿态解算对比")
    parser.add_argument("--runs", type=int, default=20, help="蒙特卡洛仿真次数")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fig", type=Path, default=CURRENT_DIR / "docs" / "ahrs_comparison.png")
    parser.add_argument("--record", type=float, metavar="SECONDS", help="用真实手柄录制数据")
    parser.add_argument("--out", type=Path, default=CURRENT_DIR / "imu_record.csv", help="录制输出文件")
    parser.add_argument("--csv", type=Path, help="回放录制的真实数据")
    args = parser.parse_args()

    if args.record:
        record(args.record, args.out)
        return
    if args.csv:
        replay(args.csv, args.fig.with_name("ahrs_replay.png"))
        return

    data = simulate(args.seed)
    ev = evaluate(data, run_filters(data))
    mc = monte_carlo(args.runs, args.seed) if args.runs > 1 else None
    print_table(mc if mc else {n: {k: (v, 0.0) for k, v in ev["metrics"][n].items()} for n in ALGORITHMS})
    plot_comparison(data, ev, mc, args.fig)
    print(f"\n对比图已保存到 {args.fig}")


if __name__ == "__main__":
    main()
