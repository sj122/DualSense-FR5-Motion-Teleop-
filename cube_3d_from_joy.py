"""用 ds/joy.py 的数据源驱动 VPython 正方体显示。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from vpython import box, color, cylinder, norm, rate, scene, vector


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from joy import JoyShockDevice, add_ahrs_argument  # noqa: E402


def rotation_from_quaternion_wxyz(quaternion_wxyz) -> R:
    w, x, y, z = quaternion_wxyz
    return R.from_quat([x, y, z, w])


def quaternion_wxyz_from_euler_deg(roll: float, pitch: float, yaw: float):
    rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    x, y, z, w = rotation.as_quat()
    return (w, x, y, z)


def rotation_to_euler_deg(rotation: R):
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
    return float(roll), float(pitch), float(yaw)


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """四元数转旋转矩阵。"""

    return rotation_from_quaternion_wxyz((w, x, y, z)).as_matrix()


class RelativeCubeOrientationController:
    """把绝对手柄姿态转换成带离合的相对姿态显示。"""

    def __init__(self, clutch_threshold: float = 0.2):
        self.clutch_threshold = clutch_threshold
        self.reference_rotation: R | None = None
        self.latched_display_rotation = R.identity()
        self.current_display_rotation = R.identity()
        self.clutch_anchor_rotation: R | None = None
        self.clutch_anchor_display_rotation = R.identity()
        self._clutch_active = False
        self._last_l1_pressed = False

    def update(self, quaternion_wxyz, l2_value: float, l1_pressed: bool) -> R:
        current_absolute_rotation = rotation_from_quaternion_wxyz(quaternion_wxyz)
        if self.reference_rotation is None:
            self.reference_rotation = current_absolute_rotation

        normalized_rotation = self.reference_rotation.inv() * current_absolute_rotation
        clutch_active = l2_value > self.clutch_threshold
        recalibrate_pressed = l1_pressed and not self._last_l1_pressed

        if recalibrate_pressed:
            self.reference_rotation = current_absolute_rotation
            normalized_rotation = R.identity()
            if self._clutch_active:
                self.clutch_anchor_rotation = normalized_rotation
                self.clutch_anchor_display_rotation = self.current_display_rotation

        if clutch_active and not self._clutch_active:
            self.clutch_anchor_rotation = normalized_rotation
            self.clutch_anchor_display_rotation = self.latched_display_rotation
            self.current_display_rotation = self.latched_display_rotation
        elif clutch_active and self.clutch_anchor_rotation is not None:
            delta_rotation = self.clutch_anchor_rotation.inv() * normalized_rotation
            self.current_display_rotation = self.clutch_anchor_display_rotation * delta_rotation
        elif not clutch_active and self._clutch_active:
            self.latched_display_rotation = self.current_display_rotation
        elif not clutch_active:
            self.current_display_rotation = self.latched_display_rotation

        self._clutch_active = clutch_active
        self._last_l1_pressed = l1_pressed
        return self.current_display_rotation


def apply_rotation_to_box(box_obj, rotation_matrix: np.ndarray) -> None:
    axis = vector(*rotation_matrix[:, 0])
    up = vector(*rotation_matrix[:, 2])
    box_obj.axis = norm(axis)
    box_obj.up = norm(up)


def configure_scene() -> None:
    scene.width = 900
    scene.height = 700
    scene.background = vector(0.2, 0.2, 0.2)
    scene.forward = vector(0, 0, -1)
    scene.range = 2
    scene.title = "JoyShockLibrary Quaternion Cube"


def main() -> None:
    parser = argparse.ArgumentParser(description="用 VPython 正方体显示手柄姿态")
    add_ahrs_argument(parser)
    args = parser.parse_args()

    configure_scene()

    cube = box(
        pos=vector(0, 0, 0),
        length=1,
        height=1,
        width=1,
        color=color.blue,
        opacity=0.9,
    )

    axis_scale = 1.2
    axis_x = cylinder(
        pos=vector(0, 0, 0),
        axis=vector(axis_scale, 0, 0),
        radius=0.05,
        color=color.red,
    )
    axis_y = cylinder(
        pos=vector(0, 0, 0),
        axis=vector(0, axis_scale, 0),
        radius=0.05,
        color=color.green,
    )
    axis_z = cylinder(
        pos=vector(0, 0, 0),
        axis=vector(0, 0, axis_scale),
        radius=0.05,
        color=color.blue,
    )

    print("正在连接手柄，请将其平放几秒钟完成底层校准...")
    print("按住 L2 才会把相对姿态增量映射到正方体。")
    print("按下 L1 会把当前手柄姿态设为新基准，但正方体不会跳变。")
    print(f"姿态解算算法: {args.ahrs}")
    print("按 Ctrl+C 退出。\n")

    with JoyShockDevice(ahrs=args.ahrs) as device:
        device.connect_first_device(warmup_seconds=3.0)
        last_caption_update = time.time()
        relative_controller = RelativeCubeOrientationController(clutch_threshold=0.2)

        try:
            while True:
                rate(10)

                frame = device.read_input_frame(l2_threshold=0.2)
                effective_l2_value = max(
                    frame["analogs"]["l2_value"],
                    1.0 if frame["buttons"]["l2_pressed"] else 0.0,
                )
                display_rotation = relative_controller.update(
                    quaternion_wxyz=frame["quaternion_wxyz"],
                    l2_value=effective_l2_value,
                    l1_pressed=frame["buttons"]["l1"],
                )
                rotation_matrix = display_rotation.as_matrix()
                display_roll, display_pitch, display_yaw = rotation_to_euler_deg(display_rotation)
                raw_roll, raw_pitch, raw_yaw = frame["euler_deg"]

                apply_rotation_to_box(cube, rotation_matrix)
                axis_x.axis = vector(*rotation_matrix[:, 0]) * axis_scale
                axis_y.axis = vector(*rotation_matrix[:, 1]) * axis_scale
                axis_z.axis = vector(*rotation_matrix[:, 2]) * axis_scale

                if time.time() - last_caption_update > 0.2:
                    scene.caption = (
                        f"Display Roll: {display_roll:6.1f} deg | "
                        f"Display Pitch: {display_pitch:6.1f} deg | "
                        f"Display Yaw: {display_yaw:6.1f} deg\n"
                        f"Raw Roll: {raw_roll:6.1f} deg | "
                        f"Raw Pitch: {raw_pitch:6.1f} deg | "
                        f"Raw Yaw: {raw_yaw:6.1f} deg\n"
                        f"L2: {frame['analogs']['l2_value']:.2f} | "
                        f"Clutch: {frame['buttons']['l2_pressed']} | "
                        f"L1: {frame['buttons']['l1']}"
                    )
                    last_caption_update = time.time()

        except KeyboardInterrupt:
            print("\n已停止显示。")


if __name__ == "__main__":
    main()
