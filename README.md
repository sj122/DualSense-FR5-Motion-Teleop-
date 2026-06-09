# DualSense Robot Teleop

基于 PS5 DualSense 手柄姿态数据的机械臂遥操作项目。

这个仓库当前的主程序是 `servo_control.py`：它读取手柄四元数、按键和摇杆输入，将其映射为机械臂末端位姿控制指令，并通过 Fairino 机械臂接口实时下发。

除了机械臂控制主程序，仓库里还保留了两个便于调试和演示的脚本：

| 脚本 | 作用 |
|---|---|
| `servo_control.py` | 主程序。使用 DualSense 控制机械臂姿态与平移 |
| `joy.py` | 直接读取手柄姿态，可在终端实时显示原始欧拉角 |
| `cube_3d_from_joy.py` | 使用 `vpython` 显示 3D 正方体，观察手柄姿态变化 |

## 功能概览

| 功能 | 说明 |
|---|---|
| 手柄姿态读取 | 通过 `JoyShockLibrary.dll` 读取 DualSense 四元数 |
| 原始角度显示 | `joy.py` 可直接输出 Roll / Pitch / Yaw |
| 3D 姿态可视化 | `cube_3d_from_joy.py` 用 VPython 显示旋转正方体 |
| 机械臂遥操作 | `servo_control.py` 将手柄姿态映射为机械臂末端姿态 |
| 平移控制 | 左摇杆控制 XY，右摇杆控制 Z |
| 平滑控制 | 使用位置低通滤波和旋转 Slerp 滤波降低抖动 |
| 离合控制 | 按住 `L2` 才接管控制，松开后停止下发 |
| 参考系重标定 | 按 `L1` 将当前姿态设为新的参考基准 |

## 运行环境

| 项目 | 说明 |
|---|---|
| 操作系统 | Windows |
| 手柄 | PS5 DualSense |
| 依赖库 | `numpy`、`scipy`、`matplotlib` 等 |
| 本地动态库 | `JoyShockLibrary.dll` |
| 机械臂接口 | 仓库内 `fairino` 目录 |

## 安装依赖

先安装基础依赖：

```bash
pip install -r requirements.txt
```

如果你需要显示 3D 正方体，再额外安装：

```bash
pip install vpython
```

## 主程序：机械臂遥操作

运行主程序：

```bash
python servo_control.py
```

运行前请先检查 `servo_control.py` 中的机械臂 IP 配置，例如：

```python
IP = '192.168.58.2'
```

程序启动后会：

1. 连接机械臂并回到初始位姿
2. 连接 DualSense 手柄
3. 读取手柄姿态、L1/L2 和摇杆输入
4. 将手柄相对姿态映射到机械臂末端姿态
5. 以固定频率向机械臂发送控制命令

### 按键与控制方式

| 输入 | 功能 |
|---|---|
| `L1` | 重新校准参考姿态 |
| `L2` | 按住时开始控制机械臂，松开停止发送 |
| 左摇杆 | 控制末端平移 XY |
| 右摇杆上下 | 控制末端平移 Z |

## 调试脚本 1：查看原始角度

如果你只是想确认手柄姿态数据是否正常，可以直接运行：

```bash
python joy.py
```

这个脚本会在终端持续输出手柄当前的原始欧拉角：

- `Roll`
- `Pitch`
- `Yaw`

适合用来做以下检查：

- 手柄是否成功连接
- 四元数转欧拉角是否正常
- 姿态方向和轴定义是否符合预期

## 调试脚本 2：显示 3D 正方体

如果你想更直观看手柄姿态变化，可以运行：

```bash
python cube_3d_from_joy.py
```

这个脚本会打开一个 VPython 窗口，显示一个会跟随手柄旋转的正方体，同时在界面中显示：

| 显示内容 | 说明 |
|---|---|
| Display Roll / Pitch / Yaw | 相对姿态显示角度 |
| Raw Roll / Pitch / Yaw | 手柄原始欧拉角 |
| L1 / L2 状态 | 当前按键与离合状态 |

这非常适合在接机械臂之前先验证：

- 姿态读取是否稳定
- 参考系是否符合你的预期
- `L1` / `L2` 的逻辑是否正确

## 项目结构

| 路径 | 说明 |
|---|---|
| `servo_control.py` | 机械臂遥操作主程序 |
| `joy.py` | JoyShockLibrary 轻量封装与原始角度读取 |
| `cube_3d_from_joy.py` | 基于 `joy.py` 的 VPython 可视化脚本 |
| `JoyShockLibrary.dll` | 手柄底层动态库 |
| `fairino/` | 机械臂 Python 接口 |
| `requirements.txt` | Python 依赖列表 |

## 注意事项

| 项目 | 说明 |
|---|---|
| 校准 | 启动手柄后尽量保持平稳几秒，等待底层完成校准 |
| 网络 | 机械臂控制前请确认本机与机械臂 IP 可连通 |
| 安全 | 首次运行 `servo_control.py` 时建议低速、空载、留足安全空间 |
| 依赖 | 若 `cube_3d_from_joy.py` 无法运行，通常是未安装 `vpython` |
| 中断 | 按 `Ctrl+C` 可安全退出脚本 |

## 说明

当前仓库以实验和调试为主，建议上传 GitHub 时优先把 `servo_control.py` 作为项目主入口来介绍，而 `joy.py` 和 `cube_3d_from_joy.py` 作为辅助调试与可视化工具说明。
