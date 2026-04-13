import time
import configargparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

from r3kit.devices.camera.realsense.d455 import D455
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--save_path", type=str, default="./tmp_data")
    parser.add_argument("--d455_id", type=str)
    parser.add_argument("--bdft_id", type=str)
    parser.add_argument("--bdft_port", type=int)
    args = parser.parse_args()
    return args


# ----------------- 容错加载 -----------------
def safe_get(idx, arr):
    try:
        return arr[idx]
    except Exception:
        return None


# ----------------- 可视化四窗口 + 力矩曲线 -----------------
def visualize_four_windows_stream_with_ft(
    d455: D455,
    bdft: Bdft = None,
    bdft_fts=None,
    bdft_poses=None,
    num_frames=100,
    pause_time=0.05,
):
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("D455 Cameras + BD-FT Poses + Force/Torque (Real-time)")

    # 左右相机 + XY/XZ 位姿在前四个窗口
    ax_left_img, ax_right_img = axes[0, 0], axes[0, 1]
    ax_xy_pose, ax_xz_pose = axes[1, 0], axes[1, 1]

    # 下方两个窗口显示 Fx,Fy,Fz 和 Mx,My,Mz
    ax_force = axes[2, 0]
    ax_moment = axes[2, 1]

    # 初始化曲线数据
    max_len = num_frames
    force_data = {"Fx": [], "Fy": [], "Fz": []}
    moment_data = {"Mx": [], "My": [], "Mz": []}
    time_data = []

    i = 0
    stop = False
    while not stop:
        try:
            # 实时获取 D455 图像和位姿
            left, right, xyz, quat = d455.get()

            # -------------------- 位姿更新 --------------------
            if bdft_poses is not None and i < len(bdft_poses):
                pose = bdft_poses[i]
            else:
                pose = D455.raw2pose(xyz, quat)

            # -------------------- 相机图像 --------------------
            ax_left_img.clear()
            if left is not None:
                ax_left_img.imshow(left, cmap="gray")
                ax_left_img.set_title("Left Camera")
            else:
                ax_left_img.text(
                    0.5, 0.5, "LEFT IMAGE ERROR", ha="center", va="center", fontsize=12
                )
            ax_left_img.axis("off")

            ax_right_img.clear()
            if right is not None:
                ax_right_img.imshow(right, cmap="gray")
                ax_right_img.set_title("Right Camera")
            else:
                ax_right_img.text(
                    0.5, 0.5, "RIGHT IMAGE ERROR", ha="center", va="center", fontsize=12
                )
            ax_right_img.axis("off")

            # -------------------- 位姿 --------------------
            ax_xy_pose.clear()
            if pose is not None:
                ax_xy_pose.quiver(
                    0,
                    0,
                    pose[0, 0],
                    pose[1, 0],
                    color="r",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="x",
                )
                ax_xy_pose.quiver(
                    0,
                    0,
                    pose[0, 1],
                    pose[1, 1],
                    color="g",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="y",
                )
                ax_xy_pose.quiver(
                    0,
                    0,
                    pose[0, 2],
                    pose[1, 2],
                    color="b",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="z",
                )
            else:
                ax_xy_pose.text(
                    0.5, 0.5, "LEFT POSE ERROR", ha="center", va="center", fontsize=12
                )
            ax_xy_pose.set_xlim(-0.1, 0.1)
            ax_xy_pose.set_ylim(-0.1, 0.1)
            ax_xy_pose.set_aspect("equal")
            ax_xy_pose.set_title("Pose XY Projection")
            ax_xy_pose.legend()

            ax_xz_pose.clear()
            if pose is not None:
                ax_xz_pose.quiver(
                    0,
                    0,
                    pose[0, 0],
                    pose[2, 0],
                    color="r",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="x",
                )
                ax_xz_pose.quiver(
                    0,
                    0,
                    pose[0, 1],
                    pose[2, 1],
                    color="g",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="y",
                )
                ax_xz_pose.quiver(
                    0,
                    0,
                    pose[0, 2],
                    pose[2, 2],
                    color="b",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    label="z",
                )
            else:
                ax_xz_pose.text(
                    0.5, 0.5, "RIGHT POSE ERROR", ha="center", va="center", fontsize=12
                )
            ax_xz_pose.set_xlim(-0.1, 0.1)
            ax_xz_pose.set_ylim(-0.1, 0.1)
            ax_xz_pose.set_aspect("equal")
            ax_xz_pose.set_title("Pose XZ Projection")
            ax_xz_pose.legend()

            # -------------------- 力矩数据 --------------------
            _t = time.time()
            if bdft is not None:
                try:
                    ft = bdft.get()  # 假设返回 dict {'F':[fx,fy,fz], 'M':[mx,my,mz]}
                    Fx, Fy, Fz = ft["F"]
                    Mx, My, Mz = ft["M"]
                except Exception:
                    Fx, Fy, Fz = 0, 0, 0
                    Mx, My, Mz = 0, 0, 0
            else:
                # 模拟数据
                Fx = np.sin(i / 10) * 5
                Fy = np.cos(i / 10) * 3
                Fz = np.sin(i / 15) * 2
                Mx = np.sin(i / 20) * 0.5
                My = np.cos(i / 25) * 0.3
                Mz = np.sin(i / 30) * 0.2

            # 更新曲线数据
            force_data["Fx"].append(Fx)
            force_data["Fy"].append(Fy)
            force_data["Fz"].append(Fz)
            moment_data["Mx"].append(Mx)
            moment_data["My"].append(My)
            moment_data["Mz"].append(Mz)
            time_data.append(i)

            # 保持长度
            if len(time_data) > max_len:
                for k in force_data:
                    force_data[k].pop(0)
                for k in moment_data:
                    moment_data[k].pop(0)
                time_data.pop(0)

            # 绘制力
            ax_force.clear()
            ax_force.plot(time_data, force_data["Fx"], "r", label="Fx")
            ax_force.plot(time_data, force_data["Fy"], "g", label="Fy")
            ax_force.plot(time_data, force_data["Fz"], "b", label="Fz")
            ax_force.set_title("Force (N)")
            ax_force.set_xlim(time_data[0], time_data[-1])
            ax_force.set_ylim(-10, 10)
            ax_force.legend()

            # 绘制力矩
            ax_moment.clear()
            ax_moment.plot(time_data, moment_data["Mx"], "r", label="Mx")
            ax_moment.plot(time_data, moment_data["My"], "g", label="My")
            ax_moment.plot(time_data, moment_data["Mz"], "b", label="Mz")
            ax_moment.set_title("Moment (Nm)")
            ax_moment.set_xlim(time_data[0], time_data[-1])
            ax_moment.set_ylim(-1, 1)
            ax_moment.legend()

            plt.pause(pause_time)

            # 检测 ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                stop = True

            i += 1
        except Exception as e:
            print(f"[WARN] Visualization error: {e}")
            continue
    plt.ioff()
    plt.close()


# ----------------- 主函数 -----------------
if __name__ == "__main__":
    args = config_parse()
    d455 = D455(id=args.d455_id, image=True, name="D455")
    try:
        d455.start_streaming(callback=None)
        bdft_device = (
            None  # 如果有 BD-FT，可初始化 Bdft(id=args.bdft_id, port=args.bdft_port)
        )
        visualize_four_windows_stream_with_ft(
            d455, bdft=bdft_device, num_frames=100, pause_time=0.05
        )
    except Exception as e:
        print(f"Error: {e}")
    finally:
        d455.stop_streaming()
        cv2.destroyAllWindows()
