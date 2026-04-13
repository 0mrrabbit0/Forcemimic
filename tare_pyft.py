import configargparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

from r3kit.devices.camera.realsense.d455 import D455
from r3kit.devices.camera.realsense.d415 import D415
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft
from r3kit.devices.encoder.pdcd.angler import Angler


# ----------------- 参数解析 -----------------
def config_parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--save_path", type=str, default="./tmp_data")

    # Camera
    parser.add_argument("--d455_id", type=str)
    parser.add_argument("--d415_id", type=str)

    # FT sensor
    parser.add_argument("--bdft_ip", type=str)
    parser.add_argument("--bdft_port", type=int)

    # Encoder
    parser.add_argument("--encoder_port", type=str, default="/dev/ttyUSB0")

    return parser.parse_args()


# ----------------- 可视化 -----------------
def visualize_stream_with_third_view(
    d455: D455,
    d415: D415,
    encoder: Angler = None,
    bdft: Bdft = None,
    num_frames=200,
    pause_time=0.03,
):
    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("D455 + D415 + Encoder + 6D FT", fontsize=16)

    ax_left, ax_right, ax_third = axes[0]
    ax_xy, ax_xz, ax_enc = axes[1]
    ax_ft = axes[2, :]

    # --------- 数据缓存 ---------
    t_buf = []
    encoder_buf = []
    ft_buf = {k: [] for k in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]}

    i = 0
    stop = False

    while not stop:
        try:
            # ================= D455 =================
            try:
                left, right, xyz, quat = d455.get()
            except Exception:
                left = right = xyz = quat = None

            # ================= D415 (第三视角) =================
            try:
                third_img = d415.get()
            except Exception:
                left = right = xyz = quat = None

            # ---------------- 左右相机 ----------------
            ax_left.clear()
            if left is not None:
                ax_left.imshow(left, cmap="gray")
                ax_left.set_title("D455 Left")
            else:
                ax_left.text(0.5, 0.5, "LEFT ERROR", ha="center")
            ax_left.axis("off")

            ax_right.clear()
            if right is not None:
                ax_right.imshow(right, cmap="gray")
                ax_right.set_title("D455 Right")
            else:
                ax_right.text(0.5, 0.5, "RIGHT ERROR", ha="center")
            ax_right.axis("off")

            # ---------------- 第三视角 ----------------
            ax_third.clear()
            if third_img is not None:
                ax_third.imshow(third_img, cmap="gray")
                ax_third.set_title("D415 Third View")
            else:
                ax_third.text(0.5, 0.5, "D415 ERROR", ha="center")
            ax_third.axis("off")

            # ---------------- 位姿 ----------------
            pose = None
            if xyz is not None and quat is not None:
                try:
                    pose = D455.raw2pose(xyz, quat)
                except Exception:
                    pose = None

            ax_xy.clear()
            ax_xz.clear()
            if pose is not None:
                ax_xy.quiver(0, 0, pose[0, 0], pose[1, 0], color="r")
                ax_xy.quiver(0, 0, pose[0, 1], pose[1, 1], color="g")
                ax_xy.quiver(0, 0, pose[0, 2], pose[1, 2], color="b")

                ax_xz.quiver(0, 0, pose[0, 0], pose[2, 0], color="r")
                ax_xz.quiver(0, 0, pose[0, 1], pose[2, 1], color="g")
                ax_xz.quiver(0, 0, pose[0, 2], pose[2, 2], color="b")
            else:
                ax_xy.text(0.5, 0.5, "POSE ERROR", ha="center")
                ax_xz.text(0.5, 0.5, "POSE ERROR", ha="center")

            ax_xy.set_title("Pose XY")
            ax_xz.set_title("Pose XZ")
            ax_xy.set_aspect("equal")
            ax_xz.set_aspect("equal")

            # ---------------- 编码器 ----------------
            enc_val = np.nan
            if encoder is not None:
                try:
                    enc_data = encoder.get()
                    if enc_data:
                        enc_val = enc_data["angle"]
                except Exception:
                    pass

            encoder_buf.append(enc_val)

            ax_enc.clear()
            ax_enc.plot(encoder_buf, label="Encoder Angle")
            ax_enc.set_title("Encoder")
            ax_enc.legend()
            ax_enc.grid(True)

            # ---------------- 力传感器 ----------------
            Fx = Fy = Fz = Mx = My = Mz = np.nan
            if bdft is not None:
                try:
                    data = bdft.get()
                    ft = data["ft"][-1]
                    Fx, Fy, Fz, Mx, My, Mz = ft.tolist()
                except Exception:
                    pass

            for k, v in zip(
                ["Fx", "Fy", "Fz", "Mx", "My", "Mz"], [Fx, Fy, Fz, Mx, My, Mz]
            ):
                ft_buf[k].append(v)

            t_buf.append(i)

            if len(t_buf) > num_frames:
                t_buf.pop(0)
                encoder_buf.pop(0)
                for k in ft_buf:
                    ft_buf[k].pop(0)

            # ---------------- FT 绘制 ----------------
            ax_ft[0].clear()
            for k in ft_buf:
                ax_ft[0].plot(t_buf, ft_buf[k], label=k)
            ax_ft[0].set_title("6D Force / Torque")
            ax_ft[0].legend(ncol=3)
            ax_ft[0].grid(True)

            plt.pause(pause_time)

            if cv2.waitKey(1) & 0xFF == 27:
                stop = True

            i += 1

        except Exception as e:
            print("[WARN]", e)
            continue

    plt.ioff()
    plt.close()


# ----------------- main -----------------
if __name__ == "__main__":
    args = config_parse()

    # ---------- Cameras ----------
    d455 = D455(id=args.d455_id, image=True, name="D455")
    d455.start_streaming()

    d415 = D415(id=args.d415_id, name="D415")
    d415.start_streaming()

    # ---------- Encoder ----------
    encoder = Angler(
        id=args.encoder_port, index=1, baudrate=115200, fps=30, gap=0.002, name="Angler"
    )

    # ---------- Force Sensor ----------
    bdft = Bdft(host=args.bdft_ip, port=args.bdft_port, fps=100)

    try:
        visualize_stream_with_third_view(
            d455=d455, d415=d415, encoder=encoder, bdft=bdft
        )
    finally:
        d455.stop_streaming()
        d415.stop_streaming()
        bdft.disconnect()
        cv2.destroyAllWindows()
