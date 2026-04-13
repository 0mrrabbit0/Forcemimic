#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import configargparse
import matplotlib.pyplot as plt
import cv2
import serial
import re
from collections import deque

from r3kit.devices.camera.realsense.d455 import D455
from r3kit.devices.camera.realsense.d415 import D415
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft


# ================= 参数解析 =================
def config_parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument("--save_path", type=str, default="./tmp_data")

    parser.add_argument("--d455_id", type=str)
    parser.add_argument("--d415_id", type=str)

    parser.add_argument("--bdft_ip", type=str)
    parser.add_argument("--bdft_port", type=int)

    parser.add_argument("--enc_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--enc_baud", type=int, default=115200)

    return parser.parse_args()


# ================= 编码器模块 =================
class EncoderReader:
    ANGLE_PATTERN = re.compile(r"angle:\s*([0-9\.]+)")

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.t0 = time.time()
        self.t_buf = deque(maxlen=20)
        self.a_buf = deque(maxlen=200)

    def update(self):
        try:
            self.ser.write(b"spi\r\n")
            time.sleep(0.01)
            raw = self.ser.read_all().decode(errors="ignore")
            for line in raw.splitlines():
                m = self.ANGLE_PATTERN.search(line)
                if m:
                    angle = float(m.group(1))
                    t = time.time() - self.t0
                    self.t_buf.append(t)
                    self.a_buf.append(angle)
        except Exception:
            pass

    def close(self):
        self.ser.close()


# ================= 可视化 =================
def visualize_stream(d455, d415=None, bdft=None, encoder=None):
    plt.ion()
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle("D455 + D415 + FT + Encoder", fontsize=16)

    ax_d455_l, ax_d455_r = axes[0]
    ax_d415_l, ax_d415_r = axes[1]
    ax_ft, ax_enc = axes[2]
    ax_xy, ax_xz = axes[3]

    ft_buf = {k: [] for k in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]}
    t_ft = []

    if bdft and not bdft.connected:
        bdft.connect(write_init_register=None)
        bdft.start_streaming()

    t0 = time.time()

    try:
        while True:
            # ---------- D455 ----------
            try:
                left, right, xyz, quat = d455.get()
            except Exception:
                left = right = xyz = quat = None

            ax_d455_l.clear()
            if left is not None:
                ax_d455_l.imshow(left, cmap="gray")
                ax_d455_l.set_title("D455 Left")
            ax_d455_l.axis("off")

            ax_d455_r.clear()
            if right is not None:
                ax_d455_r.imshow(right, cmap="gray")
                ax_d455_r.set_title("D455 Right")
            ax_d455_r.axis("off")

            # ---------- D415 ----------
            if d415:
                try:
                    l415, r415, _, _ = d415.get()
                except Exception:
                    l415 = r415 = None

                ax_d415_l.clear()
                if l415 is not None:
                    ax_d415_l.imshow(l415, cmap="gray")
                    ax_d415_l.set_title("D415 Left")
                ax_d415_l.axis("off")

                ax_d415_r.clear()
                if r415 is not None:
                    ax_d415_r.imshow(r415, cmap="gray")
                    ax_d415_r.set_title("D415 Right")
                ax_d415_r.axis("off")

            # ---------- 位姿 ----------
            ax_xy.clear()
            ax_xz.clear()
            if xyz is not None and quat is not None:
                try:
                    R = D455.raw2pose(xyz, quat)
                    ax_xy.quiver(0, 0, R[0, 0], R[1, 0], color="r", scale=1)
                    ax_xy.quiver(0, 0, R[0, 1], R[1, 1], color="g", scale=1)
                    ax_xy.quiver(0, 0, R[0, 2], R[1, 2], color="b", scale=1)
                    ax_xy.set_title("Pose XY")
                    ax_xy.set_aspect("equal")
                    ax_xy.grid(True)

                    ax_xz.quiver(0, 0, R[0, 0], R[2, 0], color="r", scale=1)
                    ax_xz.quiver(0, 0, R[0, 1], R[2, 1], color="g", scale=1)
                    ax_xz.quiver(0, 0, R[0, 2], R[2, 2], color="b", scale=1)
                    ax_xz.set_title("Pose XZ")
                    ax_xz.set_aspect("equal")
                    ax_xz.grid(True)
                except Exception:
                    pass

            # ---------- 力传感器 ----------
            if bdft:
                try:
                    data = bdft._read(n=1)
                    if data and "ft" in data:
                        Fx, Fy, Fz, Mx, My, Mz = data["ft"][0].tolist()
                        t_ft.append(time.time() - t0)
                        for k, v in zip(ft_buf, [Fx, Fy, Fz, Mx, My, Mz]):
                            ft_buf[k].append(v)
                except Exception:
                    pass

            ax_ft.clear()
            for k in ft_buf:
                ax_ft.plot(t_ft, ft_buf[k], label=k)
            ax_ft.set_title("Force / Torque")
            ax_ft.legend(ncol=3)
            ax_ft.grid(True)

            # ---------- 编码器 ----------
            if encoder:
                encoder.update()
                ax_enc.clear()
                ax_enc.plot(encoder.t_buf, encoder.a_buf, color="purple")
                ax_enc.set_title("Encoder Angle (deg)")
                ax_enc.set_xlabel("Time (s)")
                ax_enc.set_ylabel("Angle")
                ax_enc.grid(True)

            plt.pause(0.03)

    except KeyboardInterrupt:
        print("[EXIT]")

    finally:
        if bdft:
            bdft.stop_streaming()
            bdft.disconnect()
        if encoder:
            encoder.close()
        plt.close()


# ================= main =================
if __name__ == "__main__":
    args = config_parse()

    d455 = D455(id=args.d455_id, image=True)
    d455.start_streaming()

    d415 = None
    if args.d415_id:
        try:
            d415 = D415(id=args.d415_id)
            d415.start_streaming()
        except Exception as e:
            print("[WARN] D415 failed:", e)

    bdft = None
    if args.bdft_ip and args.bdft_port:
        bdft = Bdft(host=args.bdft_ip, port=args.bdft_port, fps=100)

    encoder = EncoderReader(args.enc_port, args.enc_baud)

    try:
        visualize_stream(d455, d415, bdft, encoder)
    finally:
        d455.stop_streaming()
        if d415:
            d415.stop_streaming()
        cv2.destroyAllWindows()
