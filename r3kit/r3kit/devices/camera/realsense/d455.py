import os
from typing import Tuple, Optional
import time
import numpy as np
import cv2
from threading import Lock
import pyrealsense2 as rs

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.realsense.config import D455_STREAMS
from r3kit.utils.vis import save_imgs


class D455(CameraBase):
    def __init__(
        self,
        id: Optional[str] = None,
        image: bool = True,
        depth: bool = True,
        name: str = "D455",
    ) -> None:
        super().__init__(name=name)
        self._image = image
        self._depth = depth

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if id is not None:
            self.config.enable_device(id)

        # 启用双目红外流
        for stream_item in D455_STREAMS:
            if not image and stream_item[0] == rs.stream.infrared:
                continue
            self.config.enable_stream(*stream_item)

        # 启用深度流
        if self._depth:
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        # 对齐到左红外
        self.align = rs.align(rs.stream.infrared)

        # --- 新增：初始化时获取内参和 Depth Scale (参考 L515 逻辑) ---
        self.pipeline_profile = self.pipeline.start(self.config)
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # 获取深度缩放比例

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        ir_frame = aligned_frames.get_infrared_frame(1)
        ir_intrinsics = (
            ir_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )
        # [ppx, ppy, fx, fy] 格式，供点云还原使用
        self.intrinsics = [
            ir_intrinsics.ppx,
            ir_intrinsics.ppy,
            ir_intrinsics.fx,
            ir_intrinsics.fy,
        ]

        self.pipeline.stop()
        self.in_streaming = False

    def get(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回最新帧（不消费，只读取）
        """
        if not self.in_streaming:
            raise RuntimeError("Camera not streaming. Call start_streaming() first.")

        self.image_streaming_mutex.acquire()
        self.depth_streaming_mutex.acquire()

        left_image = (
            self.image_streaming_data["left"][-1].copy()
            if self.image_streaming_data["left"]
            else None
        )
        right_image = (
            self.image_streaming_data["right"][-1].copy()
            if self.image_streaming_data["right"]
            else None
        )
        depth_image = (
            self.depth_streaming_data["depth"][-1].copy()
            if self.depth_streaming_data["depth"]
            else None
        )

        self.depth_streaming_mutex.release()
        self.image_streaming_mutex.release()

        return (left_image, right_image, depth_image)

    def start_streaming(self, callback: Optional[callable] = None) -> None:
        if self.in_streaming:
            self.stop_streaming()

        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True

        if callback is not None:
            self.pipeline_profile = self.pipeline.start(self.config, callback)
        else:
            # 初始化缓冲区
            self.image_streaming_mutex = Lock()
            self.image_streaming_data = {"left": [], "right": [], "timestamp_ms": []}

            self.depth_streaming_mutex = Lock()
            self.depth_streaming_data = {"depth": [], "timestamp_ms": []}

            self.pc = rs.pointcloud()
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)

        self.in_streaming = True

    def stop_streaming(self) -> Optional[dict]:
        streaming_data = None
        if self.in_streaming:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass

        # 合并返回字典，确保 save_streaming 能识别
        if hasattr(self, "image_streaming_data") and hasattr(
            self, "depth_streaming_data"
        ):
            streaming_data = {
                "image": self.image_streaming_data,
                "depth": self.depth_streaming_data,
            }
            # 重置以释放内存
            self.image_streaming_data = {"left": [], "right": [], "timestamp_ms": []}
            self.depth_streaming_data = {"depth": [], "timestamp_ms": []}

        self.image_streaming_mutex = None
        self.depth_streaming_mutex = None
        self.in_streaming = False
        return streaming_data

    def save_streaming(self, save_path: str, streaming_data: dict) -> None:
        # 步骤 A: 只要进来，第一件事就是建文件夹并存内参 (不加任何 if)
        os.makedirs(save_path, exist_ok=True)

        # 强制保存，不依赖 streaming_data 的内容
        print(f"[{self.name}] Force saving intrinsics...")
        np.savetxt(
            os.path.join(save_path, "intrinsics.txt"), self.intrinsics, fmt="%.16f"
        )
        np.savetxt(
            os.path.join(save_path, "depth_scale.txt"), [self.depth_scale], fmt="%.16f"
        )

        # 步骤 B: 保存时间戳
        if streaming_data and "depth" in streaming_data:
            ts_ms = np.array(streaming_data["depth"]["timestamp_ms"], dtype=float)
            np.save(os.path.join(save_path, "timestamps.npy"), ts_ms)

        # 2. 保存红外图像
        if "image" in streaming_data and self._image:
            img_data = streaming_data["image"]
            os.makedirs(os.path.join(save_path, "image", "left"), exist_ok=True)
            os.makedirs(os.path.join(save_path, "image", "right"), exist_ok=True)
            save_imgs(os.path.join(save_path, "image", "left"), img_data["left"])
            save_imgs(os.path.join(save_path, "image", "right"), img_data["right"])

        # 3. 保存深度图
        if "depth" in streaming_data and self._depth:
            depth_data = streaming_data["depth"]
            os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
            save_imgs(os.path.join(save_path, "depth"), depth_data["depth"])

    def collect_streaming(self, collect: bool = True) -> None:
        self._collect_streaming_data = collect

    def callback(self, frame):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return

        if frame.is_frameset():
            frameset = frame.as_frameset()
            aligned_frames = self.align.process(frameset)

            # 采集图像流
            if self._image:
                f_left = aligned_frames.get_infrared_frame(1)
                f_right = aligned_frames.get_infrared_frame(2)
                if f_left and f_right:
                    self.image_streaming_mutex.acquire()
                    self.image_streaming_data["left"].append(
                        np.asanyarray(f_left.get_data()).copy()
                    )
                    self.image_streaming_data["right"].append(
                        np.asanyarray(f_right.get_data()).copy()
                    )
                    self.image_streaming_data["timestamp_ms"].append(ts)
                    self.image_streaming_mutex.release()

            # 采集深度流
            if self._depth:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    self.depth_streaming_mutex.acquire()
                    self.depth_streaming_data["depth"].append(
                        np.asanyarray(depth_frame.get_data()).copy()
                    )
                    self.depth_streaming_data["timestamp_ms"].append(ts)
                    self.depth_streaming_mutex.release()

    @staticmethod
    def img2pc(
        depth_img: np.ndarray,
        intrinsics: np.ndarray,
        color_img: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """点云还原静态方法"""
        height, weight = depth_img.shape
        [pixX, pixY] = np.meshgrid(np.arange(weight), np.arange(height))
        x = (pixX - intrinsics[0]) * depth_img / intrinsics[2]
        y = (pixY - intrinsics[1]) * depth_img / intrinsics[3]
        z = depth_img
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        rgb = color_img.reshape(-1, 3) if color_img is not None else None
        return xyz, rgb

    def __del__(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    camera = D455(id=None, image=True, depth=True)
    camera.start_streaming()
    i = 0
    try:
        while True:
            print(f"Frame {i}", end="\r")
            left, right, depth, pc = camera.get()
            if left is not None:
                cv2.imshow("left", left)
                cv2.imshow("right", right)
            if depth is not None:
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
                )
                cv2.imshow("depth", depth_vis)
            if cv2.waitKey(1) == 27:
                break
            i += 1
    finally:
        cv2.destroyAllWindows()
        camera.stop_streaming()
