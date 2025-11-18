# d455.py  —— 完整可运行最终版本（对外 API 100% 等同原 T265）
import os
import time
from typing import Tuple, Optional, Callable
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from threading import Lock
import pyrealsense2 as rs


class CameraBase:
    """最小 CameraBase 保留接口一致即可"""
    def __init__(self, name='camera'):
        self.name = name


# D455 默认参数
D455_COLOR_W = 640
D455_COLOR_H = 480
D455_FPS = 30


class D455(CameraBase):
    """
    这是 D455 版，完全兼容原 T265 API：
      - start_streaming(callback=None)
      - stop_streaming()
      - get()
      - collect_streaming()
      - save_streaming()
      - raw2pose()
    """
    def __init__(self, id: Optional[str] = None, image: bool = True, name='D455') -> None:
        super().__init__(name=name)
        self._image = image

        # RealSense 管线
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)

        # Color + Depth + IMU
        self.config.enable_stream(rs.stream.color, D455_COLOR_W, D455_COLOR_H,
                                  rs.format.bgr8, D455_FPS)
        self.config.enable_stream(rs.stream.depth, D455_COLOR_W, D455_COLOR_H,
                                  rs.format.z16, D455_FPS)

        try:
            self.config.enable_stream(rs.stream.gyro)
            self.config.enable_stream(rs.stream.accel)
        except Exception:
            pass

        # streaming 数据缓冲
        self.image_streaming_mutex = Lock()
        self.pose_streaming_mutex = Lock()

        self.image_streaming_data = {"left": [], "right": [], "timestamp_ms": []}
        self.pose_streaming_data = {"xyz": [], "quat": [], "timestamp_ms": []}

        self._collect_streaming_data = True
        self.in_streaming = False

        # VIO 状态
        self._prev_color = None
        self._prev_depth = None
        self._K = None
        self._pose_world = np.eye(4)
        self._vio_lock = Lock()

        # 可外部提供一个 vio_update(color, depth, ts)
        self.vio_update: Optional[Callable] = None

    # -----------------------------------------------------------
    # 启动
    # -----------------------------------------------------------
    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        if callback:
            self.profile = self.pipeline.start(self.config, callback)
        else:
            self.profile = self.pipeline.start(self.config, self._internal_callback)

            # 获取相机内参
            intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._K = (intr.fx, intr.fy, intr.ppx, intr.ppy)

        self.in_streaming = True

    # -----------------------------------------------------------
    def stop_streaming(self) -> dict:
        try:
            self.pipeline.stop()
        except Exception:
            pass

        data = {
            "image": self.image_streaming_data,
            "pose": self.pose_streaming_data
        }

        # 清空缓存
        self.image_streaming_data = {"left": [], "right": [], "timestamp_ms": []}
        self.pose_streaming_data = {"xyz": [], "quat": [], "timestamp_ms": []}

        self.in_streaming = False
        return data

    # -----------------------------------------------------------
    def collect_streaming(self, collect: bool = True):
        self._collect_streaming_data = collect

    # -----------------------------------------------------------
    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not self.in_streaming:
            raise NotImplementedError("Not streaming")

        with self.image_streaming_mutex:
            left = self.image_streaming_data["left"][-1] if len(self.image_streaming_data["left"]) else None
            right = self.image_streaming_data["right"][-1] if len(self.image_streaming_data["right"]) else None

        with self.pose_streaming_mutex:
            xyz = self.pose_streaming_data["xyz"][-1] if len(self.pose_streaming_data["xyz"]) else np.zeros(3)
            quat = self.pose_streaming_data["quat"][-1] if len(self.pose_streaming_data["quat"]) else np.array([0,0,0,1])

        return left, right, xyz, quat

    # -----------------------------------------------------------
    def save_streaming(self, save_path: str, data: dict):
        os.makedirs(save_path, exist_ok=True)

        # 保存图像
        if self._image:
            img = data["image"]
            os.makedirs(os.path.join(save_path, "image", "left"), exist_ok=True)
            os.makedirs(os.path.join(save_path, "image", "right"), exist_ok=True)

            np.save(os.path.join(save_path, "image", "timestamps.npy"),
                    np.array(img["timestamp_ms"], dtype=float))

            for i, im in enumerate(img["left"]):
                cv2.imwrite(os.path.join(save_path, "image", "left", f"{i:06d}.png"), im)

            for i, im in enumerate(img["right"]):
                cv2.imwrite(os.path.join(save_path, "image", "right", f"{i:06d}.png"), im)

        # 保存位姿
        pose = data["pose"]
        os.makedirs(os.path.join(save_path, "pose"), exist_ok=True)

        np.save(os.path.join(save_path, "pose", "timestamps.npy"),
                np.array(pose["timestamp_ms"], dtype=float))
        np.save(os.path.join(save_path, "pose", "xyz.npy"),
                np.array(pose["xyz"], dtype=float))
        np.save(os.path.join(save_path, "pose", "quat.npy"),
                np.array(pose["quat"], dtype=float))

    # -----------------------------------------------------------
    def _internal_callback(self, frame):
        ts = time.time() * 1000.0
        if not self._collect_streaming_data:
            return

        if frame.is_frameset() and self._image:
            frameset = frame.as_frameset()
            color_f = frameset.get_color_frame()
            depth_f = frameset.get_depth_frame()

            if not color_f or not depth_f:
                return

            color = np.asanyarray(color_f.get_data())
            depth = np.asanyarray(depth_f.get_data())

            with self.image_streaming_mutex:
                self.image_streaming_data["left"].append(color)
                self.image_streaming_data["right"].append(depth)
                self.image_streaming_data["timestamp_ms"].append(ts)

            # 处理 VO
            with self._vio_lock:
                if self.vio_update:
                    xyz, quat = self.vio_update(color, depth, ts)
                    self._pose_world = self.raw2pose(xyz, quat)
                else:
                    self._builtin_vo(color, depth)

                pose = self._pose_world
                xyz = pose[:3, 3]
                quat = Rot.from_matrix(pose[:3, :3]).as_quat()

            with self.pose_streaming_mutex:
                self.pose_streaming_data["xyz"].append(xyz)
                self.pose_streaming_data["quat"].append(quat)
                self.pose_streaming_data["timestamp_ms"].append(ts)

    # -----------------------------------------------------------
    # 简单 ORB+Depth VO
    # -----------------------------------------------------------
    def _builtin_vo(self, color, depth):
        if self._prev_color is None:
            self._prev_color = color
            self._prev_depth = depth
            return

        if self._K is None:
            return

        fx, fy, cx, cy = self._K

        orb = cv2.ORB_create(2000)
        kp1, d1 = orb.detectAndCompute(self._prev_color, None)
        kp2, d2 = orb.detectAndCompute(color, None)
        if d1 is None or d2 is None:
            self._prev_color = color
            self._prev_depth = depth
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        pts1 = []
        pts2 = []
        for m in matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            u1, v1 = int(u1), int(v1)
            u2, v2 = int(u2), int(v2)
            if u1 < 0 or v1 < 0 or u2 < 0 or v2 < 0:
                continue
            if u1 >= depth.shape[1] or u2 >= depth.shape[1]:
                continue
            if v1 >= depth.shape[0] or v2 >= depth.shape[0]:
                continue
            z1 = float(self._prev_depth[v1, u1]) * 0.001
            z2 = float(depth[v2, u2]) * 0.001
            if z1 < 0.2 or z2 < 0.2:
                continue
            X1 = np.array([(u1 - cx) * z1 / fx, (v1 - cy) * z1 / fy, z1])
            X2 = np.array([(u2 - cx) * z2 / fx, (v2 - cy) * z2 / fy, z2])
            pts1.append(X1)
            pts2.append(X2)

        if len(pts1) < 6:
            self._prev_color = color
            self._prev_depth = depth
            return

        P = np.stack(pts1, axis=0).T
        Q = np.stack(pts2, axis=0).T
        R, t = self._umeyama(P, Q)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        self._pose_world = self._pose_world @ np.linalg.inv(T)

        self._prev_color = color
        self._prev_depth = depth

    @staticmethod
    def _umeyama(P, Q):
        n = P.shape[1]
        muP = np.mean(P, axis=1, keepdims=True)
        muQ = np.mean(Q, axis=1, keepdims=True)
        X = P - muP
        Y = Q - muQ
        S = X @ Y.T / n
        U, D, Vt = np.linalg.svd(S)
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        t = (muQ - R @ muP).reshape(3)
        return R, t

    @staticmethod
    def raw2pose(xyz, quat):
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(quat).as_matrix()
        T[:3, 3] = xyz
        return T


# -----------------------------------------------------------
# Demo
# -----------------------------------------------------------
if __name__ == "__main__":
    cam = D455()
    cam.start_streaming()
    print("Running D455... Press Ctrl-C to exit.")

    try:
        while True:
            time.sleep(0.05)
            color, depth, xyz, quat = cam.get()
            if color is not None:
                cv2.imshow("color", color)
            if depth is not None:
                dvis = depth.astype(np.float32)
                dvis = (dvis / (np.max(dvis) + 1e-6) * 255).astype(np.uint8)
                cv2.imshow("depth", dvis)

            print("xyz:", xyz, "quat:", quat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    data = cam.stop_streaming()
    print("Stopped, collected frames:", len(data["image"]["timestamp_ms"]))
