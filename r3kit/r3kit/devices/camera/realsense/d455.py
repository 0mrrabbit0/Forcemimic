import os
from typing import Tuple, Optional
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from threading import Lock
import pyrealsense2 as rs

# 假设 r3kit.devices.camera.base 和 r3kit.utils.vis 模块存在
from r3kit.devices.camera.base import CameraBase 
# from r3kit.devices.camera.realsense.config import *
# from r3kit.utils.vis import draw_time, save_imgs 

# -----------------------------------------------------
# 辅助函数 (占位符)
# -----------------------------------------------------
def draw_time(timestamps, save_path): pass
def save_imgs(save_dir, images): 
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(save_dir, f"{i:06d}.png"), img)
    
# -----------------------------------------------------
# D455 相机类 (集成双重积分)
# -----------------------------------------------------

class D455(CameraBase):
    def __init__(self, id:Optional[str]=None, image:bool=True, name:str='D455') -> None:
        super().__init__(name=name)
        self._image = image

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if id is not None:
            self.config.enable_device(id)
        
        # D455 流配置: 双目红外 + IMU
        if self._image:
            self.config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
            self.config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)
        
        # 启用 IMU 流 (Accel & Gyro)
        self.config.enable_stream(rs.stream.accel)
        self.config.enable_stream(rs.stream.gyro)
        
        # ------------------------------------------
        # 核心修改: 状态变量
        # ------------------------------------------
        # 姿态积分状态 (Gyro)
        self._last_ts_gyro = None
        self._current_quat = np.array([0.0, 0.0, 0.0, 1.0]) 
        
        # 位置积分状态 (Accel)
        self._current_xyz = np.array([0.0, 0.0, 0.0])  # P
        self._current_vel = np.array([0.0, 0.0, 0.0])  # V
        self._last_ts_accel = None                    # Accel 时间戳
        # ------------------------------------------

        self.in_streaming = False

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not self.in_streaming:
            raise NotImplementedError
        else:
            if hasattr(self, "image_streaming_data"):
                # 获取锁，访问数据
                self.image_streaming_mutex.acquire()
                self.pose_streaming_mutex.acquire()
                
                # 获取图像
                if self._image and len(self.image_streaming_data["left"]) > 0:
                    left_image = self.image_streaming_data["left"][-1]
                    right_image = self.image_streaming_data["right"][-1]
                else:
                    left_image = None
                    right_image = None
                
                # 获取位姿 (使用双重积分的值)
                xyz = self._current_xyz.copy() 
                quat = self._current_quat.copy()

                # 释放锁
                self.pose_streaming_mutex.release()
                self.image_streaming_mutex.release()
                return (left_image, right_image, xyz, quat)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
            
        if callback is not None:
            self.pipeline_profile = self.pipeline.start(self.config, callback)
        else:
            self.image_streaming_mutex = Lock()
            self.image_streaming_data = {
                "left": [], "right": [], "timestamp_ms": [], 
            }
            self.pose_streaming_mutex = Lock()
            self.pose_streaming_data = {
                "xyz": [], "quat": [], "timestamp_ms": [], 
            }
            # 重置 IMU 状态
            self._last_ts_gyro = None
            self._current_quat = np.array([0.0, 0.0, 0.0, 1.0]) 
            self._current_xyz = np.array([0.0, 0.0, 0.0]) 
            self._current_vel = np.array([0.0, 0.0, 0.0])
            self._last_ts_accel = None
            
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)
        self.in_streaming = True

    def stop_streaming(self) -> Optional[dict]:
        # 保持与 T265 相同的 stop_streaming 逻辑
        streaming_data = None
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
            
        # ... (数据清理逻辑省略，与上一个版本相同) ...
        if hasattr(self, "image_streaming_mutex"): self.image_streaming_mutex = None
        if hasattr(self, "image_streaming_data"):
            streaming_data = {'image': self.image_streaming_data}
            self.image_streaming_data = {"left": [], "right": [], "timestamp_ms": []}
        if hasattr(self, "pose_streaming_mutex"): self.pose_streaming_mutex = None
        if hasattr(self, "pose_streaming_data"):
            if streaming_data is None: streaming_data = {}
            streaming_data['pose'] = self.pose_streaming_data
            self.pose_streaming_data = {"xyz": [], "quat": [], "timestamp_ms": []}
            
        self.in_streaming = False
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        # 保持与 T265 相同的 save_streaming 逻辑
        assert len(streaming_data["image"]["left"]) == len(streaming_data["image"]["right"])
        assert len(streaming_data["pose"]["xyz"]) == len(streaming_data["pose"]["quat"])
        
        if self._image:
            os.makedirs(os.path.join(save_path, 'image'), exist_ok=True)
            np.save(os.path.join(save_path, 'image', "timestamps.npy"), np.array(streaming_data["image"]["timestamp_ms"], dtype=float))
            if len(streaming_data["image"]["timestamp_ms"]) > 1:
                freq = len(streaming_data["image"]["timestamp_ms"]) / (streaming_data["image"]["timestamp_ms"][-1] - streaming_data["image"]["timestamp_ms"][0] + 1e-6) * 1000
                draw_time(streaming_data["image"]["timestamp_ms"], os.path.join(save_path, 'image', f"freq_{freq:.2f}.png"))
            os.makedirs(os.path.join(save_path, 'image', 'left'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'image', 'right'), exist_ok=True)
            save_imgs(os.path.join(save_path, 'image', 'left'), streaming_data["image"]["left"])
            save_imgs(os.path.join(save_path, 'image', 'right'), streaming_data["image"]["right"])
        
        os.makedirs(os.path.join(save_path, 'pose'), exist_ok=True)
        np.save(os.path.join(save_path, 'pose', "timestamps.npy"), np.array(streaming_data["pose"]["timestamp_ms"], dtype=float))
        
        if len(streaming_data["pose"]["timestamp_ms"]) > 1:
            freq = len(streaming_data["pose"]["timestamp_ms"]) / (streaming_data["pose"]["timestamp_ms"][-1] - streaming_data["pose"]["timestamp_ms"][0] + 1e-6) * 1000
            draw_time(streaming_data["pose"]["timestamp_ms"], os.path.join(save_path, 'pose', f"freq_{freq:.2f}.png"))
            
        np.save(os.path.join(save_path, 'pose', "xyz.npy"), np.array(streaming_data["pose"]["xyz"], dtype=float))
        np.save(os.path.join(save_path, 'pose', "quat.npy"), np.array(streaming_data["pose"]["quat"], dtype=float))

    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect

    def _process_gyro(self, gyro_frame):
        # 姿态积分逻辑不变
        ts = gyro_frame.get_timestamp() / 1000.0 
        
        if self._last_ts_gyro is None:
            self._last_ts_gyro = ts
            return

        dt = ts - self._last_ts_gyro
        self._last_ts_gyro = ts
        if dt > 0.1: return

        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        wx, wy, wz = gyro_data.x, gyro_data.y, gyro_data.z
        
        rot_vec = np.array([wx, wy, wz]) * dt
        delta_rot = Rot.from_rotvec(rot_vec)
        
        current_rot = Rot.from_quat(self._current_quat)
        new_rot = current_rot * delta_rot
        
        self._current_quat = new_rot.as_quat()

    def _process_accel(self, accel_frame):
        """
        核心修改：加速度计双重积分，计算位置 (P)
        !!! WARNING: 严重漂移 !!!
        此计算未进行重力补偿或坐标系转换。结果不可用于实际导航。
        """
        ts = accel_frame.get_timestamp() / 1000.0 
        
        if self._last_ts_accel is None:
            self._last_ts_accel = ts
            return

        dt = ts - self._last_ts_accel
        self._last_ts_accel = ts
        if dt > 0.1: return

        accel_data = accel_frame.as_motion_frame().get_motion_data()
        # a 是原始测量值，包含了 9.8m/s^2 的重力加速度。
        a = np.array([accel_data.x, accel_data.y, accel_data.z])
        
        # 积分 1: 速度 V = V_old + a * dt
        # 这一步已经包含了巨大的漂移。
        self._current_vel += a * dt

        # 积分 2: 位置 P = P_old + V_new * dt
        # 这一步引入了二次方漂移。
        self._current_xyz += self._current_vel * dt
    
    def callback(self, frame):
        ts = time.time() * 1000 
        if not self._collect_streaming_data:
            return
        
        if frame.is_frameset() and self._image:
            frameset = frame.as_frameset()
            f_left = frameset.get_infrared_frame(1)
            f_right = frameset.get_infrared_frame(2)

            left_data = np.asanyarray(f_left.get_data(), dtype=np.uint8)
            right_data = np.asanyarray(f_right.get_data(), dtype=np.uint8)
            self.image_streaming_mutex.acquire()
            if len(self.image_streaming_data["timestamp_ms"]) != 0 and ts == self.image_streaming_data["timestamp_ms"][-1]:
                pass
            else:
                self.image_streaming_data["left"].append(left_data.copy())
                self.image_streaming_data["right"].append(right_data.copy())
                self.image_streaming_data["timestamp_ms"].append(ts)
            self.image_streaming_mutex.release()

        if frame.is_motion_frame():
            motion_frame = frame.as_motion_frame()
            motion_data = motion_frame.get_motion_data()
            motion_profile = frame.get_profile()
            if motion_profile.stream_type() == rs.stream.gyro:
                self._process_gyro(frame)
            elif motion_profile.stream_type() == rs.stream.accel:
                self._process_accel(frame)
            quat =  np.array(self._current_quat.copy())
            xyz = np.array(self._current_xyz.copy())
            self.pose_streaming_mutex.acquire()
            if len(self.pose_streaming_data["timestamp_ms"]) != 0 and ts == self.pose_streaming_data["timestamp_ms"][-1]:
                pass
            else:
                self.pose_streaming_data["xyz"].append(xyz)
                self.pose_streaming_data["quat"].append(quat)
                self.pose_streaming_data["timestamp_ms"].append(ts)
            self.pose_streaming_mutex.release()
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        # T265 的静态方法保持不变
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
        pose_4x4[:3, 3] = xyz
        return pose_4x4

    def __del__(self) -> None:
        try:
            self.pipeline.stop()
        except:
            pass


if __name__ == "__main__":
    camera = D455(id=None, image=True, name='D455')
    
    try:
        camera.start_streaming(callback=None)

        i = 0
        while True:
            # 持续打印位姿，可以看到 XYZ 在快速变化
            print(f"Frame: {i:04d}", end='\r')
            
            left, right, xyz, quat = camera.get()

            # 打印信息：XYZ 现在会快速漂移！
            print(f"XYZ (m): [{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] | QUAT: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]", end='\r') 
            
            if left is not None and right is not None:
                cv2.imshow('left', left)
                cv2.imshow('right', right)
            
            key = cv2.waitKey(1)
            
            if key == 27: # ESC 退出
                break
            
            i += 1
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 退出后，询问是否保存数据
        cv2.destroyAllWindows()
        cmd = input("\n\nStreaming stopped. Whether save? (y/n): ") 
        
        data = None
        if cmd == 'y':
            data = camera.stop_streaming()
            if data:
                save_path = f"./data_d455_double_integration_{int(time.time())}"
                print(f"Saving data to {save_path}...")
                camera.save_streaming(save_path, data)
                print("Saved.")
        else:
            camera.stop_streaming()