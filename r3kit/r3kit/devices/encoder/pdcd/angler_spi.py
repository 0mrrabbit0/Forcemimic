import os
from typing import Dict, Optional
import time
import numpy as np
from threading import Thread, Lock, Event
from copy import deepcopy
from functools import partial
import serial

from r3kit.devices.encoder.base import EncoderBase

from r3kit.utils.vis import draw_time, draw_items


class Angler(EncoderBase):
    """
    SPI 版本：按你提供的迷你编码器评估板文档实现
    串口协议：
        指令: spi\r\n
        返回: 字符串，格式例如 "count:1234 angle:56.78"
    """

    def __init__(
        self,
        id: str = "/dev/ttyUSB0",
        index: int = 1,
        fps: int = 30,
        baudrate: int = 115200,
        gap: float = 0.002,
        name: str = "Angler",
    ) -> None:
        super().__init__(name=name)

        self._id = id
        self._index = index
        self._fps = fps
        self._baudrate = baudrate
        self._gap = gap

        # serial
        self.ser = serial.Serial(
            id, baudrate=baudrate, bytesize=8, stopbits=1, parity="N", timeout=0.1
        )

        if not self.ser.is_open:
            raise RuntimeError("Fail to open serial port")

        self.ser.flushInput()
        self.ser.flushOutput()

        self.in_streaming = Event()

    # -------------------------
    # 核心：SPI 获取数据
    # -------------------------
    def _read(self) -> Optional[Dict[str, float]]:
        self.ser.flushInput()
        full_cmd = "spi" + "\r\n"
        self.ser.write(full_cmd.encode())
        self.ser.write(b"spi\r\n")  # 发送 SPI 命令
        time.sleep(0.01)

        raw = self.ser.read_all().decode(errors="ignore")
        if not raw:
            return None
        items = raw.split()

        count = None
        angle = None

        count = float(items[0].split(":")[1])
        angle = float(items[1].split(":")[1])

        if angle is None:
            return None

        return {"count": count, "angle": angle, "timestamp_ms": time.time() * 1000}

    def get(self) -> Optional[Dict[str, float]]:
        if not self.in_streaming.is_set():
            return self._read()
        else:
            self.streaming_mutex.acquire()
            data = {
                "angle": self.streaming_data["angle"][-1],
                "timestamp_ms": self.streaming_data["timestamp_ms"][-1],
            }
            self.streaming_mutex.release()
            return data

    def get_mean_data(self, n=10, name="angle") -> float:
        assert name in ["angle"], "name must be one of [angle]"
        tare_list = []
        count = 0
        while count < n:
            data = self.get()
            if data is not None:
                tare_list.append(data[name])
                count += 1
        tare = sum(tare_list) / n
        return tare

    def start_streaming(self, callback: Optional[callable] = None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        self.in_streaming.set()

        self.streaming_mutex = Lock()
        self.streaming_data = {"angle": [], "timestamp_ms": []}

        self.thread = Thread(
            target=partial(self._streaming_data, callback=callback), daemon=True
        )
        self.thread.start()

    def stop_streaming(self) -> dict:
        self.in_streaming.clear()
        self.thread.join()
        self.streaming_mutex = None
        data = self.streaming_data
        self.streaming_data = {"angle": [], "timestamp_ms": []}
        return data

    def save_streaming(self, save_path: str, streaming_data: dict) -> None:
        assert len(streaming_data["angle"]) == len(streaming_data["timestamp_ms"])
        np.save(
            os.path.join(save_path, "timestamps.npy"),
            np.array(streaming_data["timestamp_ms"], dtype=float),
        )
        freq = len(streaming_data["timestamp_ms"]) / (
            streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0]
        )
        draw_time(
            streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png")
        )
        np.save(
            os.path.join(save_path, "angle.npy"),
            np.array(streaming_data["angle"], dtype=float),
        )
        draw_items(
            np.array(streaming_data["angle"], dtype=float),
            os.path.join(save_path, "angle.png"),
        )

    def collect_streaming(self, collect: bool = True) -> None:
        self._collect_streaming_data = collect

    def _streaming_data(self, callback=None):
        while self.in_streaming.is_set():
            time.sleep(1 / self._fps)

            if not self._collect_streaming_data:
                continue

            data = self._read()
            if data is None:
                continue

            if callback is None:
                self.streaming_mutex.acquire()
                self.streaming_data["angle"].append(data["angle"])
                self.streaming_data["timestamp_ms"].append(data["timestamp_ms"])
                self.streaming_mutex.release()
            else:
                callback(deepcopy(data))

    @staticmethod
    def raw2angle(raw: np.ndarray) -> np.ndarray:
        result = []
        assert len(raw) > 100
        if np.any(raw[:10] < 1) and np.any(raw[:10] > 359):
            initial_angle = raw[0]
        else:
            assert np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25) < 1, (
                np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25)
            )
            initial_angle = np.median(raw[:10])
        count = 0
        result.append(raw[0] - initial_angle + 360 * count)
        for i in range(1, len(raw)):
            if abs(raw[i] - raw[i - 1]) > 100:
                count += 1 if raw[i] - raw[i - 1] < 0 else -1
            result.append(raw[i] - initial_angle + 360 * count)
        return np.array(result)


if __name__ == "__main__":
    encoder = Angler(id="/dev/ttyUSB0", index=1, baudrate=115200, fps=30)

    while True:
        data = encoder.get()
        print(data)
        time.sleep(0.1)
