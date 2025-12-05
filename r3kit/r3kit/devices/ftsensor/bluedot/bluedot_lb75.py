#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlueDotLB75 (Modbus TCP) - full implementation matching functionality of PyATI example.

Features:
- connect/disconnect via Modbus TCP (pymodbus)
- single read and multi-read parsing (tries IEEE754 floats first, fallback to 16-bit scaled ints)
- streaming mode (background thread) with optional callback or internal buffering
- thread-safe access to latest frame(s)
- save_streaming -> save .npy files and call visualization hooks
- raw2tare static method for gravity & bias compensation
- compatible method names and behaviors similar to PyATI

Requires:
- pymodbus
- numpy
- r3kit (FTSensorBase, bluedot.config, vis drawing functions)
"""

import os
import time
import struct
from copy import deepcopy
from threading import Thread, Lock, Event
from functools import partial
from typing import Optional, Callable, Dict, Union, List

import numpy as np
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# NOTE: these constants should be provided by your project's config.
# If r3kit.devices.ftsensor.bluedot.config exists, prefer that import.
try:
    from r3kit.devices.ftsensor.bluedot.config import *
except Exception:
    # Fallback defaults (override as needed)
    BLUEDOT_IP = "192.168.0.20"
    BLUEDOT_PORT = 502
    BLUEDOT_FPS = 100
    BLUEDOT_SCALE = 1.0          # generic scale; many sensors provide native floats so scale=1.0
    BLUEDOT_RETRY = 3
    BLUEDOT_RETRY_DELAY = 0.5
    BLUEDOT_ID = BLUEDOT_IP
    BLUEDOT_SLAVE_ID = 20

# visualization utilities (optional; if missing, code will still run but won't draw)
try:
    from r3kit.utils.vis import draw_time, draw_items
except Exception:
    def draw_time(*args, **kwargs):
        # placeholder
        return
    def draw_items(*args, **kwargs):
        return

# Base class
try:
    from r3kit.devices.ftsensor.base import FTSensorBase
except Exception:
    # Minimal fallback base class so code remains runnable if r3kit not present.
    class FTSensorBase:
        def __init__(self, name: str = "BlueDotLB75"):
            self.name = name


class BlueDotLB75(FTSensorBase):
    """
    Six-axis force/torque sensor interface over Modbus TCP.
    Matches PyATI-like interface: connect/disconnect, get(), start_streaming/stop_streaming,
    save_streaming(), raw2tare(), etc.
    """

    # Default register map (modify according to actual device manual)
    DEFAULT_REG_MAP = {
        'fx': 0,    # starting register for Fx (two registers for 32-bit float or two 16-bit words)
        'fy': 2,
        'fz': 4,
        'mx': 6,
        'my': 8,
        'mz': 10,
        'status': 12,
        'timestamp': 13
    }

    def __init__(
        self,
        id: str = BLUEDOT_ID,
        host: str = BLUEDOT_IP,
        port: int = BLUEDOT_PORT,
        slave_id: int = BLUEDOT_SLAVE_ID,
        fps: int = BLUEDOT_FPS,
        name: str = "BlueDotLB75",
        retry: int = BLUEDOT_RETRY,
        retry_delay: float = BLUEDOT_RETRY_DELAY,
        scale: float = BLUEDOT_SCALE,
        register_map: dict = None,
        write_init_register: tuple = (0x0000 ,1)
    ) -> None:
        super().__init__(name=name)
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.fps = fps
        self.retry = retry
        self.retry_delay = retry_delay
        self.scale = scale

        self.registers = register_map if register_map is not None else deepcopy(self.DEFAULT_REG_MAP)

        self.client: Optional[ModbusTcpClient] = None
        self.connected = False

        # Streaming related
        self.in_streaming = Event()
        self._collect_streaming_data = True
        self.streaming_mutex = Lock()
        self.streaming_data = {"ft": [], "timestamp_ms": []}
        self.thread: Optional[Thread] = None

        # for get() while streaming
        self.latest_mutex = Lock()
        self.latest_frame = None  # dict {'ft': np.array(6,), 'timestamp_ms': float}

        print(f"[BlueDotLB75] connecting to {self.host}:{self.port} ...")
        ok = self.connect(write_init_register=write_init_register)
        if ok:
            print("[BlueDotLB75] Connected successfully.")
        else:
            print("[BlueDotLB75] ⚠️ Auto-connect failed. You may need to call connect() manually.")

    # -------------------------
    # Connection Management
    # -------------------------
    def connect(self, write_init_register: Optional[tuple] = None, timeout: float = 3.0) -> bool:
        """
        Connect to Modbus TCP device. Optionally write an init register (address, value) if provided.

        Args:
            write_init_register: (address:int, value:int) to write after connect (e.g., wake/start)
            timeout: socket timeout seconds
        Returns:
            bool: True if connected
        """
        attempts = 0
        while attempts < self.retry:
            try:
                self.client = ModbusTcpClient(host=self.host, port=self.port, timeout=timeout)
                ok = self.client.connect()
                if not ok:
                    attempts += 1
                    time.sleep(self.retry_delay)
                    continue
                self.connected = True
                # optional init write (some sensors require writing to a register to enable streaming)
                if write_init_register is not None:
                    addr, val = write_init_register
                    rr = self.client.write_register(addr, val)
                    # don't fail on write error here; just warn
                    if hasattr(rr, "isError") and rr.isError():
                        print("[BlueDotLB75] warning: init write failed:", rr)
                    else:
                        print("[BlueDotLB75] init write successful")
                return True
            except Exception as e:
                print(f"[BlueDotLB75] connect attempt {attempts+1} failed: {e}")
                attempts += 1
                time.sleep(self.retry_delay)
        self.connected = False
        return False

    def disconnect(self) -> None:
        """
        Close Modbus TCP client connection.
        """
        try:
            if self.in_streaming.is_set():
                # stop if streaming
                self.stop_streaming()
        except Exception:
            pass
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.connected = False

    # -------------------------
    # Low level reading/parsing
    # -------------------------
    def _read_registers(self, start_addr: int, count: int) -> Optional[List[int]]:
        """
        Read consecutive holding registers.

        Returns:
            list of 16-bit register values (ints) or None on failure.
        """
        if not self.connected or self.client is None:
            raise RuntimeError("Not connected to sensor")

        try:
            resp = self.client.read_holding_registers(address=start_addr, count=count)
            if resp is None or hasattr(resp, "isError") and resp.isError():
                # handle None or error response
                return None
            return resp.registers
        except Exception as e:
            print("[BlueDotLB75] _read_registers exception:", e)
            return None

    @staticmethod
    def _registers_to_bytes(registers: List[int]) -> bytes:
        """
        Convert a list of 16-bit registers (ints 0..65535 or signed) to bytes in big-endian word order.
        """
        b = bytearray()
        for r in registers:
            r16 = r & 0xFFFF
            b.extend(struct.pack(">H", r16))
        return bytes(b)

    @staticmethod
    def _parse_as_floats(registers: List[int]) -> Optional[np.ndarray]:
        """
        Try to parse 12 registers -> 6 IEEE754 big-endian floats.
        Return np.array(6,) or None on failure.
        """
        if len(registers) < 12:
            return None
        raw_bytes = BlueDotLB75._registers_to_bytes(registers[:12])
        try:
            vals = struct.unpack(">6f", raw_bytes)  # big-endian float words
            return np.array(vals, dtype=float)
        except Exception:
            return None

    @staticmethod
    def _parse_as_scaled_ints(registers: List[int], scale: float = 32.768) -> Optional[np.ndarray]:
        """
        Fallback parser: interpret each value as two 16-bit words forming signed 32-bit int,
        or interpret each 16-bit as signed int16 and scale.
        Many simple sensors store each axis as a single 16-bit signed int with scale factor.
        We'll try common options:
         - If registers length >= 12, interpret each axis as signed 16 + scale: (reg - 32768)/scale
         - If interpretation as pairs -> signed 32-bit -> divide by scale (if provided)
        """
        if len(registers) < 12:
            return None
        # Option A: treat each axis as signed 16-bit (first word only) with offset 32768 logic
        try:
            arr = []
            for i in range(6):
                w = registers[i*2] & 0xFFFF
                # if typical sensor encodes uint16 representing signed via offset 32768:
                val = (int(w) - 32768) / scale
                arr.append(val)
            return np.array(arr, dtype=float)
        except Exception:
            pass

        return None

    def _recv_data(self) -> Optional[np.ndarray]:
        """
        Read consecutive registers and parse into a 6-vector [Fx, Fy, Fz, Mx, My, Mz].
        Tries IEEE754 floats first, then falls back to scaled int interpretation.
        Returns numpy array shape (6,) or None on failure.
        """
        start = self.registers['fx']
        # we expect 6 * 32-bit values => 12 registers
        reg_count = 12
        registers = self._read_registers(start, reg_count)
        if registers is None:
            return None

        # Try parsing as 6 IEEE754 floats (common in many sensors)
        vals = self._parse_as_floats(registers)
        if vals is not None and np.isfinite(vals).all():
            return vals * self.scale  # apply additional scale if needed

        # Fallback: scaled ints style
        vals = self._parse_as_scaled_ints(registers, scale=32.768)
        if vals is not None:
            return vals * self.scale

        # If all parsing fails, return None
        return None

    # -------------------------
    # High level read / get
    # -------------------------
    def _read(self, n: int = 1) -> Dict[str, Union[float, np.ndarray]]:
        """
        Read n samples sequentially (blocking).
        Returns dict {'ft': np.ndarray (n,6), 'timestamp_ms': float}
        """
        fts = np.empty((n, 6), dtype=float)
        for i in range(n):
            vals = self._recv_data()
            if vals is None:
                # if a read failed, fill with nan and continue
                fts[i, :] = np.nan
            else:
                fts[i, :] = vals
        receive_time = time.time() * 1000.0
        return {"ft": fts, "timestamp_ms": receive_time}

    def get(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Return latest single sample.
        If streaming is active, return last cached frame; else perform single blocking read.
        """
        if not self.in_streaming.is_set():
            # not streaming -> do a single read
            return self._read(n=1)
        else:
            # streaming -> return latest cached frame
            with self.latest_mutex:
                if self.latest_frame is None:
                    # no data yet -> fallback to blocking read
                    return self._read(n=1)
                # return copy
                return {"ft": np.array(self.latest_frame["ft"]).reshape(1, 6), "timestamp_ms": float(self.latest_frame["timestamp_ms"])}

    # -------------------------
    # Streaming control
    # -------------------------
    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        """
        Start background streaming.
        If callback is provided, it's called with a deepcopy of {'ft': np.array(6,), 'timestamp_ms': float} per frame.
        Otherwise data are buffered internally in self.streaming_data.
        """
        if self.in_streaming.is_set():
            # already streaming
            return

        # prepare buffers
        with self.streaming_mutex:
            self._collect_streaming_data = True
            self.streaming_data = {"ft": [], "timestamp_ms": []}
        self.in_streaming.set()

        self.thread = Thread(target=partial(self._streaming_loop, callback), daemon=True)
        self.thread.start()

    def _streaming_loop(self, callback: Optional[Callable] = None) -> None:
        """
        Internal loop run in background thread to poll sensor at self.fps
        """
        interval = 1.0 / max(1.0, float(self.fps))
        while self.in_streaming.is_set():
            t0 = time.time()
            vals = self._recv_data()
            ts = time.time() * 1000.0
            if vals is not None:
                frame = {"ft": vals.copy(), "timestamp_ms": ts}
                # update latest frame
                with self.latest_mutex:
                    self.latest_frame = deepcopy(frame)
                # store or callback
                if callback is None:
                    with self.streaming_mutex:
                        if self._collect_streaming_data:
                            self.streaming_data["ft"].append(frame["ft"])
                            self.streaming_data["timestamp_ms"].append(frame["timestamp_ms"])
                else:
                    try:
                        callback(deepcopy(frame))
                    except Exception as e:
                        # user callback error shouldn't kill loop
                        print("[BlueDotLB75] callback exception:", e)
            # sleep remainder
            elapsed = time.time() - t0
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def collect_streaming(self, collect: bool = True) -> None:
        """
        Enable/disable internal buffering of streaming data.
        """
        with self.streaming_mutex:
            self._collect_streaming_data = bool(collect)

    def stop_streaming(self) -> Dict[str, List]:
        """
        Stop streaming and return buffered data (ft list, timestamp list).
        If callback mode was used, buffered data will be empty.
        """
        if not self.in_streaming.is_set():
            return self.streaming_data
        self.in_streaming.clear()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        # return a copy
        with self.streaming_mutex:
            data = {"ft": list(self.streaming_data["ft"]), "timestamp_ms": list(self.streaming_data["timestamp_ms"])}
            # clear internal storage
            self.streaming_data = {"ft": [], "timestamp_ms": []}
        return data

    # -------------------------
    # Persistence & visualization
    # -------------------------
    def save_streaming(self, save_path: str, streaming_data: Dict[str, List]) -> None:
        """
        Save buffered streaming data to folder:
         - timestamps.npy
         - ft.npy
         - freq_{freq:.2f}.png (via draw_time)
         - ft.png (via draw_items)
        """
        assert len(streaming_data["ft"]) == len(streaming_data["timestamp_ms"]), "length mismatch"
        os.makedirs(save_path, exist_ok=True)
        ts_arr = np.array(streaming_data["timestamp_ms"], dtype=float)
        ft_arr = np.array(streaming_data["ft"], dtype=float)
        np.save(os.path.join(save_path, "timestamps.npy"), ts_arr)
        np.save(os.path.join(save_path, "ft.npy"), ft_arr)
        # compute approximate sampling frequency (Hz)
        if len(ts_arr) >= 2 and (ts_arr[-1] - ts_arr[0]) > 0:
            freq = len(ts_arr) / ((ts_arr[-1] - ts_arr[0]) / 1000.0)
        else:
            freq = float(self.fps)
        try:
            draw_time(list(streaming_data["timestamp_ms"]), os.path.join(save_path, f"freq_{freq:.2f}.png"))
        except Exception:
            pass
        try:
            draw_items(ft_arr, os.path.join(save_path, "ft.png"))
        except Exception:
            pass

    # -------------------------
    # Utility: tare compensation
    # -------------------------
    @staticmethod
    def raw2tare(raw_ft: np.ndarray, tare: Dict[str, Union[float, np.ndarray]], pose: np.ndarray) -> np.ndarray:
        """
        Compensate raw force/torque reading with tare (bias) and gravity.
        raw_ft: (6,) array: [fx,fy,fz,mx,my,mz]
        tare: dict with keys:
            'f0' : np.array(3,)   (force bias)
            't0' : np.array(3,)   (torque bias)
            'm'  : float          (mass of tool)
            'c'  : np.array(3,)   (center of mass relative to sensor)
        pose: 3x3 rotation matrix from tool/sensor to base frame
        """
        raw_f = np.array(raw_ft[:3], dtype=float)
        raw_t = np.array(raw_ft[3:], dtype=float)

        f = raw_f - np.array(tare.get('f0', np.zeros(3)), dtype=float)
        # gravity in base frame: [0,0,-9.8*m]; map to sensor via inv(pose)
        m = float(tare.get('m', 0.0))
        c = np.array(tare.get('c', np.zeros(3)), dtype=float)
        t0 = np.array(tare.get('t0', np.zeros(3)), dtype=float)

        # inverse rotation (transpose)
        Rinv = np.linalg.inv(pose)
        gravity_sensor = Rinv @ np.array([0.0, 0.0, -9.8 * m])
        f = f - gravity_sensor

        t = raw_t - t0
        # torque due to weight: r x F (r is center in sensor frame)
        r_sensor = Rinv @ c
        t = t - np.cross(r_sensor, gravity_sensor)
        return np.concatenate([f, t])

    # -------------------------
    # Debug printing helper
    # -------------------------
    def print_sensor_data(self, data: Dict[str, Union[float, np.ndarray]]) -> None:
        """
        Nicely print the latest data (single-frame or multi-frame).
        Accepts dict returned by get() or _read().
        """
        if data is None:
            print("[BlueDotLB75] No data")
            return

        # if data['ft'] is (N,6) choose last row
        ft = data['ft']
        if isinstance(ft, np.ndarray) and ft.ndim == 2:
            f = ft[-1]
        else:
            f = np.array(ft).reshape(6,)

        ts = data.get('timestamp_ms', time.time() * 1000.0)
        print("=" * 50)
        print(f"Timestamp: {ts:.3f} ms")
        print(f"Fx: {f[0]:8.4f}  Fy: {f[1]:8.4f}  Fz: {f[2]:8.4f}")
        print(f"Mx: {f[3]:8.4f}  My: {f[4]:8.4f}  Mz: {f[5]:8.4f}")
        print("=" * 50)
    
    def get_mean_data(self, n: int = 1, name: str = 'ft') -> Optional[np.ndarray]:
        """
        Read n samples sequentially and return the mean of the specified data ('ft').
        This is a blocking read operation.
        """
        # 使用已有的 _read 方法获取 n 帧数据
        data = self._read(n=n) 
        
        # 检查返回的数据是否有效
        if data is None or name not in data or data[name].size == 0:
            return None
        
        # 确保数据是 (N, 6) 的 NumPy 数组
        ft_array = data[name]
        
        # 返回平均值 (沿第一轴 N 取平均)
        if ft_array.ndim == 2 and ft_array.shape[0] > 0:
            return np.mean(ft_array, axis=0)
        elif ft_array.ndim == 1:
            return ft_array # 如果只有一帧 (N=1), 直接返回该帧
        else:
            return None # 数据格式异常


# -------------------------
# Example main
# -------------------------
def main():
    SENSOR_IP = BLUEDOT_IP
    SENSOR_PORT = BLUEDOT_PORT
    SLAVE_ID = BLUEDOT_SLAVE_ID

    sensor = BlueDotLB75(host=SENSOR_IP, port=SENSOR_PORT, slave_id=SLAVE_ID, fps=100)

    print(f"Connecting to {SENSOR_IP}:{SENSOR_PORT} ...")
    if not sensor.connect(write_init_register=None):
        print("Connection failed. Check network / device / config.")
        return

    try:
        # Example: start streaming to internal buffer
        sensor.start_streaming(callback=None)
        # collect for 5 seconds
        time.sleep(5.0)
        buffered = sensor.stop_streaming()
        print(f"Buffered frames: {len(buffered['ft'])}")
        # print last frame
        if len(buffered['ft']) > 0:
            last = {"ft": np.array(buffered['ft']), "timestamp_ms": buffered['timestamp_ms'][-1]}
            sensor.print_sensor_data(last)
        # save
        sensor.save_streaming("sensor_saved", buffered)

        # Example: blocking single reads
        for i in range(3):
            d = sensor._read(n=1)
            sensor.print_sensor_data(d)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sensor.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
