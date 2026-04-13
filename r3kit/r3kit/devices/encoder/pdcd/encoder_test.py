#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import time

# 串口配置
PORT = "/dev/ttyUSB0"
BAUD = 115200


def open_serial():
    """打开串口"""
    try:
        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=0.5,  # 读超时
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        print(f"[OK] 成功打开串口: {PORT}")
        return ser
    except Exception as e:
        print(f"[ERROR] 打开串口失败: {e}")
        exit(1)


def send_cmd(ser, cmd: str):
    """发送命令（自动添加 \r\n）"""
    full_cmd = cmd + "\r\n"
    ser.write(full_cmd.encode())
    print(f"[SEND] {cmd}")


def read_response(ser):
    """读取所有可读数据"""
    time.sleep(0.05)
    data = ser.read_all().decode(errors="ignore")
    if data:
        print("[RECV]")
        print(data)
    else:
        print("[RECV] (无返回)")


def main():
    ser = open_serial()

    print("\n=== 迷你编码器测试工具 ===")
    print("输入命令，例如：")
    print(" help  adc  pulse  enc  zero  spi  mode 1  zc 200  reboot")
    print(" 输入 exit 退出程序")
    print("==========================\n")

    while True:
        try:
            cmd = input("输入串口命令> ").strip()
            if cmd == "":
                continue
            if cmd.lower() == "exit":
                print("退出程序")
                break

            send_cmd(ser, cmd)
            read_response(ser)

        except KeyboardInterrupt:
            print("\n用户中断，退出")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

    ser.close()


if __name__ == "__main__":
    # main()
    while True:
        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=0.5,  # 读超时
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        full_cmd = "spi" + "\r\n"
        ser.write(full_cmd.encode())
        ser.write(b"spi\r\n")  # 发送 SPI 命令
        time.sleep(0.05)
        raw = ser.read_all().decode(errors="ignore")
        print(raw)
