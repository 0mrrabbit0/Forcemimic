import os
import shutil
import time
import configargparse
import json

# 第三视角摄像头
from r3kit.devices.camera.realsense.d415 import D415
# 第一视角摄像头
from r3kit.devices.camera.realsense.d455 import D455
# 力矩传感器
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft
# 角度编码器
from r3kit.devices.encoder.pdcd.angler_hz import Angler


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--d415_id', type=str)
    parser.add_argument('--l515_id', type=str)
    parser.add_argument('--d455_id', type=str)
    parser.add_argument('--bdft_id', type=str)
    parser.add_argument('--bdft_port', type=int)
    parser.add_argument('--bdft_tare_path', type=str)
    parser.add_argument('--angler_id', type=str)
    parser.add_argument('--angler_index', type=int)

    args = parser.parse_args()
    return args


def main(args):
    # create devices
    d415 = D415(id=args.d415_id, name='d415')
    d455 = D455(id=args.d455_id, name='d455')
    bluedot_lb75 = Bdft(id=args.bdft_id, port=args.bdft_port, fps=200, name='bdft')
    angler = Angler(id=args.angler_id, index=args.angler_index, fps=30, name='angler')

    # special prepare to close the gripper for angler to know bias
    # 关闭夹爪，校准编码器
    start = input('Press enter to start closing gripper')
    if start != '':
        del d415, d455, bluedot_lb75, angler
        return
    print("Start closing gripper")

    # special prepare for d455 to construct map
    # 构建SLAM
    stop_start = input('Press enter to stop closing gripper and start D455 construct map')
    if stop_start != '':
        del d415, d455, bluedot_lb75, angler
        return

    print("Stop closing gripper and Start D455 construct map")
    d455.collect_streaming(False)
    d455.start_streaming()

    # stable d455 pose
    # 稳定SLAM相机
    stop_start = input('Press enter to stop D455 construct map and start stabling D455 pose')
    if stop_start != '':
        del d415, d455, bluedot_lb75, angler
        return

    print("Stop D455 construct map and start stabling D455 pose")
    d455.collect_streaming(True)
    d455_pose_start_timestamp_ms = time.time() * 1000

    # mounting d455
    # 安装SLAM相机
    stop_start = input('Press enter to stop stabling D455 pose and start mounting D455')
    if stop_start != '':
        del d415, d455, bluedot_lb75, angler
        return

    print("Stop stabling D455 pose and start mounting D455")
    d455_pose_end_timestamp_ms = time.time() * 1000

    # start streaming
    # 开始采集
    stop_start = input('Press enter to stop mounting D455 and start streaming')
    if stop_start != '':
        del d415, d455, bluedot_lb75, angler
        return

    print("Stop mounting D455 and start streaming")
    start_timestamp_ms = time.time() * 1000
    d415.start_streaming()
    bluedot_lb75.start_streaming()
    angler.start_streaming()

    # collect data
    # 停止采集
    stop = input('Press enter to stop streaming')
    if stop != '':
        del d415, d455, bluedot_lb75, angler
        return
    print("Stop streaming")

    # stop streaming
    d415_data = d415.stop_streaming()
    d455_data = d455.stop_streaming()
    bdft_data = bluedot_lb75.stop_streaming()
    angler_data = angler.stop_streaming()

    # save data
    print("Start saving data")
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'stage_timestamp_ms.json'), 'w') as f:
        json.dump({
            'd455_pose_start_timestamp_ms': d455_pose_start_timestamp_ms,
            'd455_pose_end_timestamp_ms': d455_pose_end_timestamp_ms,
            'start_timestamp_ms': start_timestamp_ms
        }, f, indent=4)
    os.makedirs(os.path.join(args.save_path, 'd415'), exist_ok=True)
    d415.save_streaming(os.path.join(args.save_path, 'd415'), d415_data)
    os.makedirs(os.path.join(args.save_path, 'd455'), exist_ok=True)
    d455.save_streaming(os.path.join(args.save_path, 'd455'), d455_data)
    os.makedirs(os.path.join(args.save_path, 'bdft'), exist_ok=True)
    shutil.copyfile(os.path.join(args.bdft_tare_path, "tare_bdft.json"), os.path.join(args.save_path, 'bdft', "tare_bdft.json"))
    bluedot_lb75.save_streaming(os.path.join(args.save_path, 'bdft'), bdft_data)
    os.makedirs(os.path.join(args.save_path, 'angler'), exist_ok=True)
    angler.save_streaming(os.path.join(args.save_path, 'angler'), angler_data)
    print("Stop saving data")

    del d415, d455, bluedot_lb75, angler


if __name__ == '__main__':
    args = config_parse()
    main(args)
