import os
import time
import configargparse
import json
from copy import deepcopy
from pynput import keyboard
import numpy as np

from r3kit.devices.camera.realsense.d455 import D455
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft
from configs.pose import *  # noqa: F403, F405


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--d455_id", type=str)
    parser.add_argument("--bdft_id", type=str)
    parser.add_argument("--bdft_port", type=int)

    args = parser.parse_args()
    return args


def main(args):
    # read tare data
    with open(os.path.join(args.save_path, "tare_bdft.json"), "r") as f:
        tare_data = json.load(f)
        for key, value in tare_data.items():
            if isinstance(value, list):
                tare_data[key] = np.array(value)

    # create devices
    d455 = D455(id=args.d455_id, image=False, name="d455")
    bdft = Bdft(id=args.bdft_id, port=args.bdft_port, fps=200, name="bdft")

    # special prepare for d455 to construct map
    start = input("Press enter to start D455 construct map")
    if start != "":
        del d455, bdft
        return
    print("Start D455 construct map")
    d455.collect_streaming(False)
    d455.start_streaming()

    # stable d455 pose
    stop_start = input(
        "Press enter to stop D455 construct map and start stabling D455 pose"
    )
    if stop_start != "":
        del d455, bdft
        return
    print("Stop D455 construct map and start stabling D455 pose")
    d455.collect_streaming(True)
    d455_pose_start_timestamp_ms = time.time() * 1000

    # mounting d455
    stop_start = input("Press enter to stop stabling D455 pose and start mounting D455")
    if stop_start != "":
        del d455, bdft
        return
    print("Stop stabling D455 pose and start mounting D455")
    d455_pose_end_timestamp_ms = time.time() * 1000

    # NOTE: urgly code to get d455 initial pose
    d455.pose_streaming_mutex.acquire()
    d455_all_poses = deepcopy(d455.pose_streaming_data)
    d455.pose_streaming_mutex.release()
    d455_initial_pose_mask = np.logical_and(
        np.array(d455_all_poses["timestamp_ms"]) > d455_pose_start_timestamp_ms,
        np.array(d455_all_poses["timestamp_ms"]) < d455_pose_end_timestamp_ms,
    )
    d455_initial_xyzs = np.array(d455_all_poses["xyz"])[d455_initial_pose_mask]
    d455_initial_quats = np.array(d455_all_poses["quat"])[d455_initial_pose_mask]
    d455_initial_xyz = np.median(d455_initial_xyzs, axis=0)
    d455_initial_quat = np.median(d455_initial_quats, axis=0)
    d455_initial_pose = D455.raw2pose(d455_initial_xyz, d455_initial_quat)  # c02w

    # start streaming
    stop_start = input("Press enter to stop mounting D455 and start streaming")
    if stop_start != "":
        del d455, bdft
        return
    print("Stop mounting D455 and start streaming")
    print("Press enter to stop streaming")
    stop = False

    def _on_press(key):
        nonlocal stop
        if key == keyboard.Key.enter:
            stop = True

    def _on_release(key):
        pass

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()
    while not stop:
        bdft_ft = bdft.get_mean_data(n=10, name="ft")
        _, _, d455_xyz, d455_quat = d455.get()
        d455_pose = D455.raw2pose(d455_xyz, d455_quat)  # c2w
        d455_pose = np.linalg.inv(d455_initial_pose) @ d455_pose  # c2c0 = w2c0 @ c2w
        d455_pose = T265r_2_BASE @ d455_pose  # noqa: F405  # c2b = c02b @ c2c0
        bdft_pose = d455_pose @ np.linalg.inv(T265r_2_PYFT)  # noqa: F405  # f2b = c2b @ f2c
        bdft_ft = Bdft.raw2tare(raw_ft=bdft_ft, tare=tare_data, pose=bdft_pose[:3, :3])
        print(np.linalg.norm(bdft_ft[:3]), np.linalg.norm(bdft_ft[3:]))
        print(bdft_ft[:3], bdft_ft[3:])
        time.sleep(0.1)

    # stop streaming
    print("Stop streaming")
    d455.stop_streaming()
    listener.stop()

    # destroy devices
    del d455, bdft


if __name__ == "__main__":
    args = config_parse()
    main(args)
