import os
import time
import configargparse
import json
from pynput import keyboard
import numpy as np
import open3d as o3d

from r3kit.devices.camera.realsense.d455 import D455
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft
from r3kit.algos.tare.linear import LinearMFTarer, LinearFTarer, LinearCTTarer
from configs.pose_hz import *

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--d455_id', type=str)
    parser.add_argument('--bdft_id', type=str)
    parser.add_argument('--bdft_port', type=int)

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        # create devices
        d455 = D455(id=args.d455_id, image=False, name='d455')
        bdft = Bdft(id=args.bdft_id, port=args.bdft_port, fps=200, name='bdft')

        # special prepare for d455 to construct map
        start = input('Press enter to start D455 construct map')
        if start != '':
            del d455, bdft
            return
        print("Start D455 construct map")
        d455.collect_streaming(False)
        d455.start_streaming()

        # stable d455 pose
        stop_start = input('Press enter to stop D455 construct map and start stabling D455 pose')
        if stop_start != '':
            del d455, bdft
            return
        print("Stop D455 construct map and start stabling D455 pose")
        d455.collect_streaming(True)
        d455_pose_start_timestamp_ms = time.time() * 1000

        # mounting d455
        stop_start = input('Press enter to stop stabling D455 pose and start mounting D455')
        if stop_start != '':
            del d455, bdft
            return
        print("Stop stabling D455 pose and start mounting D455")
        d455_pose_end_timestamp_ms = time.time() * 1000

        # start streaming
        stop_start = input('Press enter to stop mounting D455 and start streaming')
        if stop_start != '':
            del d455, bdft
            return
        print("Stop mounting D455 and start streaming")
        bdft_fts, d455_poses = [], []
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
            bdft_ft = bdft.get_mean_data(n=10, name='ft')
            _, _, d455_xyz, d455_quat = d455.get()
            d455_pose = D455.raw2pose(d455_xyz, d455_quat)
            bdft_fts.append(bdft_ft)
            d455_poses.append(d455_pose)

        # stop streaming
        print("Stop streaming")
        d455_data = d455.stop_streaming()
        listener.stop()

        # destroy devices
        del d455, bdft

        # NOTE: urgly code to save data
        os.makedirs(args.save_path, exist_ok=True)
        np.save(os.path.join(args.save_path, 'd455_timestamp.npy'), np.array(d455_data['pose']["timestamp_ms"]))
        np.savetxt(os.path.join(args.save_path, 'd455_stage_timestamps.txt'), np.array([d455_pose_start_timestamp_ms, d455_pose_end_timestamp_ms]))
        np.save(os.path.join(args.save_path, 'd455_xyz.npy'), np.array(d455_data['pose']["xyz"]))
        np.save(os.path.join(args.save_path, 'd455_quat.npy'), np.array(d455_data['pose']["quat"]))
        np.save(os.path.join(args.save_path, 'd455_poses.npy'), np.array(d455_poses))
        np.save(os.path.join(args.save_path, 'bdft_fts.npy'), np.array(bdft_fts))
    else:
        # reload data
        d455_data = {
            'pose': {
                'timestamp_ms': np.load(os.path.join(args.save_path, 'd455_timestamp.npy')), 
                'xyz': np.load(os.path.join(args.save_path, 'd455_xyz.npy')), 
                'quat': np.load(os.path.join(args.save_path, 'd455_quat.npy'))
            }
        }
        d455_pose_start_timestamp_ms, d455_pose_end_timestamp_ms = np.loadtxt(os.path.join(args.save_path, 'd455_stage_timestamps.txt'))
        d455_poses = np.load(os.path.join(args.save_path, 'd455_poses.npy'))
        bdft_fts = np.load(os.path.join(args.save_path, 'bdft_fts.npy'))

        d455_data = {
            'pose': {
                'timestamp_ms': np.load(os.path.join(args.save_path, 'd455_timestamp.npy')), 
                'xyz': np.load(os.path.join(args.save_path, 'd455_xyz.npy')), 
                'quat': np.load(os.path.join(args.save_path, 'd455_quat.npy'))
            }
        }
        d455_pose_start_timestamp_ms, d455_pose_end_timestamp_ms = np.loadtxt(os.path.join(args.save_path, 'd455_stage_timestamps.txt'))
        d455_poses = np.load(os.path.join(args.save_path, 'd455_poses.npy'))
        bdft_fts = np.load(os.path.join(args.save_path, 'bdft_fts.npy'))

    # tare
    d455_all_poses = d455_data['pose']
    d455_initial_pose_mask = np.logical_and(np.array(d455_all_poses["timestamp_ms"]) > d455_pose_start_timestamp_ms, 
                                            np.array(d455_all_poses["timestamp_ms"]) < d455_pose_end_timestamp_ms)
    d455_initial_xyzs = np.array(d455_all_poses["xyz"])[d455_initial_pose_mask]
    d455_initial_quats = np.array(d455_all_poses["quat"])[d455_initial_pose_mask]
    d455_initial_xyz = np.median(d455_initial_xyzs, axis=0)
    d455_initial_quat = np.median(d455_initial_quats, axis=0)
    d455_initial_pose = D455.raw2pose(d455_initial_xyz, d455_initial_quat)  # c02w
    bdft_poses = []
    for d455_pose in d455_poses:                                            # c2w
        d455_pose = np.linalg.inv(d455_initial_pose) @ d455_pose            # c2c0 = w2c0 @ c2w
        d455_pose = D455_2_BASE @ d455_pose                                # c2b = c02b @ c2c0
        bdft_pose = d455_pose @ np.linalg.inv(D455_2_BDFT)                 # f2b = c2b @ f2c
        bdft_poses.append(bdft_pose)
    mftarer = LinearMFTarer()
    for bdft_ft, bdft_pose in zip(bdft_fts, bdft_poses):
        mftarer.add_data(bdft_ft[:3], bdft_pose[:3, :3])
    result = mftarer.run()
    ftarer = LinearFTarer()
    ftarer.set_m(result['m'])
    for bdft_ft, bdft_pose in zip(bdft_fts, bdft_poses):
        ftarer.add_data(bdft_ft[:3], bdft_pose[:3, :3])
    result.update(ftarer.run())
    ctarer = LinearCTTarer()
    ctarer.set_m(result['m'])
    for bdft_ft, bdft_pose in zip(bdft_fts, bdft_poses):
        ctarer.add_data(bdft_ft[3:], bdft_pose[:3, :3])
    result.update(ctarer.run())
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
    print(result)

    # save data
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'tare_bdft.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    # visualize
    geometries = []
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(base_frame)
    print(len(bdft_poses))
    for bdft_pose in bdft_poses[::(len(bdft_poses) // 10)]:
        bdft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
        bdft_frame.transform(bdft_pose)
        geometries.append(bdft_frame)
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    args = config_parse()
    main(args)
