import os
import configargparse
import json
import numpy as np
import cv2
import open3d as o3d
from pynput import keyboard
import tqdm

# 第三视角摄像头
from r3kit.devices.camera.realsense.d415 import D415
# 第一视角摄像头
from r3kit.devices.camera.realsense.d455 import D455
# 力矩传感器
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft
# 角度编码器
from r3kit.devices.encoder.pdcd.angler_hz import Angler
from r3kit.utils.vis import rotation_vec2mat
from configs.pose_hz import *
from utils.annotation import search_stage

'''
Synchronize with `create_hdf5.py` some part
'''


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    # general config
    data_path = args.data_path
    fps = args.fps
    frame_interval_ms = 1000. / fps

    # load stage data
    with open(os.path.join(data_path, 'stage_timestamp_ms.json'), 'r') as f:
        stage_timestamp_ms = json.load(f)
        d455_pose_start_timestamp_ms = stage_timestamp_ms['d455_pose_start_timestamp_ms']
        d455_pose_end_timestamp_ms = stage_timestamp_ms['d455_pose_end_timestamp_ms']
        start_timestamp_ms = stage_timestamp_ms['start_timestamp_ms']
    # load d415 data
    d415_path = os.path.join(data_path, 'd415')
    d415_intrinsics = np.loadtxt(os.path.join(d415_path, 'intrinsics.txt'))     # (4,), float64
    d415_depth_scale = np.loadtxt(os.path.join(d415_path, 'depth_scale.txt')).item()
    d415_timestamps = np.load(os.path.join(d415_path, 'timestamps.npy'))
    ### d415_depth_img, d415_color_img loaded during iteration
    # load d455 data
    d455_path = os.path.join(data_path, 'd455')
    ### d455_image_path = os.path.join(d455_path, 'image')
    ### d455_image_timestamps = np.load(os.path.join(d455_image_path, 'timestamps.npy'))
    ### d455_left_img, d455_right_img loaded during iteration
    d455_pose_path = os.path.join(d455_path, 'pose')
    d455_pose_timestamps = np.load(os.path.join(d455_pose_path, 'timestamps.npy'))
    d455_xyzs = np.load(os.path.join(d455_pose_path, 'xyz.npy'))
    d455_quats = np.load(os.path.join(d455_pose_path, 'quat.npy'))
    # load bdft data
    bdft_path = os.path.join(data_path, 'bdft')
    with open(os.path.join(bdft_path, 'tare_bdft.json'), 'r') as f:
        bdft_tare = json.load(f)
    bdft_timestamps = np.load(os.path.join(bdft_path, 'timestamps.npy'))
    bdft_fts = np.load(os.path.join(bdft_path, 'ft.npy'))
    # load angler data
    angler_path = os.path.join(data_path, 'angler')
    angler_timestamps = np.load(os.path.join(angler_path, 'timestamps.npy'))
    angler_angles = np.load(os.path.join(angler_path, 'angle.npy'))
    # load annotation data
    annotation_path = os.path.join(data_path, 'annotation.json')
    has_annotation = os.path.exists(annotation_path)
    if has_annotation:
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

    # deal with d455 special prepare
    d455_initial_pose_mask = np.logical_and(d455_pose_timestamps > d455_pose_start_timestamp_ms, d455_pose_timestamps < d455_pose_end_timestamp_ms)
    d455_initial_xyz = np.median(d455_xyzs[d455_initial_pose_mask, :], axis=0)
    d455_initial_quat = np.median(d455_quats[d455_initial_pose_mask, :], axis=0)
    d455_initial_pose = D455.raw2pose(d455_initial_xyz, d455_initial_quat)       # c02w

    # deal with angler special prepare
    angler_angles = Angler.raw2angle(angler_angles)
    angler_angles[angler_angles < 0] = 0.0
    angler_widths = angler_angles * ANGLE_2_WIDTH

    # process d415 variables
    d415_current_idx = 0
    d415_current_timestamp = d415_timestamps[d415_current_idx]
    d415_start_timestamp = d415_timestamps[0]
    d415_end_timestamp = d415_timestamps[-1]
    # process d455 variables
    ### d455_image_current_idx = np.searchsorted(d455_image_timestamps, d415_current_timestamp)
    d455_pose_current_idx = np.searchsorted(d455_pose_timestamps, d415_current_timestamp)
    # process bdft variables
    bdft_current_idx = np.searchsorted(bdft_timestamps, d415_current_timestamp)
    # process angler variables
    angler_current_idx = np.searchsorted(angler_timestamps, d415_current_timestamp)

    # create keyboard listener
    quit = False
    reset = False
    pause = False
    zero = False
    forward = False
    backward = False
    speed = 1
    if not has_annotation:
        minus = False
        d455_xyz_d455w_bias = np.array([0., 0., 0.])
        stages = [{'timestamp_ms': d415_current_timestamp, 
                   'd455_xyz_d455w_bias': d455_xyz_d455w_bias.tolist(), 
                   'stage': 'unrelated'}]
    else:
        stages = annotation
        stage_idx = search_stage(d415_current_timestamp, stages)
        stage = stages[stage_idx]
        d455_xyz_d455w_bias = np.array(stage['d455_xyz_d455w_bias'])
    def _on_press(key):
        nonlocal quit, reset, pause, zero, forward, backward, speed
        nonlocal current_timestamp, stages, minus, d455_xyz_d455w_bias
        if hasattr(key, 'char') and key.char == 'q':
            quit = True
            print("quit")
        if hasattr(key, 'char') and key.char == 'r':
            reset = True
            print("reset")
        if hasattr(key, 'char') and key.char == 'p':
            pause = not pause
            forward = False
            backward = False
            print("pause" if pause else "continue")
        if key == keyboard.Key.backspace:
            zero = True
            print("zero")
        if pause and key == keyboard.Key.right:
            forward = True
            print("forward")
        if pause and key == keyboard.Key.left:
            backward = True
            print("backward")
        if pause and key == keyboard.Key.up:
            speed *= 2
            print(f"speed {speed}")
        if pause and key == keyboard.Key.down:
            speed //= 2
            speed = max(speed, 1)
            print(f"speed {speed}")
        if not has_annotation:
            if hasattr(key, 'char') and key.char == 'u':
                stages.append({'timestamp_ms': current_timestamp, 
                                'd455_xyz_d455w_bias': d455_xyz_d455w_bias.tolist(), 
                                'stage': 'unrelated'})
                print(f"unrelated from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 'g':
                stages.append({'timestamp_ms': current_timestamp, 
                                'd455_xyz_d455w_bias': d455_xyz_d455w_bias.tolist(), 
                                'stage': 'grasp'})
                print(f"grasp from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 's':
                stages.append({'timestamp_ms': current_timestamp, 
                                'd455_xyz_d455w_bias': d455_xyz_d455w_bias.tolist(), 
                                'stage': 'shave'})
                print(f"shave from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 't':
                stages.append({'timestamp_ms': current_timestamp, 
                                'd455_xyz_d455w_bias': d455_xyz_d455w_bias.tolist(), 
                                'stage': 'turn'})
                print(f"turn from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 'm':
                minus = not minus
                print("bias minus" if minus else "bias plus")
            if hasattr(key, 'char') and key.char == 'x':
                d455_xyz_d455w_bias = d455_xyz_d455w_bias + np.array([0.005 if not minus else -0.005, 0., 0.])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['d455_xyz_d455w_bias'] = d455_xyz_d455w_bias.tolist()
            if hasattr(key, 'char') and key.char == 'y':
                d455_xyz_d455w_bias = d455_xyz_d455w_bias + np.array([0., 0.005 if not minus else -0.005, 0.])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['d455_xyz_d455w_bias'] = d455_xyz_d455w_bias.tolist()
            if hasattr(key, 'char') and key.char == 'z':
                d455_xyz_d455w_bias = d455_xyz_d455w_bias + np.array([0., 0., 0.005 if not minus else -0.005])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['d455_xyz_d455w_bias'] = d455_xyz_d455w_bias.tolist()
            if key == keyboard.Key.delete:
                stages.pop()
                print("delete")
    def _on_release(key):
        pass
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # create visualizer
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1280, height=720, left=200, top=200, visible=True, window_name='data')

    # add d415 elements
    d415_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    visualizer.add_geometry(d415_frame)
    d415_color_img = cv2.imread(os.path.join(d415_path, 'color', f'{str(d415_current_idx).zfill(16)}.png'), cv2.IMREAD_COLOR)
    d415_color_img = cv2.cvtColor(d415_color_img, cv2.COLOR_BGR2RGB)                    # (H, W, 3), uint8
    d415_color_img = d415_color_img / 255.                                              # (H, W, 3), float64
    d415_depth_img = cv2.imread(os.path.join(d415_path, 'depth', f'{str(d415_current_idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH)    # (H, W), uint16
    d415_depth_img = d415_depth_img * d415_depth_scale                                  # (H, W), float64
    d415_pc_xyz_d415, d415_pc_rgb = D415.img2pc(d415_depth_img, d415_intrinsics, d415_color_img)
    d415_pcd = o3d.geometry.PointCloud()
    d415_pcd.points = o3d.utility.Vector3dVector(d415_pc_xyz_d415)
    d415_pcd.colors = o3d.utility.Vector3dVector(d415_pc_rgb)
    visualizer.add_geometry(d415_pcd)
    # add d455 elements
    d455_xyz_d455w, d455_quat_d455w = d455_xyzs[d455_pose_current_idx], d455_quats[d455_pose_current_idx]
    d455_xyz_d455w = d455_xyz_d455w + d455_xyz_d455w_bias
    d455_pose_d455w = D455.raw2pose(d455_xyz_d455w, d455_quat_d455w)              # c2w
    d455_pose_d4550 = np.linalg.inv(d455_initial_pose) @ d455_pose_d455w           # c2c0 = w2c0 @ c2w
    d455_pose_d415 = np.linalg.inv(D415_2_D455) @ d455_pose_d4550                   # c2l = c02l @ c2c0
    ### d455_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    ### d455_frame.transform(d455_pose_d415)
    ### visualizer.add_geometry(d455_frame)
    ### d455_left_img = cv2.imread(os.path.join(d455_image_path, 'left', f'{str(d455_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)      # (H, W), uint8
    ### cv2.namedWindow('d455_left', cv2.WINDOW_NORMAL)
    ### cv2.imshow('d455_left', d455_left_img)
    ### cv2.waitKey(1)
    ### d455_right_img = cv2.imread(os.path.join(d455_image_path, 'right', f'{str(d455_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)    # (H, W), uint8
    ### cv2.namedWindow('d455_right', cv2.WINDOW_NORMAL)
    ### cv2.imshow('d455_right', d455_right_img)
    ### cv2.waitKey(1)
    gripper_pose_d415 = d455_pose_d415 @ np.linalg.inv(D455_2_GRIPPER) 
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    gripper_frame.transform(gripper_pose_d415)
    visualizer.add_geometry(gripper_frame)
    gripper = o3d.io.read_triangle_mesh(os.path.join("objs", "gripper.obj"))
    gripper.transform(gripper_pose_d415)
    visualizer.add_geometry(gripper)
    # add bdft elements
    bdft_pose_d415 = d455_pose_d415 @ np.linalg.inv(D455_2_BDFT)                      # f2l = c2l @ f2c
    bdft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    bdft_frame.transform(bdft_pose_d415)
    visualizer.add_geometry(bdft_frame)
    bdft_ft_bdft = bdft_fts[bdft_current_idx]
    bdft_pose_base = D415_2_BASE @ bdft_pose_d415                                       # f2b = l2b @ f2l
    bdft_ft_bdft = Bdft.raw2tare(bdft_ft_bdft, bdft_tare, bdft_pose_base[:3, :3])
    bdft_f_bdft, bdft_t_bdft = bdft_ft_bdft[:3], bdft_ft_bdft[3:]
    bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
    bdft_f_value = np.linalg.norm(bdft_f_d415)
    bdft_f_rotation_d415 = rotation_vec2mat(bdft_f_d415 / bdft_f_value)
    bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
    bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
    bdft_t_value = np.linalg.norm(bdft_t_d415)
    bdft_t_rotation_d415 = rotation_vec2mat(bdft_t_d415 / bdft_t_value)
    bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
    bdft_f_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.04 * 0.025, cone_radius=0.04 * 0.05, cylinder_height=0.04 * 0.875, cone_height=0.04 * 0.125, 
                                                            resolution=20, cylinder_split=4, cone_split=1)
    bdft_f_arrow.paint_uniform_color([1., 1., 0.])
    bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
    bdft_f_arrow.rotate(bdft_f_rotation_d415, np.array([[0], [0], [0]]))
    bdft_f_arrow.translate(bdft_f_translation_d415)
    visualizer.add_geometry(bdft_f_arrow)
    bdft_t_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.4 * 0.025, cone_radius=0.4 * 0.05, cylinder_height=0.4 * 0.875, cone_height=0.4 * 0.125, 
                                                            resolution=20, cylinder_split=4, cone_split=1)
    bdft_t_arrow.paint_uniform_color([0., 1., 1.])
    bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
    bdft_t_arrow.rotate(bdft_t_rotation_d415, np.array([[0], [0], [0]]))
    bdft_t_arrow.translate(bdft_t_translation_d415)
    visualizer.add_geometry(bdft_t_arrow)
    bdft_peeler = o3d.io.read_triangle_mesh(os.path.join("objs", "peeler.obj"))
    bdft_peeler.transform(bdft_pose_d415)
    visualizer.add_geometry(bdft_peeler)
    # add angler elements
    angler_width = angler_widths[angler_current_idx]
    angler_right_finger = o3d.io.read_triangle_mesh(os.path.join("objs", "right_finger.obj"))
    angler_left_finger = o3d.io.read_triangle_mesh(os.path.join("objs", "left_finger.obj"))
    angler_finger_pose_gripper = np.identity(4)
    angler_finger_pose_gripper[0, 3] = angler_width / 2.
    gripper_right_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
    angler_right_finger.transform(gripper_right_finger_pose_d415)
    visualizer.add_geometry(angler_right_finger)
    angler_finger_pose_gripper[0, 3] = -angler_width / 2.
    gripper_left_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
    angler_left_finger.transform(gripper_left_finger_pose_d415)
    visualizer.add_geometry(angler_left_finger)

    # visualizer setup
    view_control = visualizer.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # visualize loop
    show_timestamps = np.arange(d415_start_timestamp+frame_interval_ms, d415_end_timestamp+1e-3, frame_interval_ms)
    with tqdm.tqdm(total=len(show_timestamps)) as pbar:
        show_idx = 0
        while show_idx < len(show_timestamps):
            current_timestamp = show_timestamps[show_idx]
            print(current_timestamp)
            stage_idx = search_stage(current_timestamp, stages)
            stage = stages[stage_idx]
            d455_xyz_d455w_bias = np.array(stage['d455_xyz_d455w_bias'])

            # update d415 variables
            d415_current_idx = np.searchsorted(d415_timestamps, current_timestamp)
            d415_current_idx = min(d415_current_idx, len(d415_timestamps)-1)
            d415_current_time = (d415_timestamps[d415_current_idx] - d415_start_timestamp) / 1000.
            # update d455 variables
            ### d455_image_current_idx = np.searchsorted(d455_image_timestamps, current_timestamp)
            ### d455_image_current_idx = min(d455_image_current_idx, len(d455_image_timestamps)-1)
            ### d455_image_current_time = (d455_image_timestamps[d455_image_current_idx] - d415_start_timestamp) / 1000.
            d455_pose_current_idx = np.searchsorted(d455_pose_timestamps, current_timestamp)
            d455_pose_current_idx = min(d455_pose_current_idx, len(d455_pose_timestamps)-1)
            d455_pose_current_time = (d455_pose_timestamps[d455_pose_current_idx] - d415_start_timestamp) / 1000.

            # update bdft variables
            bdft_current_idx = np.searchsorted(bdft_timestamps, current_timestamp)
            bdft_current_idx = min(bdft_current_idx, len(bdft_timestamps)-1)
            bdft_current_time = (bdft_timestamps[bdft_current_idx] - d415_start_timestamp) / 1000.
            # update angler variables
            angler_current_idx = np.searchsorted(angler_timestamps, current_timestamp)
            angler_current_idx = min(angler_current_idx, len(angler_timestamps)-1)
            angler_current_time = (angler_timestamps[angler_current_idx] - d415_start_timestamp) / 1000.
            
            # update d415 elements
            d415_color_img = cv2.imread(os.path.join(d415_path, 'color', f'{str(d415_current_idx).zfill(16)}.png'), cv2.IMREAD_COLOR)
            d415_color_img = cv2.cvtColor(d415_color_img, cv2.COLOR_BGR2RGB)            # (H, W, 3), uint8
            d415_color_img = d415_color_img / 255.                                      # (H, W, 3), float64
            d415_depth_img = cv2.imread(os.path.join(d415_path, 'depth', f'{str(d415_current_idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH)    # (H, W), uint16
            d415_depth_img = d415_depth_img * d415_depth_scale                                     # (H, W), float64
            d415_pc_xyz_d415, d415_pc_rgb = D415.img2pc(d415_depth_img, d415_intrinsics, d415_color_img)
            d415_pcd.points = o3d.utility.Vector3dVector(d415_pc_xyz_d415)
            d415_pcd.colors = o3d.utility.Vector3dVector(d415_pc_rgb)
            visualizer.update_geometry(d415_pcd)
            # update d455 elements
            d455_xyz_d455w, d455_quat_d455w = d455_xyzs[d455_pose_current_idx], d455_quats[d455_pose_current_idx]
            d455_xyz_d455w = d455_xyz_d455w + d455_xyz_d455w_bias
            d455_pose_d455w = D455.raw2pose(d455_xyz_d455w, d455_quat_d455w)              # c2w
            d455_pose_d4550 = np.linalg.inv(d455_initial_pose) @ d455_pose_d455w           # c2c0 = w2c0 @ c2w
            d455_pose_d415_last = d455_pose_d415.copy()
            d455_pose_d415 = np.linalg.inv(D415_2_D455) @ d455_pose_d4550                   # c2l = c02l @ c2c0
            ### d455_frame.transform(np.linalg.inv(d455_pose_d415_last))
            ### d455_frame.transform(d455_pose_d415)
            ### visualizer.update_geometry(d455_frame)
            ### d455_left_img = cv2.imread(os.path.join(d455_image_path, 'left', f'{str(d455_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)      # (H, W), uint8
            ### cv2.imshow('d455_left', d455_left_img)
            ### cv2.waitKey(1)
            ### d455_right_img = cv2.imread(os.path.join(d455_image_path, 'right', f'{str(d455_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)    # (H, W), uint8
            ### cv2.imshow('d455_right', d455_right_img)
            ### cv2.waitKey(1)
            
            gripper_pose_d415_last = gripper_pose_d415.copy()
          
            gripper_frame.transform(np.linalg.inv(gripper_pose_d415_last))
            gripper_frame.transform(gripper_pose_d415)
            visualizer.update_geometry(gripper_frame)
            gripper.transform(np.linalg.inv(gripper_pose_d415_last))
            gripper.transform(gripper_pose_d415)
            visualizer.update_geometry(gripper)
            # update bdft elements
            bdft_pose_d415_last = bdft_pose_d415.copy()
            bdft_pose_d415 = d455_pose_d415 @ np.linalg.inv(D455_2_BDFT)
            bdft_frame.transform(np.linalg.inv(bdft_pose_d415_last))
            bdft_frame.transform(bdft_pose_d415)
            visualizer.update_geometry(bdft_frame)
            bdft_ft_bdft = bdft_fts[bdft_current_idx]
            ### print(bdft_ft_bdft)
            bdft_pose_base = D415_2_BASE @ bdft_pose_d415                                       # f2b = l2b @ f2l
            bdft_ft_bdft = Bdft.raw2tare(bdft_ft_bdft, bdft_tare, bdft_pose_base[:3, :3])
            bdft_f_bdft, bdft_t_bdft = bdft_ft_bdft[:3], bdft_ft_bdft[3:]
            bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
            bdft_f_value_last = bdft_f_value.copy()
            bdft_f_value = np.linalg.norm(bdft_f_d415)
            bdft_f_rotation_d415_last = bdft_f_rotation_d415.copy()
            bdft_f_rotation_d415 = rotation_vec2mat(bdft_f_d415 / bdft_f_value)
            bdft_f_translation_d415_last = bdft_f_translation_d415.copy()
            bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
            bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
            bdft_t_value_last = bdft_t_value.copy()
            bdft_t_value = np.linalg.norm(bdft_t_d415)
            bdft_t_rotation_d415_last = bdft_t_rotation_d415.copy()
            bdft_t_rotation_d415 = rotation_vec2mat(bdft_t_d415 / bdft_t_value)
            bdft_t_translation_d415_last = bdft_t_translation_d415.copy()
            bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
            bdft_f_arrow.translate(-bdft_f_translation_d415_last)
            bdft_f_arrow.rotate(np.linalg.inv(bdft_f_rotation_d415_last), np.array([[0], [0], [0]]))
            bdft_f_arrow.scale(1/bdft_f_value_last, np.array([[0], [0], [0]]))
            bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
            bdft_f_arrow.rotate(bdft_f_rotation_d415, np.array([[0], [0], [0]]))
            bdft_f_arrow.translate(bdft_f_translation_d415)
            visualizer.update_geometry(bdft_f_arrow)
            bdft_t_arrow.translate(-bdft_t_translation_d415_last)
            bdft_t_arrow.rotate(np.linalg.inv(bdft_t_rotation_d415_last), np.array([[0], [0], [0]]))
            bdft_t_arrow.scale(1/bdft_t_value_last, np.array([[0], [0], [0]]))
            bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
            bdft_t_arrow.rotate(bdft_t_rotation_d415, np.array([[0], [0], [0]]))
            bdft_t_arrow.translate(bdft_t_translation_d415)
            visualizer.update_geometry(bdft_t_arrow)
            bdft_peeler.transform(np.linalg.inv(bdft_pose_d415_last))
            bdft_peeler.transform(bdft_pose_d415)
            visualizer.update_geometry(bdft_peeler)
            # update angler elements
            angler_width_last = angler_width.copy()
            angler_width = angler_widths[angler_current_idx]
            angler_finger_pose_gripper = np.identity(4)
            angler_finger_pose_gripper[0, 3] = angler_width / 2.
            gripper_right_finger_pose_d415_last = gripper_right_finger_pose_d415.copy()
            gripper_right_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
            angler_right_finger.transform(np.linalg.inv(gripper_right_finger_pose_d415_last))
            angler_right_finger.transform(gripper_right_finger_pose_d415)
            visualizer.update_geometry(angler_right_finger)
            angler_finger_pose_gripper[0, 3] = -angler_width / 2.
            gripper_left_finger_pose_d415_last = gripper_left_finger_pose_d415.copy()
            gripper_left_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
            angler_left_finger.transform(np.linalg.inv(gripper_left_finger_pose_d415_last))
            angler_left_finger.transform(gripper_left_finger_pose_d415)
            visualizer.update_geometry(angler_left_finger)

            # visualizer update
            visualizer.poll_events()
            visualizer.update_renderer()

            # pbar update
            if not pause:
                show_idx += 1
                pbar.update(1)
            else:
                if forward:
                    show_idx += speed
                    pbar.update(speed)
                    forward = False
                elif backward:
                    show_idx -= speed
                    pbar.update(-speed)
                    backward = False
                else:
                    pass
            pbar.set_postfix(f=bdft_f_value, t=bdft_t_value, s=stage['stage'])

            # keyboard quit
            if quit:
                break
            # keyboard reset
            if reset:
                view_control = visualizer.get_view_control()
                params = view_control.convert_to_pinhole_camera_parameters()
                params.extrinsic = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                reset = False
            # keyboard zero
            if zero:
                pbar.update(-show_idx)
                show_idx = 0
                zero = False

    visualizer.destroy_window()
    listener.stop()

    if not has_annotation:
        print(stages)
        with open(annotation_path, 'w') as f:
            json.dump(stages, f, indent=4)


if __name__ == '__main__':
    args = config_parse()
    main(args)
