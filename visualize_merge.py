import os
import configargparse
import numpy as np
import open3d as o3d
from pynput import keyboard
import tqdm
import h5py

from r3kit.utils.vis import rotation_vec2mat
from utils.transformation import xyzquat2mat

"""
Synchronize with `visualize_hdf5.py` some part
"""


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--hdf5_path", type=str)
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    # general config
    hdf5_path = args.hdf5_path
    is_sample = args.sample

    # load hdf5
    with h5py.File(hdf5_path, "r") as data_hdf5:
        data_group = data_hdf5["data"]
        data_attributes = dict(data_group.attrs)
        print(data_attributes)
        ft_coord = data_attributes["ft_coord"]

        for demo_idx, demo_name in enumerate(tqdm.tqdm(sorted(data_group.keys()))):
            print(demo_name)
            demo_group = data_group[demo_name]
            demo_attributes = dict(demo_group.attrs)
            print(demo_attributes)
            num_samples = demo_attributes["num_samples"]

            d415_pc_xyzs_d415 = demo_group["d415_pc_xyzs_d415"][:].astype(np.float32)
            d415_pc_rgbs = demo_group["d415_pc_rgbs"][:].astype(np.float32)
            gripper_xyzs_d415 = demo_group["gripper_xyzs_d415"][:].astype(np.float32)
            gripper_quats_d415 = demo_group["gripper_quats_d415"][:].astype(np.float32)
            bdft_xyzs_d415 = demo_group["bdft_xyzs_d415"][:].astype(np.float32)
            bdft_quats_d415 = demo_group["bdft_quats_d415"][:].astype(np.float32)
            bdft_fs = demo_group["bdft_fs"][:].astype(np.float32)
            bdft_ts = demo_group["bdft_ts"][:].astype(np.float32)
            angler_widths = demo_group["angler_widths"][:].astype(np.float32)
            len_seq = d415_pc_xyzs_d415.shape[0]
            print(len_seq)
            o_idxs = demo_group["o"][:].astype(int)
            a_idxs = demo_group["a"][:].astype(int)
            print(o_idxs[[0, num_samples // 2, -1]])
            print(a_idxs[[0, num_samples // 2, -1]])

            if is_sample:
                # create visualizer
                if demo_idx == 0:
                    visualizer = o3d.visualization.Visualizer()
                    visualizer.create_window(
                        width=1280,
                        height=720,
                        left=200,
                        top=200,
                        visible=True,
                        window_name="data",
                    )

                # loop samples
                for sample_idx in tqdm.trange(num_samples):
                    o_idx = o_idxs[sample_idx]
                    a_idx = a_idxs[sample_idx]

                    for current_idx in o_idx:
                        # add d415 elements
                        d415_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.05, origin=np.array([0.0, 0.0, 0.0])
                        )
                        visualizer.add_geometry(d415_frame)
                        d415_pc_xyz_d415, d415_pc_rgb = (
                            d415_pc_xyzs_d415[current_idx],
                            d415_pc_rgbs[current_idx],
                        )
                        d415_pcd = o3d.geometry.PointCloud()
                        d415_pcd.points = o3d.utility.Vector3dVector(d415_pc_xyz_d415)
                        d415_pcd.colors = o3d.utility.Vector3dVector(d415_pc_rgb)
                        visualizer.add_geometry(d415_pcd)
                        # add gripper elements
                        gripper_xyz_d415, gripper_quat_d415 = (
                            gripper_xyzs_d415[current_idx],
                            gripper_quats_d415[current_idx],
                        )
                        gripper_pose_d415 = xyzquat2mat(
                            gripper_xyz_d415, gripper_quat_d415
                        )
                        gripper_frame = (
                            o3d.geometry.TriangleMesh.create_coordinate_frame(
                                size=0.05, origin=np.array([0.0, 0.0, 0.0])
                            )
                        )
                        gripper_frame.transform(gripper_pose_d415)
                        visualizer.add_geometry(gripper_frame)
                        gripper = o3d.io.read_triangle_mesh(
                            os.path.join("objs", "gripper.obj")
                        )
                        gripper.transform(gripper_pose_d415)
                        visualizer.add_geometry(gripper)
                        # add bdft elements
                        bdft_xyz_d415, bdft_quat_d415 = (
                            bdft_xyzs_d415[current_idx],
                            bdft_quats_d415[current_idx],
                        )
                        bdft_pose_d415 = xyzquat2mat(bdft_xyz_d415, bdft_quat_d415)
                        bdft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.05, origin=np.array([0.0, 0.0, 0.0])
                        )
                        bdft_frame.transform(bdft_pose_d415)
                        visualizer.add_geometry(bdft_frame)
                        if ft_coord:
                            bdft_f_d415, bdft_t_d415 = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                        else:
                            bdft_f_bdft, bdft_t_bdft = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                            bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
                            bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
                        bdft_f_value = np.linalg.norm(bdft_f_d415)
                        bdft_f_rotation_d415 = rotation_vec2mat(
                            bdft_f_d415 / bdft_f_value
                        )
                        bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_t_value = np.linalg.norm(bdft_t_d415)
                        bdft_t_rotation_d415 = rotation_vec2mat(
                            bdft_t_d415 / bdft_t_value
                        )
                        bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_f_arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.04 * 0.025,
                            cone_radius=0.04 * 0.05,
                            cylinder_height=0.04 * 0.875,
                            cone_height=0.04 * 0.125,
                            resolution=20,
                            cylinder_split=4,
                            cone_split=1,
                        )
                        bdft_f_arrow.paint_uniform_color([1.0, 1.0, 0.0])
                        bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
                        bdft_f_arrow.rotate(
                            bdft_f_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_f_arrow.translate(bdft_f_translation_d415)
                        visualizer.add_geometry(bdft_f_arrow)
                        bdft_t_arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.4 * 0.025,
                            cone_radius=0.4 * 0.05,
                            cylinder_height=0.4 * 0.875,
                            cone_height=0.4 * 0.125,
                            resolution=20,
                            cylinder_split=4,
                            cone_split=1,
                        )
                        bdft_t_arrow.paint_uniform_color([0.0, 1.0, 1.0])
                        bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
                        bdft_t_arrow.rotate(
                            bdft_t_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_t_arrow.translate(bdft_t_translation_d415)
                        visualizer.add_geometry(bdft_t_arrow)
                        bdft_peeler = o3d.io.read_triangle_mesh(
                            os.path.join("objs", "peeler.obj")
                        )
                        bdft_peeler.transform(bdft_pose_d415)
                        visualizer.add_geometry(bdft_peeler)
                        # add angler elements
                        angler_width = angler_widths[current_idx]
                        angler_right_finger = o3d.io.read_triangle_mesh(
                            os.path.join("objs", "right_finger.obj")
                        )
                        angler_left_finger = o3d.io.read_triangle_mesh(
                            os.path.join("objs", "left_finger.obj")
                        )
                        angler_finger_pose_gripper = np.identity(4)
                        angler_finger_pose_gripper[0, 3] = angler_width / 2.0
                        gripper_right_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_right_finger.transform(gripper_right_finger_pose_d415)
                        visualizer.add_geometry(angler_right_finger)
                        angler_finger_pose_gripper[0, 3] = -angler_width / 2.0
                        gripper_left_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_left_finger.transform(gripper_left_finger_pose_d415)
                        visualizer.add_geometry(angler_left_finger)

                    # visualizer setup
                    view_control = visualizer.get_view_control()
                    visualizer.get_render_option().background_color = [0, 0, 0]
                    params = view_control.convert_to_pinhole_camera_parameters()
                    params.extrinsic = np.array(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    )
                    view_control.convert_from_pinhole_camera_parameters(
                        params, allow_arbitrary=True
                    )

                    # visualize loop
                    for current_idx in a_idx:
                        # update gripper elements
                        gripper_xyz_d415, gripper_quat_d415 = (
                            gripper_xyzs_d415[current_idx],
                            gripper_quats_d415[current_idx],
                        )
                        gripper_pose_d415_last = gripper_pose_d415.copy()
                        gripper_pose_d415 = xyzquat2mat(
                            gripper_xyz_d415, gripper_quat_d415
                        )
                        gripper_frame.transform(np.linalg.inv(gripper_pose_d415_last))
                        gripper_frame.transform(gripper_pose_d415)
                        visualizer.update_geometry(gripper_frame)
                        gripper.transform(np.linalg.inv(gripper_pose_d415_last))
                        gripper.transform(gripper_pose_d415)
                        gripper_delta_pose = np.dot(
                            np.linalg.inv(gripper_pose_d415_last), gripper_pose_d415
                        )
                        visualizer.update_geometry(gripper)
                        # update bdft elements
                        bdft_xyz_d415, bdft_quat_d415 = (
                            bdft_xyzs_d415[current_idx],
                            bdft_quats_d415[current_idx],
                        )
                        bdft_pose_d415_last = bdft_pose_d415.copy()
                        bdft_pose_d415 = xyzquat2mat(bdft_xyz_d415, bdft_quat_d415)
                        bdft_frame.transform(np.linalg.inv(bdft_pose_d415_last))
                        bdft_frame.transform(bdft_pose_d415)
                        bdft_delta_pose = np.dot(
                            np.linalg.inv(bdft_pose_d415_last), bdft_pose_d415
                        )
                        visualizer.update_geometry(bdft_frame)
                        if ft_coord:
                            bdft_f_d415_last, bdft_t_d415_last = (
                                bdft_f_d415.copy(),
                                bdft_t_d415.copy(),
                            )
                            bdft_f_d415, bdft_t_d415 = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                            bdft_delta_f, bdft_delta_t = (
                                bdft_f_d415 - bdft_f_d415_last,
                                bdft_t_d415 - bdft_t_d415_last,
                            )
                        else:
                            bdft_f_bdft_last, bdft_t_bdft_last = (
                                bdft_f_bdft.copy(),
                                bdft_t_bdft.copy(),
                            )
                            bdft_f_bdft, bdft_t_bdft = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                            bdft_delta_f, bdft_delta_t = (
                                bdft_f_bdft - bdft_f_bdft_last,
                                bdft_t_bdft - bdft_t_bdft_last,
                            )
                            bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
                            bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
                        bdft_f_value_last = bdft_f_value.copy()
                        bdft_f_value = np.linalg.norm(bdft_f_d415)
                        bdft_f_rotation_d415_last = bdft_f_rotation_d415.copy()
                        bdft_f_rotation_d415 = rotation_vec2mat(
                            bdft_f_d415 / bdft_f_value
                        )
                        bdft_f_translation_d415_last = bdft_f_translation_d415.copy()
                        bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_t_value_last = bdft_t_value.copy()
                        bdft_t_value = np.linalg.norm(bdft_t_d415)
                        bdft_t_rotation_d415_last = bdft_t_rotation_d415.copy()
                        bdft_t_rotation_d415 = rotation_vec2mat(
                            bdft_t_d415 / bdft_t_value
                        )
                        bdft_t_translation_d415_last = bdft_t_translation_d415.copy()
                        bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_f_arrow.translate(-bdft_f_translation_d415_last)
                        bdft_f_arrow.rotate(
                            np.linalg.inv(bdft_f_rotation_d415_last),
                            np.array([[0], [0], [0]]),
                        )
                        bdft_f_arrow.scale(
                            1 / bdft_f_value_last, np.array([[0], [0], [0]])
                        )
                        bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
                        bdft_f_arrow.rotate(
                            bdft_f_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_f_arrow.translate(bdft_f_translation_d415)
                        visualizer.update_geometry(bdft_f_arrow)
                        bdft_t_arrow.translate(-bdft_t_translation_d415_last)
                        bdft_t_arrow.rotate(
                            np.linalg.inv(bdft_t_rotation_d415_last),
                            np.array([[0], [0], [0]]),
                        )
                        bdft_t_arrow.scale(
                            1 / bdft_t_value_last, np.array([[0], [0], [0]])
                        )
                        bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
                        bdft_t_arrow.rotate(
                            bdft_t_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_t_arrow.translate(bdft_t_translation_d415)
                        visualizer.update_geometry(bdft_t_arrow)
                        bdft_peeler.transform(np.linalg.inv(bdft_pose_d415_last))
                        bdft_peeler.transform(bdft_pose_d415)
                        visualizer.update_geometry(bdft_peeler)
                        # update angler elements
                        angler_width_last = angler_width.copy()
                        angler_width = angler_widths[current_idx]
                        angler_finger_pose_gripper = np.identity(4)
                        angler_finger_pose_gripper[0, 3] = angler_width / 2.0
                        gripper_right_finger_pose_d415_last = (
                            gripper_right_finger_pose_d415.copy()
                        )
                        gripper_right_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_right_finger.transform(
                            np.linalg.inv(gripper_right_finger_pose_d415_last)
                        )
                        angler_right_finger.transform(gripper_right_finger_pose_d415)
                        visualizer.update_geometry(angler_right_finger)
                        angler_finger_pose_gripper[0, 3] = -angler_width / 2.0
                        gripper_left_finger_pose_d415_last = (
                            gripper_left_finger_pose_d415.copy()
                        )
                        gripper_left_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_left_finger.transform(
                            np.linalg.inv(gripper_left_finger_pose_d415_last)
                        )
                        angler_left_finger.transform(gripper_left_finger_pose_d415)
                        visualizer.update_geometry(angler_left_finger)

                        # visualizer update
                        visualizer.poll_events()
                        visualizer.update_renderer()

                    visualizer.clear_geometries()
            else:
                # create keyboard listener
                quit = False
                reset = False
                pause = False
                zero = False
                forward = False
                backward = False
                speed = 1

                def _on_press(key):
                    nonlocal quit, reset, pause, zero, forward, backward, speed
                    if hasattr(key, "char") and key.char == "q":
                        quit = True
                        print("quit")
                    if hasattr(key, "char") and key.char == "r":
                        reset = True
                        print("reset")
                    if hasattr(key, "char") and key.char == "p":
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
                    if key == keyboard.Key.up:
                        speed *= 2
                        print(f"speed {speed}")
                    if key == keyboard.Key.down:
                        speed //= 2
                        speed = max(speed, 1)
                        print(f"speed {speed}")

                def _on_release(key):
                    pass

                listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
                listener.start()

                # process variables
                current_idx = 0

                # create visualizer
                if demo_idx == 0:
                    visualizer = o3d.visualization.Visualizer()
                    visualizer.create_window(
                        width=1280,
                        height=720,
                        left=200,
                        top=200,
                        visible=True,
                        window_name="data",
                    )

                # add d415 elements
                d415_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.05, origin=np.array([0.0, 0.0, 0.0])
                )
                visualizer.add_geometry(d415_frame)
                d415_pc_xyz_d415, d415_pc_rgb = (
                    d415_pc_xyzs_d415[current_idx],
                    d415_pc_rgbs[current_idx],
                )
                d415_pcd = o3d.geometry.PointCloud()
                d415_pcd.points = o3d.utility.Vector3dVector(d415_pc_xyz_d415)
                d415_pcd.colors = o3d.utility.Vector3dVector(d415_pc_rgb)
                visualizer.add_geometry(d415_pcd)
                # add gripper elements
                gripper_xyz_d415, gripper_quat_d415 = (
                    gripper_xyzs_d415[current_idx],
                    gripper_quats_d415[current_idx],
                )
                gripper_pose_d415 = xyzquat2mat(gripper_xyz_d415, gripper_quat_d415)
                gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.05, origin=np.array([0.0, 0.0, 0.0])
                )
                gripper_frame.transform(gripper_pose_d415)
                visualizer.add_geometry(gripper_frame)
                gripper = o3d.io.read_triangle_mesh(os.path.join("objs", "gripper.obj"))
                gripper.transform(gripper_pose_d415)
                visualizer.add_geometry(gripper)
                # add bdft elements
                bdft_xyz_d415, bdft_quat_d415 = (
                    bdft_xyzs_d415[current_idx],
                    bdft_quats_d415[current_idx],
                )
                bdft_pose_d415 = xyzquat2mat(bdft_xyz_d415, bdft_quat_d415)
                bdft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.05, origin=np.array([0.0, 0.0, 0.0])
                )
                bdft_frame.transform(bdft_pose_d415)
                visualizer.add_geometry(bdft_frame)
                if ft_coord:
                    bdft_f_d415, bdft_t_d415 = (
                        bdft_fs[current_idx],
                        bdft_ts[current_idx],
                    )
                else:
                    bdft_f_bdft, bdft_t_bdft = (
                        bdft_fs[current_idx],
                        bdft_ts[current_idx],
                    )
                    bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
                    bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
                bdft_f_value = np.linalg.norm(bdft_f_d415)
                bdft_f_rotation_d415 = rotation_vec2mat(bdft_f_d415 / bdft_f_value)
                bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
                bdft_t_value = np.linalg.norm(bdft_t_d415)
                bdft_t_rotation_d415 = rotation_vec2mat(bdft_t_d415 / bdft_t_value)
                bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
                bdft_f_arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.04 * 0.025,
                    cone_radius=0.04 * 0.05,
                    cylinder_height=0.04 * 0.875,
                    cone_height=0.04 * 0.125,
                    resolution=20,
                    cylinder_split=4,
                    cone_split=1,
                )
                bdft_f_arrow.paint_uniform_color([1.0, 1.0, 0.0])
                bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
                bdft_f_arrow.rotate(bdft_f_rotation_d415, np.array([[0], [0], [0]]))
                bdft_f_arrow.translate(bdft_f_translation_d415)
                visualizer.add_geometry(bdft_f_arrow)
                bdft_t_arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.4 * 0.025,
                    cone_radius=0.4 * 0.05,
                    cylinder_height=0.4 * 0.875,
                    cone_height=0.4 * 0.125,
                    resolution=20,
                    cylinder_split=4,
                    cone_split=1,
                )
                bdft_t_arrow.paint_uniform_color([0.0, 1.0, 1.0])
                bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
                bdft_t_arrow.rotate(bdft_t_rotation_d415, np.array([[0], [0], [0]]))
                bdft_t_arrow.translate(bdft_t_translation_d415)
                visualizer.add_geometry(bdft_t_arrow)
                bdft_peeler = o3d.io.read_triangle_mesh(
                    os.path.join("objs", "peeler.obj")
                )
                bdft_peeler.transform(bdft_pose_d415)
                visualizer.add_geometry(bdft_peeler)
                # add angler elements
                angler_width = angler_widths[current_idx]
                angler_right_finger = o3d.io.read_triangle_mesh(
                    os.path.join("objs", "right_finger.obj")
                )
                angler_left_finger = o3d.io.read_triangle_mesh(
                    os.path.join("objs", "left_finger.obj")
                )
                angler_finger_pose_gripper = np.identity(4)
                angler_finger_pose_gripper[0, 3] = angler_width / 2.0
                gripper_right_finger_pose_d415 = (
                    gripper_pose_d415 @ angler_finger_pose_gripper
                )
                angler_right_finger.transform(gripper_right_finger_pose_d415)
                visualizer.add_geometry(angler_right_finger)
                angler_finger_pose_gripper[0, 3] = -angler_width / 2.0
                gripper_left_finger_pose_d415 = (
                    gripper_pose_d415 @ angler_finger_pose_gripper
                )
                angler_left_finger.transform(gripper_left_finger_pose_d415)
                visualizer.add_geometry(angler_left_finger)

                # visualizer setup
                view_control = visualizer.get_view_control()
                visualizer.get_render_option().background_color = [0, 0, 0]
                params = view_control.convert_to_pinhole_camera_parameters()
                params.extrinsic = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
                view_control.convert_from_pinhole_camera_parameters(
                    params, allow_arbitrary=True
                )

                # visualize loop
                (
                    gripper_xyz_deltas,
                    gripper_quat_deltas,
                    bdft_xyz_deltas,
                    bdft_quat_deltas,
                    bdft_f_deltas,
                    bdft_t_deltas,
                    angler_width_deltas,
                ) = [], [], [], [], [], [], []
                with tqdm.tqdm(total=len_seq) as pbar:
                    while current_idx < len_seq:
                        # update d415 elements
                        d415_pc_xyz_d415, d415_pc_rgb = (
                            d415_pc_xyzs_d415[current_idx],
                            d415_pc_rgbs[current_idx],
                        )
                        d415_pcd.points = o3d.utility.Vector3dVector(d415_pc_xyz_d415)
                        d415_pcd.colors = o3d.utility.Vector3dVector(d415_pc_rgb)
                        visualizer.update_geometry(d415_pcd)
                        # update gripper elements
                        gripper_xyz_d415, gripper_quat_d415 = (
                            gripper_xyzs_d415[current_idx],
                            gripper_quats_d415[current_idx],
                        )
                        gripper_pose_d415_last = gripper_pose_d415.copy()
                        gripper_pose_d415 = xyzquat2mat(
                            gripper_xyz_d415, gripper_quat_d415
                        )
                        gripper_frame.transform(np.linalg.inv(gripper_pose_d415_last))
                        gripper_frame.transform(gripper_pose_d415)
                        visualizer.update_geometry(gripper_frame)
                        gripper.transform(np.linalg.inv(gripper_pose_d415_last))
                        gripper.transform(gripper_pose_d415)
                        gripper_delta_pose = np.dot(
                            np.linalg.inv(gripper_pose_d415_last), gripper_pose_d415
                        )
                        visualizer.update_geometry(gripper)
                        # update bdft elements
                        bdft_xyz_d415, bdft_quat_d415 = (
                            bdft_xyzs_d415[current_idx],
                            bdft_quats_d415[current_idx],
                        )
                        bdft_pose_d415_last = bdft_pose_d415.copy()
                        bdft_pose_d415 = xyzquat2mat(bdft_xyz_d415, bdft_quat_d415)
                        bdft_frame.transform(np.linalg.inv(bdft_pose_d415_last))
                        bdft_frame.transform(bdft_pose_d415)
                        bdft_delta_pose = np.dot(
                            np.linalg.inv(bdft_pose_d415_last), bdft_pose_d415
                        )
                        visualizer.update_geometry(bdft_frame)
                        if ft_coord:
                            bdft_f_d415_last, bdft_t_d415_last = (
                                bdft_f_d415.copy(),
                                bdft_t_d415.copy(),
                            )
                            bdft_f_d415, bdft_t_d415 = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                            bdft_delta_f, bdft_delta_t = (
                                bdft_f_d415 - bdft_f_d415_last,
                                bdft_t_d415 - bdft_t_d415_last,
                            )
                        else:
                            bdft_f_bdft_last, bdft_t_bdft_last = (
                                bdft_f_bdft.copy(),
                                bdft_t_bdft.copy(),
                            )
                            bdft_f_bdft, bdft_t_bdft = (
                                bdft_fs[current_idx],
                                bdft_ts[current_idx],
                            )
                            bdft_delta_f, bdft_delta_t = (
                                bdft_f_bdft - bdft_f_bdft_last,
                                bdft_t_bdft - bdft_t_bdft_last,
                            )
                            bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
                            bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
                        bdft_f_value_last = bdft_f_value.copy()
                        bdft_f_value = np.linalg.norm(bdft_f_d415)
                        bdft_f_rotation_d415_last = bdft_f_rotation_d415.copy()
                        bdft_f_rotation_d415 = rotation_vec2mat(
                            bdft_f_d415 / bdft_f_value
                        )
                        bdft_f_translation_d415_last = bdft_f_translation_d415.copy()
                        bdft_f_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_t_value_last = bdft_t_value.copy()
                        bdft_t_value = np.linalg.norm(bdft_t_d415)
                        bdft_t_rotation_d415_last = bdft_t_rotation_d415.copy()
                        bdft_t_rotation_d415 = rotation_vec2mat(
                            bdft_t_d415 / bdft_t_value
                        )
                        bdft_t_translation_d415_last = bdft_t_translation_d415.copy()
                        bdft_t_translation_d415 = bdft_pose_d415[:3, 3]
                        bdft_f_arrow.translate(-bdft_f_translation_d415_last)
                        bdft_f_arrow.rotate(
                            np.linalg.inv(bdft_f_rotation_d415_last),
                            np.array([[0], [0], [0]]),
                        )
                        bdft_f_arrow.scale(
                            1 / bdft_f_value_last, np.array([[0], [0], [0]])
                        )
                        bdft_f_arrow.scale(bdft_f_value, np.array([[0], [0], [0]]))
                        bdft_f_arrow.rotate(
                            bdft_f_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_f_arrow.translate(bdft_f_translation_d415)
                        visualizer.update_geometry(bdft_f_arrow)
                        bdft_t_arrow.translate(-bdft_t_translation_d415_last)
                        bdft_t_arrow.rotate(
                            np.linalg.inv(bdft_t_rotation_d415_last),
                            np.array([[0], [0], [0]]),
                        )
                        bdft_t_arrow.scale(
                            1 / bdft_t_value_last, np.array([[0], [0], [0]])
                        )
                        bdft_t_arrow.scale(bdft_t_value, np.array([[0], [0], [0]]))
                        bdft_t_arrow.rotate(
                            bdft_t_rotation_d415, np.array([[0], [0], [0]])
                        )
                        bdft_t_arrow.translate(bdft_t_translation_d415)
                        visualizer.update_geometry(bdft_t_arrow)
                        bdft_peeler.transform(np.linalg.inv(bdft_pose_d415_last))
                        bdft_peeler.transform(bdft_pose_d415)
                        visualizer.update_geometry(bdft_peeler)
                        # update angler elements
                        angler_width_last = angler_width.copy()
                        angler_width = angler_widths[current_idx]
                        angler_finger_pose_gripper = np.identity(4)
                        angler_finger_pose_gripper[0, 3] = angler_width / 2.0
                        gripper_right_finger_pose_d415_last = (
                            gripper_right_finger_pose_d415.copy()
                        )
                        gripper_right_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_right_finger.transform(
                            np.linalg.inv(gripper_right_finger_pose_d415_last)
                        )
                        angler_right_finger.transform(gripper_right_finger_pose_d415)
                        visualizer.update_geometry(angler_right_finger)
                        angler_finger_pose_gripper[0, 3] = -angler_width / 2.0
                        gripper_left_finger_pose_d415_last = (
                            gripper_left_finger_pose_d415.copy()
                        )
                        gripper_left_finger_pose_d415 = (
                            gripper_pose_d415 @ angler_finger_pose_gripper
                        )
                        angler_left_finger.transform(
                            np.linalg.inv(gripper_left_finger_pose_d415_last)
                        )
                        angler_left_finger.transform(gripper_left_finger_pose_d415)
                        visualizer.update_geometry(angler_left_finger)

                        # visualizer update
                        visualizer.poll_events()
                        visualizer.update_renderer()

                        # pbar update
                        current_idx_last = current_idx
                        if not pause:
                            current_idx += speed
                            pbar.update(speed)
                        else:
                            if forward:
                                current_idx += speed
                                pbar.update(speed)
                                forward = False
                            elif backward:
                                current_idx -= speed
                                pbar.update(-speed)
                                backward = False
                            else:
                                pass
                        pbar.set_postfix(f=bdft_f_value, t=bdft_t_value)

                        # keyboard quit
                        if quit:
                            break
                        # keyboard reset
                        if reset:
                            view_control = visualizer.get_view_control()
                            params = view_control.convert_to_pinhole_camera_parameters()
                            params.extrinsic = np.array(
                                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                            )
                            view_control.convert_from_pinhole_camera_parameters(
                                params, allow_arbitrary=True
                            )
                            reset = False
                        # keyboard zero
                        if zero:
                            pbar.update(-current_idx)
                            current_idx = 0
                            zero = False

                        if current_idx != current_idx_last:
                            gripper_xyz_delta_mm = (
                                np.linalg.norm(gripper_delta_pose[:3, 3]) * 1000
                            )
                            gripper_quat_delta_deg = (
                                np.arccos(
                                    np.clip(
                                        (np.trace(gripper_delta_pose[:3, :3]) - 1) / 2,
                                        -1,
                                        1,
                                    )
                                )
                                / np.pi
                                * 180
                            )
                            bdft_xyz_delta_mm = (
                                np.linalg.norm(bdft_delta_pose[:3, 3]) * 1000
                            )
                            bdft_quat_delta_deg = (
                                np.arccos(
                                    np.clip(
                                        (np.trace(bdft_delta_pose[:3, :3]) - 1) / 2,
                                        -1,
                                        1,
                                    )
                                )
                                / np.pi
                                * 180
                            )
                            angler_width_delta_mm = (
                                abs(angler_width - angler_width_last) * 1000
                            )
                            bdft_f_delta_n = np.linalg.norm(bdft_delta_f)
                            bdft_t_delta = np.linalg.norm(bdft_delta_t)
                            ### print(gripper_xyz_delta_mm)

                            gripper_xyz_deltas.append(gripper_xyz_delta_mm)
                            gripper_quat_deltas.append(gripper_quat_delta_deg)
                            bdft_xyz_deltas.append(bdft_xyz_delta_mm)
                            bdft_quat_deltas.append(bdft_quat_delta_deg)
                            angler_width_deltas.append(angler_width_delta_mm)
                            bdft_f_deltas.append(bdft_f_delta_n)
                            bdft_t_deltas.append(bdft_t_delta)

                visualizer.clear_geometries()
                listener.stop()


if __name__ == "__main__":
    args = config_parse()
    main(args)
