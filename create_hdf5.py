import os
import configargparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import tqdm
import h5py

# 第三视角摄像头
from r3kit.devices.camera.realsense.d415 import D415

# 第一视角摄像头
from r3kit.devices.camera.realsense.d455 import D455

# 力矩传感器
from r3kit.devices.ftsensor.bluedot.bluedot_lb75 import BlueDotLB75 as Bdft

# 角度编码器
from r3kit.devices.encoder.pdcd.angler_hz import Angler
from configs.pose import (
    D415_2_D455,
    D455_2_BDFT,
    D415_2_BASE,
    ANGLE_2_WIDTH,
    OBJECT_SPACE,
    BDFT_SPACE,
    BASE_SPACE,
)
from utils.annotation import search_stage
from utils.transformation import transform_pc
from utils.process import voxelize, mesh2pc

"""
Synchronize with `visualize_data.py` some part
"""


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--clip_object", action="store_true")
    parser.add_argument("--clip_gripper", action="store_true")
    parser.add_argument("--clip_bdft", action="store_true")
    parser.add_argument("--clip_base", action="store_true")
    parser.add_argument("--render_gripper_num", type=int)
    parser.add_argument("--render_bdft_num", type=int)
    parser.add_argument("--voxel_size", type=float)
    parser.add_argument("--pc_num", type=int)

    args = parser.parse_args()
    return args


def main(args):
    # general config
    data_path = args.data_path
    save_path = args.save_path
    clip_object = args.clip_object
    clip_gripper = args.clip_gripper
    clip_bdft = args.clip_bdft
    clip_base = args.clip_base
    render_gripper_num = args.render_gripper_num
    render_bdft_num = args.render_bdft_num
    voxel_size = args.voxel_size
    pc_num = args.pc_num

    # load stage data
    # 用D455构建的基坐标系
    with open(os.path.join(data_path, "stage_timestamp_ms.json"), "r") as f:
        stage_timestamp_ms = json.load(f)
        d455_pose_start_timestamp_ms = stage_timestamp_ms[
            "d455_pose_start_timestamp_ms"
        ]
        d455_pose_end_timestamp_ms = stage_timestamp_ms["d455_pose_end_timestamp_ms"]
        _start_timestamp_ms = stage_timestamp_ms["start_timestamp_ms"]

    # load d415 data
    d415_path = os.path.join(data_path, "d415")
    d415_intrinsics = np.loadtxt(
        os.path.join(d415_path, "intrinsics.txt")
    )  # (4,), float64
    d415_depth_scale = np.loadtxt(os.path.join(d415_path, "depth_scale.txt")).item()
    d415_timestamps = np.load(os.path.join(d415_path, "timestamps.npy"))

    # load d455 data
    d455_path = os.path.join(data_path, "d455")
    d455_pose_path = os.path.join(d455_path, "pose")
    d455_pose_timestamps = np.load(os.path.join(d455_pose_path, "timestamps.npy"))
    d455_xyzs = np.load(os.path.join(d455_pose_path, "xyz.npy"))
    d455_quats = np.load(os.path.join(d455_pose_path, "quat.npy"))

    # load bdft data
    bdft_path = os.path.join(data_path, "bdft")
    with open(os.path.join(bdft_path, "tare_bdft.json"), "r") as f:
        bdft_tare = json.load(f)
    bdft_timestamps = np.load(os.path.join(bdft_path, "timestamps.npy"))
    bdft_fts = np.load(os.path.join(bdft_path, "ft.npy"))

    # load angler data
    angler_path = os.path.join(data_path, "angler")
    angler_timestamps = np.load(os.path.join(angler_path, "timestamps.npy"))
    angler_angles = np.load(os.path.join(angler_path, "angle.npy"))
    # load annotation data
    annotation_path = os.path.join(data_path, "annotation.json")
    with open(annotation_path, "r") as f:
        stages = json.load(f)

    # deal with d455 special prepare
    d455_initial_pose_mask = np.logical_and(
        d455_pose_timestamps > d455_pose_start_timestamp_ms,
        d455_pose_timestamps < d455_pose_end_timestamp_ms,
    )
    d455_initial_xyz = np.median(d455_xyzs[d455_initial_pose_mask, :], axis=0)
    d455_initial_quat = np.median(d455_quats[d455_initial_pose_mask, :], axis=0)
    d455_initial_pose = D455.raw2pose(d455_initial_xyz, d455_initial_quat)  # c02w

    # deal with angler special prepare
    angler_angles = Angler.raw2angle(angler_angles)
    angler_angles[angler_angles < 0] = 0.0
    angler_widths = angler_angles * ANGLE_2_WIDTH

    # process loop
    data_dict = {}
    stage_idxs = {"grasp": 0, "shave": 0, "turn": 0}
    stage_idx_map = {}
    for d415_current_idx in tqdm.trange(len(d415_timestamps)):
        # process d415 variables
        d415_current_timestamp = d415_timestamps[d415_current_idx]
        # process d455 variables
        d455_pose_current_idx = np.searchsorted(
            d455_pose_timestamps, d415_current_timestamp
        )
        d455_pose_current_idx = min(
            d455_pose_current_idx, len(d455_pose_timestamps) - 1
        )

        # process bdft variables
        bdft_current_idx = np.searchsorted(bdft_timestamps, d415_current_timestamp)
        bdft_current_idx = min(bdft_current_idx, len(bdft_timestamps) - 1)
        # process angler variables
        angler_current_idx = np.searchsorted(angler_timestamps, d415_current_timestamp)
        angler_current_idx = min(angler_current_idx, len(angler_timestamps) - 1)

        # process stage
        stage_idx = search_stage(d415_current_timestamp, stages)
        stage = stages[stage_idx]
        d455_xyz_d455w_bias = np.array(stage["d455_xyz_d455w_bias"])
        if stage["stage"] == "unrelated":
            continue
        if stage_idx not in stage_idx_map:
            stage_idx_map[stage_idx] = (
                stage["stage"] + "_" + str(stage_idxs[stage["stage"]]).zfill(2)
            )
            stage_idxs[stage["stage"]] = stage_idxs[stage["stage"]] + 1
            data_dict[stage_idx_map[stage_idx]] = {
                "d415_pc_xyzs_d415": [],
                "d415_pc_rgbs": [],
                "d415_pc_xyzs_d415_mesh": [],
                "d415_pc_rgbs_mesh": [],
                "gripper_xyzs_d415": [],
                "gripper_quats_d415": [],
                "bdft_xyzs_d415": [],
                "bdft_quats_d415": [],
                "bdft_fs_bdft": [],
                "bdft_ts_bdft": [],
                "bdft_fs_d415": [],
                "bdft_ts_d415": [],
                "angler_widths": [],
            }

        # process d455 elements
        d455_xyz_d455w, d455_quat_d455w = (
            d455_xyzs[d455_pose_current_idx],
            d455_quats[d455_pose_current_idx],
        )
        d455_xyz_d455w = d455_xyz_d455w + d455_xyz_d455w_bias
        d455_pose_d455w = D455.raw2pose(d455_xyz_d455w, d455_quat_d455w)  # c2w
        d455_pose_d4550 = (
            np.linalg.inv(d455_initial_pose) @ d455_pose_d455w
        )  # c2c0 = w2c0 @ c2w
        d455_pose_d415 = (
            np.linalg.inv(D415_2_D455) @ d455_pose_d4550
        )  # c2l = c02l @ c2c0

        # process d455 elements
        # gripper_pose_d415 = d455_pose_d415 @ np.linalg.inv(D455_2_GRIPPER)                # g2l = c2l @ g2c
        # gripper_xyz_d415, gripper_quat_d415 = gripper_pose_d415[:3, 3], Rot.from_matrix(gripper_pose_d415[:3, :3]).as_quat()
        # data_dict[stage_idx_map[stage_idx]]['gripper_xyzs_d415'].append(gripper_xyz_d415)
        # data_dict[stage_idx_map[stage_idx]]['gripper_quats_d415'].append(gripper_quat_d415)
        # process bdft elements
        bdft_pose_d415 = d455_pose_d415 @ np.linalg.inv(D455_2_BDFT)  # f2l = c2l @ f2c
        bdft_xyz_d415, bdft_quat_d415 = (
            bdft_pose_d415[:3, 3],
            Rot.from_matrix(bdft_pose_d415[:3, :3]).as_quat(),
        )
        data_dict[stage_idx_map[stage_idx]]["bdft_xyzs_d415"].append(bdft_xyz_d415)
        data_dict[stage_idx_map[stage_idx]]["bdft_quats_d415"].append(bdft_quat_d415)
        bdft_ft_bdft = bdft_fts[bdft_current_idx]
        bdft_pose_base = D415_2_BASE @ bdft_pose_d415  # f2b = l2b @ f2l
        bdft_ft_bdft = Bdft.raw2tare(bdft_ft_bdft, bdft_tare, bdft_pose_base[:3, :3])
        bdft_f_bdft, bdft_t_bdft = bdft_ft_bdft[:3], bdft_ft_bdft[3:]
        bdft_f_d415 = bdft_pose_d415[:3, :3] @ bdft_f_bdft
        bdft_t_d415 = bdft_pose_d415[:3, :3] @ bdft_t_bdft
        data_dict[stage_idx_map[stage_idx]]["bdft_fs_bdft"].append(bdft_f_bdft)
        data_dict[stage_idx_map[stage_idx]]["bdft_ts_bdft"].append(bdft_t_bdft)
        data_dict[stage_idx_map[stage_idx]]["bdft_fs_d415"].append(bdft_f_d415)
        data_dict[stage_idx_map[stage_idx]]["bdft_ts_d415"].append(bdft_t_d415)
        # process angler elements
        angler_width = angler_widths[angler_current_idx]
        data_dict[stage_idx_map[stage_idx]]["angler_widths"].append(angler_width)
        # process d415 elements
        d415_color_img = cv2.imread(
            os.path.join(d415_path, "color", f"{str(d415_current_idx).zfill(16)}.png"),
            cv2.IMREAD_COLOR,
        )
        d415_color_img = cv2.cvtColor(
            d415_color_img, cv2.COLOR_BGR2RGB
        )  # (H, W, 3), uint8
        d415_color_img = d415_color_img / 255.0  # (H, W, 3), float64
        d415_depth_img = cv2.imread(
            os.path.join(d415_path, "depth", f"{str(d415_current_idx).zfill(16)}.png"),
            cv2.IMREAD_ANYDEPTH,
        )  # (H, W), uint16
        d415_depth_img = d415_depth_img * d415_depth_scale  # (H, W), float64
        d415_pc_xyz_d415, d415_pc_rgb = D415.img2pc(
            d415_depth_img, d415_intrinsics, d415_color_img
        )
        if clip_object:
            d415_pc_xyz_base = transform_pc(d415_pc_xyz_d415, D415_2_BASE)
            clip_object_mask = (
                (d415_pc_xyz_base[:, 0] > OBJECT_SPACE[0][0])
                & (d415_pc_xyz_base[:, 0] < OBJECT_SPACE[0][1])
                & (d415_pc_xyz_base[:, 1] > OBJECT_SPACE[1][0])
                & (d415_pc_xyz_base[:, 1] < OBJECT_SPACE[1][1])
                & (d415_pc_xyz_base[:, 2] > OBJECT_SPACE[2][0])
                & (d415_pc_xyz_base[:, 2] < OBJECT_SPACE[2][1])
            )
        else:
            clip_object_mask = np.zeros((d415_pc_xyz_d415.shape[0],), dtype=bool)
        # if clip_gripper:
        #     d415_pc_xyz_gripper = transform_pc(d415_pc_xyz_d415, np.linalg.inv(gripper_pose_d415))
        #     clip_gripper_mask = (d415_pc_xyz_gripper[:, 0] > GRIPPER_SPACE[0][0]) & (d415_pc_xyz_gripper[:, 0] < GRIPPER_SPACE[0][1]) & \
        #                         (d415_pc_xyz_gripper[:, 1] > GRIPPER_SPACE[1][0]) & (d415_pc_xyz_gripper[:, 1] < GRIPPER_SPACE[1][1]) & \
        #                         (d415_pc_xyz_gripper[:, 2] > GRIPPER_SPACE[2][0]) & (d415_pc_xyz_gripper[:, 2] < GRIPPER_SPACE[2][1])
        # else:
        #     clip_gripper_mask = np.zeros((d415_pc_xyz_d415.shape[0],), dtype=bool)
        if clip_bdft:
            d415_pc_xyz_bdft = transform_pc(
                d415_pc_xyz_d415, np.linalg.inv(bdft_pose_d415)
            )
            clip_bdft_mask = (
                (d415_pc_xyz_bdft[:, 0] > BDFT_SPACE[0][0])
                & (d415_pc_xyz_bdft[:, 0] < BDFT_SPACE[0][1])
                & (d415_pc_xyz_bdft[:, 1] > BDFT_SPACE[1][0])
                & (d415_pc_xyz_bdft[:, 1] < BDFT_SPACE[1][1])
                & (d415_pc_xyz_bdft[:, 2] > BDFT_SPACE[2][0])
                & (d415_pc_xyz_bdft[:, 2] < BDFT_SPACE[2][1])
            )
        else:
            clip_bdft_mask = np.zeros((d415_pc_xyz_d415.shape[0],), dtype=bool)
        if clip_base:
            d415_pc_xyz_base = transform_pc(d415_pc_xyz_d415, D415_2_BASE)
            clip_base_mask = (
                (d415_pc_xyz_base[:, 0] > BASE_SPACE[0][0])
                & (d415_pc_xyz_base[:, 0] < BASE_SPACE[0][1])
                & (d415_pc_xyz_base[:, 1] > BASE_SPACE[1][0])
                & (d415_pc_xyz_base[:, 1] < BASE_SPACE[1][1])
                & (d415_pc_xyz_base[:, 2] > BASE_SPACE[2][0])
                & (d415_pc_xyz_base[:, 2] < BASE_SPACE[2][1])
            )
        else:
            clip_base_mask = np.ones((d415_pc_xyz_d415.shape[0],), dtype=bool)
        # valid_mask = np.logical_and(clip_base_mask, np.logical_or(clip_object_mask, np.logical_or(clip_gripper_mask, clip_bdft_mask)))
        valid_mask = np.logical_and(
            clip_base_mask, np.logical_or(clip_object_mask, clip_bdft_mask)
        )
        # TODO: hardcode to throw out hands
        # d415_pc_xyz_gripper = transform_pc(d415_pc_xyz_d415, np.linalg.inv(gripper_pose_d415))
        # valid_mask = np.logical_and(valid_mask, d415_pc_xyz_gripper[:, 2] > 0.)
        d415_pc_xyz_bdft = transform_pc(d415_pc_xyz_d415, np.linalg.inv(bdft_pose_d415))
        valid_mask = np.logical_and(valid_mask, d415_pc_xyz_bdft[:, 2] > 0.0)
        valid_mask = np.where(valid_mask)[0]
        d415_pc_xyz_d415 = d415_pc_xyz_d415[valid_mask]
        d415_pc_rgb = d415_pc_rgb[valid_mask]
        d415_pc_xyz_d415_mesh = d415_pc_xyz_d415.copy()
        d415_pc_rgb_mesh = d415_pc_rgb.copy()
        # if render_gripper_num != 0:
        #     rfinger_pc_xyz_rfinger_mesh = mesh2pc(os.path.join("objs", "right_finger.obj"), num_points=render_gripper_num//2)
        #     lfinger_pc_xyz_lfinger_mesh = mesh2pc(os.path.join("objs", "left_finger.obj"), num_points=render_gripper_num//2)
        #     angler_finger_pose_gripper = np.identity(4)
        #     angler_finger_pose_gripper[0, 3] = angler_width / 2.
        # gripper_right_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
        # rfinger_pc_xyz_d415_mesh = transform_pc(rfinger_pc_xyz_rfinger_mesh, gripper_right_finger_pose_d415)
        # angler_finger_pose_gripper[0, 3] = -angler_width / 2.
        # gripper_left_finger_pose_d415 = gripper_pose_d415 @ angler_finger_pose_gripper
        # lfinger_pc_xyz_d415_mesh = transform_pc(lfinger_pc_xyz_lfinger_mesh, gripper_left_finger_pose_d415)
        # rfinger_pc_rgb_mesh = np.ones_like(rfinger_pc_xyz_d415_mesh)
        # lfinger_pc_rgb_mesh = np.ones_like(lfinger_pc_xyz_d415_mesh)
        # d415_pc_xyz_d415_mesh = np.concatenate([d415_pc_xyz_d415_mesh, rfinger_pc_xyz_d415_mesh, lfinger_pc_xyz_d415_mesh], axis=0)
        # d415_pc_rgb_mesh = np.concatenate([d415_pc_rgb_mesh, rfinger_pc_rgb_mesh, lfinger_pc_rgb_mesh], axis=0)
        if render_bdft_num != 0:
            bdft_pc_xyz_bdft_mesh = mesh2pc(
                os.path.join("objs", "only_peeler.obj"), num_points=render_bdft_num
            )
            bdft_pc_xyz_d415_mesh = transform_pc(bdft_pc_xyz_bdft_mesh, bdft_pose_d415)
            bdft_pc_rgb_mesh = np.ones_like(bdft_pc_xyz_d415_mesh)
            d415_pc_xyz_d415_mesh = np.concatenate(
                [d415_pc_xyz_d415_mesh, bdft_pc_xyz_d415_mesh], axis=0
            )
            d415_pc_rgb_mesh = np.concatenate(
                [d415_pc_rgb_mesh, bdft_pc_rgb_mesh], axis=0
            )
        if voxel_size != 0:
            d415_pc_xyz_d415, d415_pc_rgb = voxelize(
                d415_pc_xyz_d415, d415_pc_rgb, voxel_size
            )
            d415_pc_xyz_d415_mesh, d415_pc_rgb_mesh = voxelize(
                d415_pc_xyz_d415_mesh, d415_pc_rgb_mesh, voxel_size
            )
        # if pc_num != -1:
        #     print(f"d415_pc_xyz_d415.shape[0] = {d415_pc_xyz_d415.shape[0]}")
        #     if d415_pc_xyz_d415.shape[0] > pc_num:
        #         valid_mask = np.random.choice(d415_pc_xyz_d415.shape[0], pc_num, replace=False)
        #     elif d415_pc_xyz_d415.shape[0] < pc_num:
        #         print(f"Warning: {d415_pc_xyz_d415.shape[0] = }")
        #         valid_mask = np.concatenate([np.arange(d415_pc_xyz_d415.shape[0]), np.random.choice(d415_pc_xyz_d415.shape[0], pc_num - d415_pc_xyz_d415.shape[0], replace=False)], axis=0)
        #     d415_pc_xyz_d415 = d415_pc_xyz_d415[valid_mask]
        #     d415_pc_rgb = d415_pc_rgb[valid_mask]
        #     if d415_pc_xyz_d415_mesh.shape[0] > pc_num:
        #         valid_mask = np.random.choice(d415_pc_xyz_d415_mesh.shape[0], pc_num, replace=False)
        #     elif d415_pc_xyz_d415_mesh.shape[0] < pc_num:
        #         print(f"Warning: {d415_pc_xyz_d415_mesh.shape[0] = }")
        #         valid_mask = np.concatenate([np.arange(d415_pc_xyz_d415_mesh.shape[0]), np.random.choice(d415_pc_xyz_d415_mesh.shape[0], pc_num - d415_pc_xyz_d415_mesh.shape[0], replace=False)], axis=0)
        #     d415_pc_xyz_d415_mesh = d415_pc_xyz_d415_mesh[valid_mask]
        #     d415_pc_rgb_mesh = d415_pc_rgb_mesh[valid_mask]
        data_dict[stage_idx_map[stage_idx]]["d415_pc_xyzs_d415"].append(
            d415_pc_xyz_d415
        )
        data_dict[stage_idx_map[stage_idx]]["d415_pc_rgbs"].append(d415_pc_rgb)
        data_dict[stage_idx_map[stage_idx]]["d415_pc_xyzs_d415_mesh"].append(
            d415_pc_xyz_d415_mesh
        )
        data_dict[stage_idx_map[stage_idx]]["d415_pc_rgbs_mesh"].append(
            d415_pc_rgb_mesh
        )

        # 假设你全局有这几个变量
        # pc_num = 1024  # 你想固定的点数
        # clip_object, clip_gripper, clip_bdft, clip_base, render_gripper_num, render_bdft_num, voxel_size 都是你定义好的

        def fix_pointcloud_list(pc_list, target_num):
            fixed_list = []
            for pc in pc_list:
                pc = np.array(pc)
                n = pc.shape[0]
                if n == target_num:
                    fixed_list.append(pc)
                elif n > target_num:
                    idxs = np.random.choice(n, target_num, replace=False)
                    fixed_list.append(pc[idxs])
                else:
                    pad = np.zeros((target_num - n, pc.shape[1]))
                    fixed_list.append(np.vstack([pc, pad]))
            return np.array(fixed_list)

        def fix_rgb_list(rgb_list, target_num):
            fixed_list = []
            for rgb in rgb_list:
                rgb = np.array(rgb)
                n = rgb.shape[0]
                if n == target_num:
                    fixed_list.append(rgb)
                elif n > target_num:
                    idxs = np.random.choice(n, target_num, replace=False)
                    fixed_list.append(rgb[idxs])
                else:
                    pad = np.zeros((target_num - n, rgb.shape[1]))
                    fixed_list.append(np.vstack([rgb, pad]))
            return np.array(fixed_list)

        # hdf5 loop
        os.makedirs(save_path, exist_ok=True)
        for stage_name in tqdm.tqdm(data_dict.keys()):
            with h5py.File(
                os.path.join(save_path, stage_name + ".hdf5"), "w"
            ) as stage_hdf5:
                stage_hdf5_data_group = stage_hdf5.create_group("data")

                # save observation
                stage_hdf5_o_group = stage_hdf5_data_group.create_group("o")

                # 直接保存的字段
                fixed_xyzs_mesh = fix_pointcloud_list(
                    data_dict[stage_name]["d415_pc_xyzs_d415"], pc_num
                )
                fixed_rgbs_mesh = fix_rgb_list(
                    data_dict[stage_name]["d415_pc_rgbs"], pc_num
                )
                stage_hdf5_o_group.create_dataset(
                    "d415_pc_xyzs_d415", data=fixed_xyzs_mesh
                )
                stage_hdf5_o_group.create_dataset("d415_pc_rgbs", data=fixed_rgbs_mesh)

                # 先修正点云和颜色，使每帧点数固定为 pc_num
                fixed_xyzs_mesh = fix_pointcloud_list(
                    data_dict[stage_name]["d415_pc_xyzs_d415_mesh"], pc_num
                )
                fixed_rgbs_mesh = fix_rgb_list(
                    data_dict[stage_name]["d415_pc_rgbs_mesh"], pc_num
                )

                stage_hdf5_o_group.create_dataset(
                    "d415_pc_xyzs_d415_mesh", data=fixed_xyzs_mesh
                )
                stage_hdf5_o_group.create_dataset(
                    "d415_pc_rgbs_mesh", data=fixed_rgbs_mesh
                )

                # 其余字段正常保存
                stage_hdf5_o_group.create_dataset(
                    "gripper_xyzs_d415",
                    data=np.array(data_dict[stage_name]["gripper_xyzs_d415"]),
                )
                stage_hdf5_o_group.create_dataset(
                    "gripper_quats_d415",
                    data=np.array(data_dict[stage_name]["gripper_quats_d415"]),
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_xyzs_d415",
                    data=np.array(data_dict[stage_name]["bdft_xyzs_d415"]),
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_quats_d415",
                    data=np.array(data_dict[stage_name]["bdft_quats_d415"]),
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_fs_bdft", data=np.array(data_dict[stage_name]["bdft_fs_bdft"])
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_ts_bdft", data=np.array(data_dict[stage_name]["bdft_ts_bdft"])
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_fs_d415", data=np.array(data_dict[stage_name]["bdft_fs_d415"])
                )
                stage_hdf5_o_group.create_dataset(
                    "bdft_ts_d415", data=np.array(data_dict[stage_name]["bdft_ts_d415"])
                )
                stage_hdf5_o_group.create_dataset(
                    "angler_widths",
                    data=np.array(data_dict[stage_name]["angler_widths"]),
                )

                # save attributes
                stage_hdf5_data_group.attrs["num_samples"] = len(
                    data_dict[stage_name]["d415_pc_xyzs_d415"]
                )
                stage_hdf5_o_group.attrs["clip_object"] = clip_object
                stage_hdf5_o_group.attrs["clip_gripper"] = clip_gripper
                stage_hdf5_o_group.attrs["clip_bdft"] = clip_bdft
                stage_hdf5_o_group.attrs["clip_base"] = clip_base
                stage_hdf5_o_group.attrs["render_gripper_num"] = render_gripper_num
                stage_hdf5_o_group.attrs["render_bdft_num"] = render_bdft_num
                stage_hdf5_o_group.attrs["voxel_size"] = voxel_size
                stage_hdf5_o_group.attrs["pc_num"] = pc_num


if __name__ == "__main__":
    args = config_parse()
    main(args)
