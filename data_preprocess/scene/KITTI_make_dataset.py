import os
import sys
sys.dont_write_bytecode = True
import argparse

import numpy as np
import open3d as o3d
from KITTI_utils import load_lidar_bin, load_ground_truth, compute_indices_dists, load_calibration



def convert_cam_pose_to_lidar_pose(T_cam, Tr_velo_to_cam):
    """
    Convert camera-frame GT pose into LiDAR (Velodyne) frame.

    Tr_velo_to_cam: T_cam <- T_velo (4x4)
    T_lidar = inv(Tr_velo_to_cam) @ T_cam @ Tr_velo_to_cam
    """
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    return Tr_cam_to_velo @ T_cam @ Tr_velo_to_cam


def compute_relative_transform_lidar(T1_lidar, T2_lidar):
    T_rel = np.linalg.inv(T2_lidar) @ T1_lidar
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3].reshape(3, 1)
    return R_rel, t_rel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_scan', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--voxel_size', type=float, default=0.2)
    parser.add_argument('--calib', type=str, required=True)
    return parser.parse_args()


def main(args):
    lidar_folder = args.lidar_scan
    gt_file = args.ground_truth
    calib_file = args.calib
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gt_cam = load_ground_truth(gt_file)
    Tr_velo_to_cam = load_calibration(calib_file)

    print("Loaded calib:\n", Tr_velo_to_cam)

    lidar_files = sorted(os.listdir(lidar_folder), key=lambda f: float(os.path.splitext(f)[0]))

    max_points = 0

    for i in range(len(lidar_files) - 1):

        out_name = os.path.join(output_folder, f"sample_{i:06d}.npz")
        if os.path.exists(out_name):
            print("[Skip]", out_name)
            continue

        f1, f2 = lidar_files[i], lidar_files[i+1]
        P = load_lidar_bin(os.path.join(lidar_folder, f1))
        Q = load_lidar_bin(os.path.join(lidar_folder, f2))

        P_p = o3d.geometry.PointCloud()
        Q_p = o3d.geometry.PointCloud()
        P_p.points = o3d.utility.Vector3dVector(P.T)
        Q_p.points = o3d.utility.Vector3dVector(Q.T)
        P = np.asarray(P_p.voxel_down_sample(args.voxel_size).points).T
        Q = np.asarray(Q_p.voxel_down_sample(args.voxel_size).points).T

        max_points = max(max_points, P.shape[1])
        print("max points =", max_points)


        T1_cam = gt_cam[i]
        T2_cam = gt_cam[i + 1]

        T1_lidar = convert_cam_pose_to_lidar_pose(T1_cam, Tr_velo_to_cam)
        T2_lidar = convert_cam_pose_to_lidar_pose(T2_cam, Tr_velo_to_cam)


        R_rel, t_rel = compute_relative_transform_lidar(T1_lidar, T2_lidar)


        idx, dst = compute_indices_dists(P, Q, R_rel, t_rel)

        np.savez(
            out_name,
            P=P, Q=Q,
            R_rel=R_rel, t_rel=t_rel,
            indices=idx, dists=dst
        )

        print("[Saved]", out_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
