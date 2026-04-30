import os 
import numpy as np
import argparse
import open3d as o3d
from HeLiPR_utils import load_lidar_bin, load_ground_truth, get_pose_from_gt, compute_relative_transform, compute_indices_dists

def parse_args():
    parser = argparse.ArgumentParser(description='Make dataset from LiDAR scans and Ground Truth')
    parser.add_argument('--lidar_scan', type = str, required = True, help= 'Path to folder containing LiDAR scans')
    parser.add_argument('--ground_truth', type = str, required = True, help = 'Path to ground truth txt files (TUM format)')
    parser.add_argument('--output_folder', type = str, required = True, default='./output_dataset', help = 'Path to output folder')
    parser.add_argument('--voxel_size', type = float, default = 1.0, help = 'Voxel size for downsampling')
    return  parser.parse_args()

def main(args):
    lidar_folder = args.lidar_scan
    gt_file = args.ground_truth
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gt_data = load_ground_truth(gt_file)
    
    lidar_files = sorted(os.listdir(lidar_folder), key=lambda f: float(os.path.splitext(f)[0]))

    for i in range(len(lidar_files)-1):
        if os.path.exists(os.path.join(output_folder, f"sample_{i:06d}.npz")):
            print(f"Sample {i} already exists. Skipping...")
            continue
        file1 = lidar_files[i]
        file2 = lidar_files[i+1]
        path1 = os.path.join(lidar_folder, file1)
        path2 = os.path.join(lidar_folder, file2)

        P = load_lidar_bin(path1) # shape: (3, N)
        Q = load_lidar_bin(path2) # shape: (3, M)
        
        # ✅ Voxel Downsampling
        P_pcd = o3d.geometry.PointCloud()
        P_pcd.points = o3d.utility.Vector3dVector(P.T)  # float32도 허용됨
        P = np.asarray(P_pcd.voxel_down_sample(voxel_size=args.voxel_size).points).T  # (N, 3)

        Q_pcd = o3d.geometry.PointCloud()
        Q_pcd.points = o3d.utility.Vector3dVector(Q.T)
        Q = np.asarray(Q_pcd.voxel_down_sample(voxel_size=args.voxel_size).points).T
        
        timestamp1 = float(os.path.splitext(file1)[0])
        timestamp2 = float(os.path.splitext(file2)[0])

        pose1 = get_pose_from_gt(gt_data, timestamp1)
        pose2 = get_pose_from_gt(gt_data, timestamp2)

        R_rel, t_rel = compute_relative_transform(pose1, pose2)

        indices, dists = compute_indices_dists(P, Q, R_rel, t_rel)
        sample_filename = os.path.join(output_folder, f"sample_{i:06d}")
        np.savez(sample_filename, P=P, Q=Q, R_rel=R_rel, t_rel=t_rel, indices=indices, dists=dists)
        print(f"Saved sample {i} to {sample_filename}")

if __name__ == '__main__':
    args = parse_args()
    main(args)