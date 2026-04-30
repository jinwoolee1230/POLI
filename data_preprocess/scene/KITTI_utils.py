import numpy as np
import torch
from scipy.spatial import cKDTree


# ============================================================
# Load KITTI Velodyne LiDAR .bin
# ============================================================
def load_lidar_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3].T  # (3, N)


# ============================================================
# Load KITTI poses.txt (camera frame)
# ============================================================
def load_ground_truth(path):
    poses = []
    with open(path, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T_h = np.eye(4, dtype=np.float32)
            T_h[:3, :] = T
            poses.append(T_h)
    return poses


# ============================================================
# Return camera-frame pose by index
# ============================================================
def get_pose_from_gt(gt_data, index):
    return gt_data[index]


# ============================================================
# Load KITTI calib.txt (Tr: row)
# ============================================================
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    Tr = None
    for line in lines:
        if line.startswith("Tr:"):
            vals = line.split()[1:]
            vals = [float(v) for v in vals]
            Tr = np.array(vals).reshape(3, 4)
            break

    if Tr is None:
        raise ValueError("Tr not found in calib.txt")

    Tr_h = np.eye(4, dtype=np.float32)
    Tr_h[:3, :] = Tr
    return Tr_h  # (T_cam <- T_velo)


# ============================================================
# Apply homogeneous transform to point cloud
# ============================================================
def apply_transform(P, T):
    N = P.shape[1]
    homog = np.vstack((P, np.ones((1, N))))
    P_trans = T @ homog
    return P_trans[:3, :]


# ============================================================
# Compute relative transform in camera frame
# ============================================================
def compute_relative_transform(pose1, pose2):
    """
    pose_i: 4x4 camera-frame pose
    T_rel = inv(T2) @ T1
    """
    T_rel = np.linalg.inv(pose2) @ pose1
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3].reshape(3,1)
    return R_rel, t_rel


# ============================================================
# NN correspondences
# ============================================================
def compute_indices_dists(P, Q, R_rel, t_rel):
    P_trans = (R_rel @ P) + t_rel
    tree = cKDTree(Q.T)
    dists, idx = tree.query(P_trans.T, k=1)
    return idx, dists
