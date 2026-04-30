import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import struct

def load_lidar_bin(path):
    # if os2, num = 26
    # if vlp, num = 22
    #* chenck https://github.com/minwoo0611/HeLiPR-Pointcloud-Toolbox/blob/master/python/binVisualizer.py
    points = []
    with open(path, "rb") as f:
        while True:
            data = f.read(26)
            if len(data) < 26:
                break
            # unpack 첫 16바이트에서 x, y, z, intensity (intensity는 사용하지 않음)
            x, y, z, _ = struct.unpack('ffff', data[:16])
            points.append([x, y, z])
    points = np.array(points, dtype=np.float32)  # (N, 3)
    return points.T  # (3, N)

def load_ground_truth(path):
    data = np.loadtxt(path)
    return data

def get_pose_from_gt(gt_data, timestamp):
    idx = np.argmin(np.abs(gt_data[:, 0] - timestamp))
    return gt_data[idx, 1:]

def compute_relative_transform(pose1, pose2):
    t1 = pose1[:3] # G_t_P
    t2 = pose2[:3] # G_t_Q
    r1 = R.from_quat(pose1[3:])
    r2 = R.from_quat(pose2[3:])
    R1 = r1.as_matrix() # G_R_P
    R2 = r2.as_matrix() # G_R_Q
    R_rel = R2.T @ R1 # Q_R_P = (G_R_Q)^T @ G_R_P
    t_rel = R2.T @ (t1 - t2) # Q_t_P = (G_R_Q)^T @ (G_t_P - G_t_Q)
    
    #! output: Q_T_P = [R_rel, t_rel]
    return R_rel, t_rel.reshape(3,1)

def compute_correspondences(P, Q, R_rel, t_rel):
    '''
    #! P: (3, N) - LiDAR scan at time t
    #! Q: (3, M) - LiDAR scan at time t+1
    '''
    P_trans = (R_rel @ P) + t_rel # (3 x N)
    tree = cKDTree(Q.T)
    dist, idx = tree.query(P_trans.T, k=1) # (M x 1), (M x 1)
    Q_corr = Q.T[idx] # (M x 3)
    C = np.stack([P.T, Q_corr], axis=1) # (M x 2 x 3)
    return C

def compute_indices_dists(P, Q, R_rel, t_rel):
    '''
    #! P: (3, N) - LiDAR scan at time t
    #! Q: (3, M) - LiDAR scan at time t+1
    '''
    P_trans = (R_rel @ P) + t_rel
    tree = cKDTree(Q.T)
    dist, idx = tree.query(P_trans.T, k=1)
    return idx, dist

def visualize_point_clouds(P, Q):
    pcd_P = o3d.geometry.PointCloud()
    pcd_Q = o3d.geometry.PointCloud()

    pcd_P.points = o3d.utility.Vector3dVector(P.T)  # (N, 3) 
    pcd_Q.points = o3d.utility.Vector3dVector(Q.T)  # (M, 3) 

    pcd_P.paint_uniform_color([1, 0, 0])  
    pcd_Q.paint_uniform_color([1, 0, 0])  

    title = f"P: {P.shape[1]} points | Q: {Q.shape[1]} points"
    print(title)

    o3d.visualization.draw_geometries([pcd_P, pcd_Q], window_name=title)