import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d

def visualize_normals(points, covariances, colormap='turbo',
                          show_normals=False, normal_length=3.0, normal_color=(0.0, 1.0, 0.0)):
    """
    Visualize point cloud with optional normals (LineSet or cylinders).
    """
    # Normalize Z for coloring
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
    cmap = cm.get_cmap(colormap)
    colors = cmap(z_norm)[:, :3]

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis_list = [pcd]

    if show_normals:
        N = points.shape[0]
        print(f"\nGenerating normals for {N} points...")

        # Extract normals (smallest eigenvector)
        normals = []
        for i in range(N):
            cov = covariances[i]
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal_vec = eigvecs[:, 0]  # smallest eigenvalue
            if np.dot(normal_vec, points[i]) > 0:
                normal_vec = -normal_vec
            normals.append(normal_vec)
        normals = np.array(normals)

        line_set = create_normal_lines(points, normals, normal_length, normal_color)
        vis_list.append(line_set)

    print("Rendering... (Close window to proceed)")
    o3d.visualization.draw_geometries(vis_list)

def create_normal_lines(points, normals, length, color):
    """
    Fast normal visualization using LineSet.
    """
    start_points = points
    end_points = points + normals * length
    line_points = np.vstack([start_points, end_points])
    lines = [[i, i + len(points)] for i in range(len(points))]
    colors = [color for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def covariance_generation(elements):
    B= elements.shape[0]
    N= elements.shape[1]
    L = torch.zeros(B, N, 3, 3, device=elements.device, dtype=elements.dtype)

    L[:, :, 0, 0] = elements[..., 0]  # l0
    L[:, :, 1, 0] = elements[..., 3]  # l1
    L[:, :, 1, 1] = elements[..., 1]  # l2
    L[:, :, 2, 0] = elements[..., 4]  # l3
    L[:, :, 2, 1] = elements[..., 5]  # l4
    L[:, :, 2, 2] = elements[..., 2]  # l5

    covariance_matrix = torch.matmul(L, L.transpose(-2, -1))  # (B, N, 3, 3)
    return covariance_matrix


def make_loss(R, t, R_rel, t_rel):

    T_est = torch.eye(4, device=R.device).unsqueeze(0).repeat(4, 1, 1)
    T_est[:, :3, :3] = R
    T_est[:, :3, 3] = t.squeeze(-1) # [B, 4, 4] , S2T^

    T_gt = torch.eye(4, device=R.device).unsqueeze(0).repeat(4, 1, 1)
    T_gt[:, :3, :3] = R_rel
    T_gt[:, :3, 3] = t_rel.squeeze(-1) # [B, 4, 4] , S2T*

    loss = torch.norm(T_est - T_gt, p='fro', dim=(1, 2)) # [B]

    loss = loss.mean()  # 평균 손실값 계산
    return loss

def is_psd(C_p: torch.Tensor, atol=1e-6) -> bool:
    """
    C_p: (B, N, 3, 3) 또는 (N, 3, 3) 텐서
    """
    # (B, N, 3, 3) -> (-1, 3, 3)
    C_p_flat = C_p.view(-1, 3, 3)

    # 고유값 계산 (symmetric이므로 eigvalsh)
    eigvals = torch.linalg.eigvalsh(C_p_flat)  # shape: (B*N, 3)

    # 음수 고유값이 있는지 확인
    is_psd_mask = (eigvals >= -atol).all(dim=1)

    return is_psd_mask.all().item()

def scale_aware_normalization(P, Q, mean, scaling_factor):
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=P.dtype, device=P.device)
    else:
        mean = mean.to(device=P.device, dtype=P.dtype)

    if mean.ndim == 1:
        mean = mean.view(1, -1, 1)
    elif mean.ndim == 2:
        mean = mean.unsqueeze(-1)

    if mean.shape[1] != P.shape[1]:
        raise ValueError(
            f"Expected mean to have {P.shape[1]} channels, but got shape {tuple(mean.shape)}."
        )

    if not torch.is_tensor(scaling_factor):
        scaling_factor = torch.tensor(scaling_factor, dtype=P.dtype, device=P.device)
    else:
        scaling_factor = scaling_factor.to(device=P.device, dtype=P.dtype)

    normalized_P = (P - mean) / scaling_factor           # [B, 3, N]
    normalized_Q = (Q - mean) / scaling_factor           # [B, 3, M]

    return normalized_P, normalized_Q

def sigmoid_mask(dists, threshold=1.0, alpha=30):
    mask= torch.sigmoid(alpha*(threshold - dists))
    return mask

def to_homogeneous_xyz(points_xyz):
    B, N, _ = points_xyz.shape
    DEVICE = points_xyz.device
    DTYPE = points_xyz.dtype
    ones = torch.ones(B, N, 1, device=DEVICE, dtype=DTYPE)
    return torch.cat([points_xyz, ones], dim=-1).permute(0, 2, 1).contiguous()
