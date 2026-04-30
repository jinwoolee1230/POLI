import math

import numpy as np
import torch


def covariance_generation(elements):
    """
    elements: [B, N, 6]
    return: [B, N, 3, 3]
    """
    B, N, _ = elements.shape
    L = torch.zeros(B, N, 3, 3, device=elements.device, dtype=elements.dtype)
    L[:, :, 0, 0] = elements[..., 0]
    L[:, :, 1, 0] = elements[..., 3]
    L[:, :, 1, 1] = elements[..., 1]
    L[:, :, 2, 0] = elements[..., 4]
    L[:, :, 2, 1] = elements[..., 5]
    L[:, :, 2, 2] = elements[..., 2]
    return L @ L.transpose(-2, -1)


def normals_from_covariance(covariance):
    eig_vals, eig_vecs = torch.linalg.eigh(covariance)
    return eig_vecs[..., 0]


def compute_fpfh_batch(points_b3n, normals_bn3, radius=0.3, max_nn=90):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for FPFH feature extraction.") from exc

    batch_size = points_b3n.shape[0]
    points_np = points_b3n.transpose(1, 2).detach().cpu().numpy()
    normals_np = normals_bn3.detach().cpu().numpy()
    fpfh_list = []

    for i in range(batch_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np[i].astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals_np[i].astype(np.float64))
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)),
        )
        fpfh_list.append(np.asarray(feat.data, dtype=np.float32))

    return torch.tensor(np.stack(fpfh_list, axis=0), dtype=torch.float32, device=points_b3n.device)


def estimate_correspondences(feat_src, feat_tgt):
    """
    feat_src: [D, Ns]
    feat_tgt: [D, Nt]
    return matches: [Ns], source i -> target matches[i]
    """
    dists = torch.cdist(feat_src.T, feat_tgt.T, p=2)
    return torch.argmin(dists, dim=1)


def svd_registration(src_matched, tgt_matched):
    src = np.asarray(src_matched, dtype=np.float64)
    tgt = np.asarray(tgt_matched, dtype=np.float64)
    if src.shape != tgt.shape:
        raise ValueError(f"src/tgt shape mismatch: {src.shape} vs {tgt.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required.")

    src_centroid = np.mean(src, axis=0)
    tgt_centroid = np.mean(tgt, axis=0)
    src_demean = src - src_centroid
    tgt_demean = tgt - tgt_centroid

    H = src_demean.T @ tgt_demean
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tgt_centroid - R @ src_centroid

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def maximum_consensus(src_matched, tgt_matched, beta=0.1):
    """
    ROBIN maximum clique consensus.
    """
    src = np.asarray(src_matched, dtype=np.float64)
    tgt = np.asarray(tgt_matched, dtype=np.float64)

    import spark_robin

    graph = spark_robin.Make3dRegInvGraph(
        src.T,
        tgt.T,
        float(beta),
        spark_robin.GraphStorageType.ADJ_LIST,
    )
    indices = spark_robin.FindInlierStructure(
        graph,
        spark_robin.InlierGraphStructure.MAX_CLIQUE,
    )
    return src[indices, :], tgt[indices, :]


def rotation_translation_error(R_est, t_est, R_gt, t_gt):
    R_diff = R_est.T @ R_gt
    cos_theta = (np.trace(R_diff) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_error = math.degrees(math.acos(cos_theta))
    trans_error = float(np.linalg.norm(t_est.reshape(3) - t_gt.reshape(3)))
    return rot_error, trans_error


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")
    return checkpoint
