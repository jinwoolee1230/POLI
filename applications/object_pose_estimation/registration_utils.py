import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils import covariance_generation


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


def compute_fpfh(points, normal_radius, feature_radius, normal_max_nn, feature_max_nn):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for FPFH feature extraction.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    pcd.normalize_normals()
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius,
            max_nn=feature_max_nn,
        ),
    )
    fpfh = np.asarray(feat.data, dtype=np.float32).T
    return np.nan_to_num(fpfh, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_correspondences(feat_src, feat_tgt):
    """
    feat_src: [D, Ns]
    feat_tgt: [D, Nt]
    return matches: [Ns], source i -> target matches[i]
    """
    dists = torch.cdist(feat_src.T, feat_tgt.T, p=2)
    return torch.argmin(dists, dim=1)


def estimate_mutual_feature_matches(fpfh_src, fpfh_tgt):
    target_tree = cKDTree(fpfh_tgt)
    _, src_to_tgt = target_tree.query(fpfh_src, k=1, workers=-1)
    src_to_tgt = src_to_tgt.astype(np.int64)

    source_tree = cKDTree(fpfh_src)
    _, tgt_to_src = source_tree.query(fpfh_tgt, k=1, workers=-1)
    tgt_to_src = tgt_to_src.astype(np.int64)

    src_idx = np.arange(fpfh_src.shape[0], dtype=np.int64)
    mutual = tgt_to_src[src_to_tgt] == src_idx
    return src_idx[mutual], src_to_tgt[mutual]


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
        spark_robin.InlierGraphStructure.MAX_CORE,
    )

    return src[indices, :], tgt[indices, :]


def horns_method(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    U, _, Vt = np.linalg.svd(P_centered.T @ Q_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t


def weighted_horns(P, Q, weights):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    sum_w = np.sum(w)
    if sum_w <= 1e-12:
        raise ValueError("Sum of weights must be greater than zero.")

    centroid_P = np.sum(w * P, axis=0) / sum_w
    centroid_Q = np.sum(w * Q, axis=0) / sum_w
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    U, _, Vt = np.linalg.svd(P_centered.T @ (w * Q_centered))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t


def GNC_GM(P, Q, noise_bound=1.0, max_iterations=100):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    c2 = max(float(noise_bound) ** 2, 1e-12)

    R, t = horns_method(P, Q)
    r = Q - (R @ P.T).T - t
    r2 = np.sum(r**2, axis=1)
    mu = max(2.0 * float(np.max(r2)) / c2, 1.0)

    for _ in range(max_iterations):
        den = r2 + mu * c2
        weights = (mu * c2 / np.maximum(den, 1e-12)) ** 2
        R, t = weighted_horns(P, Q, weights)
        r = Q - (R @ P.T).T - t
        r2 = np.sum(r**2, axis=1)
        mu = mu / 1.1
        if mu < 1.0:
            break
    return R, t


def GNC_TLS(P, Q, noise_bound=1.0, max_iterations=100):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    c_bar = float(noise_bound)
    c2 = max(c_bar * c_bar, 1e-12)

    R, t = horns_method(P, Q)
    r = Q - (R @ P.T).T - t
    r2 = np.sum(r**2, axis=1)
    r2max = float(np.max(r2))
    denom = 2.0 * r2max - c2
    mu = c2 / denom if abs(denom) > 1e-12 else 1.0
    mu = max(mu, 1e-12)

    R_old = np.eye(3)
    t_old = np.zeros(3)
    for _ in range(max_iterations):
        th1 = (mu / (mu + 1.0)) * c2
        th2 = ((mu + 1.0) / mu) * c2
        mid = (c_bar / (np.sqrt(r2) + 1e-9)) * np.sqrt(mu * (mu + 1.0)) - mu
        weights = np.where(r2 < th1, 1.0, np.where(r2 > th2, 0.0, mid))

        R, t = weighted_horns(P, Q, weights)
        r = Q - (R @ P.T).T - t
        r2 = np.sum(r**2, axis=1)
        mu = mu * 1.1

        if np.linalg.norm(R - R_old) < 1e-6 and np.linalg.norm(t - t_old) < 1e-6:
            break
        R_old, t_old = R, t
    return R, t


def gnc_registration(P, Q, solver, noise_bound):
    if solver == "gm":
        return GNC_GM(P, Q, noise_bound=noise_bound)
    if solver == "tls":
        return GNC_TLS(P, Q, noise_bound=noise_bound)
    raise ValueError(f"Unknown GNC solver: {solver}")


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
