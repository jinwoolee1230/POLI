#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pointnetpp_core import PointPP


NORMALIZATION_MEAN = [-0.6547287702560425, 0.36115387082099915, 0.7911769151687622]
SCALING_FACTOR = 100.0


def covariance_generation(elements):
    """
    Cholesky compositions.
    elements # [1, N, 6]
    """
    B, N, _ = elements.shape
    L = torch.zeros(B, N, 3, 3, device=elements.device, dtype=elements.dtype) # [B, N, 3, 3]

    L[:, :, 0, 0] = elements[..., 0]
    L[:, :, 1, 0] = elements[..., 3]
    L[:, :, 1, 1] = elements[..., 1]
    L[:, :, 2, 0] = elements[..., 4]
    L[:, :, 2, 1] = elements[..., 5]
    L[:, :, 2, 2] = elements[..., 2]

    return L @ L.transpose(-2, -1) # [1, N, 3, 3]


def parse_args():
    parser = argparse.ArgumentParser(description="POLI densification -> FPFH -> ROBIN -> GNC global registration")
    parser.add_argument("--sample", dest="sample_flag", default=None, help="Input .npz pair")
    parser.add_argument("--checkpoint", default="/home/jiwoo/Desktop/POLI/weights/HeLiPR/vlp_helipr_0.2m.pth", help="POLI checkpoint")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--samples-per-point", type=int, default=100)
    parser.add_argument(
        "--n-std",
        type=float,
        default=0.5,
        help="Mahalanobis sigma radius for accepted samples; 1.0 is the 1-sigma ellipsoid, about 1.88 keeps 68% in 3D",
    )
    parser.add_argument("--fpfh-normal-radius", type=float, default=3.0)
    parser.add_argument("--fpfh-feature-radius", type=float, default=5.0)
    parser.add_argument("--fpfh-normal-max-nn", type=int, default=30)
    parser.add_argument("--fpfh-feature-max-nn", type=int, default=90)
    parser.add_argument("--robin-beta", type=float, default=1.0)
    parser.add_argument("--gnc-solver", choices=("tls", "gm"), default="tls")
    parser.add_argument("--gnc-noise-bound", type=float, default=None)
    parser.add_argument("--max-vis-correspondences", type=int, default=300)
    return parser.parse_args()


def ensure_3xn(points, name):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={points.shape}")
    if points.shape[0] == 3:
        return points
    if points.shape[1] == 3:
        return points.T
    raise ValueError(f"{name} must be [3,N] or [N,3], got shape={points.shape}")


def load_pair_npz(path):
    data = np.load(path, allow_pickle=True)

    if {"P_a", "P_b", "R_ab", "t_ab"}.issubset(data.files):
        P = ensure_3xn(data["P_a"], "P_a")
        Q = ensure_3xn(data["P_b"], "P_b")
        R_gt = data["R_ab"].astype(np.float32)
        t_gt = data["t_ab"].astype(np.float32).reshape(3)
    elif {"P", "Q", "R_rel", "t_rel"}.issubset(data.files):
        P = ensure_3xn(data["P"], "P")
        Q = ensure_3xn(data["Q"], "Q")
        R_gt = data["R_rel"].astype(np.float32)
        t_gt = data["t_rel"].astype(np.float32).reshape(3)
    else:
        raise KeyError(
            "Expected either keys {P_a,P_b,R_ab,t_ab} or {P,Q,R_rel,t_rel}. "
            f"Found keys: {list(data.files)}"
        )

    return P, Q, R_gt, t_gt


def load_poli_model(checkpoint_path, device):
    model = PointPP().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def normalize_points(P_b3n, Q_b3m):
    """
    Normalize point clouds.
    P_b3n # [1, 3, N]
    Q_b3m # [1, 3, M]
    """
    mean = torch.tensor(NORMALIZATION_MEAN, dtype=P_b3n.dtype, device=P_b3n.device).view(1, 3, 1) # [1, 3, 1]
    return (P_b3n - mean) / SCALING_FACTOR, (Q_b3m - mean) / SCALING_FACTOR


@torch.no_grad()
def predict_covariances(model, P_3xn, Q_3xm, device):
    """
    Predict covariances for point clouds.
    P_3xn # [3, N]
    Q_3xm # [3, M]
    """
    P = torch.from_numpy(P_3xn).unsqueeze(0).to(device=device, dtype=torch.float32) # [1, 3, N]
    Q = torch.from_numpy(Q_3xm).unsqueeze(0).to(device=device, dtype=torch.float32) # [1, 3, M]
    P_norm, Q_norm = normalize_points(P, Q) # [1, 3, N], [1, 3, M]

    C_p = covariance_generation(model(P_norm).permute(0, 2, 1))[0] # [N, 3, 3]
    C_q = covariance_generation(model(Q_norm).permute(0, 2, 1))[0] # [M, 3, 3]

    return P[0].T.contiguous(), Q[0].T.contiguous(), C_p, C_q
    # [N, 3], [M, 3], [N, 3, 3], [M, 3, 3]


@torch.no_grad()
def densify_points(points_n3, covariances_n33, samples_per_point, n_std):
    """
    Densify points using Gaussian sampling.
    points_n3 # [N, 3]
    covariances_n33 # [N, 3, 3]
    """
    device = points_n3.device
    eps = 1e-6
    C = covariances_n33 + torch.eye(3, device=device).unsqueeze(0) * eps

    try:
        L = torch.linalg.cholesky(C) # [N, 3, 3]
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals = torch.clamp(eigvals, min=eps)
        L = eigvecs @ torch.diag_embed(torch.sqrt(eigvals))

    z = torch.randn(
        points_n3.shape[0],
        samples_per_point,
        3,
        device=device,
        dtype=points_n3.dtype,
    ) # [N, samples_per_point, 3]
    keep = torch.linalg.norm(z, dim=-1) <= n_std # [N, samples_per_point]
    delta = z @ L.transpose(1, 2) # [N, samples_per_point, 3]
    sampled = (points_n3.unsqueeze(1) + delta)[keep] # [K, 3]
    sampled = torch.cat([points_n3, sampled], dim=0) # [N + K, 3]
    return sampled.detach().cpu().numpy().astype(np.float32)


def make_o3d_pcd(points):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for FPFH and visualization. Install open3d first.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def colored_pcd(points, color):
    pcd = make_o3d_pcd(points)
    pcd.paint_uniform_color(color)
    return pcd


def draw_pretty_geometries(geometries, window_name, point_size=2.0, line_width=2.0):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for visualization. Install open3d first.") from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.015, 0.015, 0.018])
    render_option.point_size = point_size
    if hasattr(render_option, "line_width"):
        render_option.line_width = line_width
    if hasattr(render_option, "light_on"):
        render_option.light_on = True

    vis.run()
    vis.destroy_window()


def vis_colors():
    return {
        "source": [1.0, 0.0, 0.85],
        "target": [0.0, 0.25, 1.0],
        "correspondence": [0.0, 1.0, 0.25],
    }


def coordinate_frame_for(points):
    import open3d as o3d

    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    frame_size = max(float(extent) * 0.03, 1.0)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)


def visualize_pair(source_points, target_points, title, point_size=2.0, separate=False):
    colors = vis_colors()
    if separate:
        offset = visual_separation_offset(source_points, target_points)
        source_points, target_points = separated_for_visualization(
            source_points,
            target_points,
            offset,
        )

    all_points = np.vstack([source_points, target_points])
    geometries = [
        colored_pcd(source_points, colors["source"]),
        colored_pcd(target_points, colors["target"]),
        coordinate_frame_for(all_points),
    ]
    draw_pretty_geometries(geometries, title, point_size=point_size)


def visualize_densification(P_raw, Q_raw, P_dense, Q_dense):
    visualize_pair(
        P_raw,
        Q_raw,
        "Before densification: separated source(magenta), target(blue)",
        point_size=2.5,
        separate=True,
    )
    visualize_pair(
        P_dense,
        Q_dense,
        "After densification: separated source(magenta), target(blue)",
        point_size=1.6,
        separate=True,
    )


def visual_separation_offset(source_points, target_points):
    all_points = np.vstack([source_points, target_points])
    extent = all_points.max(axis=0) - all_points.min(axis=0)
    x_gap = max(float(extent[0]) * 0.75, float(np.linalg.norm(extent)) * 0.25, 1.0)
    return np.array([x_gap, 0.0, 0.0], dtype=np.float32)


def separated_for_visualization(source_points, target_points, offset):
    return source_points - offset, target_points + offset


def correspondence_subset(src_corr, tgt_corr, max_count, seed):
    if src_corr.shape[0] <= max_count:
        return src_corr, tgt_corr

    rng = np.random.default_rng(seed)
    chosen = rng.choice(src_corr.shape[0], size=max_count, replace=False)
    return src_corr[chosen], tgt_corr[chosen]


def make_line_set(start_points, end_points, color):
    import open3d as o3d

    line_points = np.vstack([start_points, end_points]).astype(np.float64)
    count = start_points.shape[0]
    lines = np.column_stack([np.arange(count), np.arange(count) + count]).astype(np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (count, 1)))
    return line_set


def visualize_correspondences(
    source_points,
    target_points,
    src_corr,
    tgt_corr,
    max_correspondences,
    seed,
):
    colors = vis_colors()
    offset = visual_separation_offset(source_points, target_points)
    source_vis, target_vis = separated_for_visualization(source_points, target_points, offset)
    src_corr_vis, tgt_corr_vis = separated_for_visualization(src_corr, tgt_corr, offset)
    src_corr_vis, tgt_corr_vis = correspondence_subset(
        src_corr_vis,
        tgt_corr_vis,
        max_correspondences,
        seed,
    )

    all_points = np.vstack([source_vis, target_vis])
    geometries = [
        colored_pcd(source_vis, colors["source"]),
        colored_pcd(target_vis, colors["target"]),
        make_line_set(src_corr_vis, tgt_corr_vis, colors["correspondence"]),
        coordinate_frame_for(all_points),
    ]
    draw_pretty_geometries(
        geometries,
        "ROBIN correspondences: source(magenta), target(blue), matches(green)",
        point_size=1.4,
        line_width=2.5,
    )


def transform_points(points, R, t):
    return (R @ points.T).T + t.reshape(1, 3)


def visualize_registration(source_points, target_points, R_est, t_est):
    source_aligned = transform_points(source_points, R_est, t_est)
    visualize_pair(
        source_aligned,
        target_points,
        "Registration result: transformed source(magenta), target(blue)",
        point_size=1.6,
    )


def compute_fpfh(points, normal_radius, feature_radius, normal_max_nn, feature_max_nn):
    """
    FPFH
    points # [N', 3]
    """

    pcd = make_o3d_pcd(points)
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

def robin_consensus(src_matched, tgt_matched, beta):
    import spark_robin

    graph = spark_robin.Make3dRegInvGraph(
        src_matched.T,
        tgt_matched.T,
        beta,
        spark_robin.GraphStorageType.ADJ_LIST,
    )
    indices = spark_robin.FindInlierStructure(
        graph,
        spark_robin.InlierGraphStructure.MAX_CORE,
    )
    return src_matched[indices], tgt_matched[indices]


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


def svd_registration(src_matched, tgt_matched):
    if src_matched.shape[0] < 3:
        raise RuntimeError("At least 3 inlier correspondences are required.")

    src_centroid = src_matched.mean(axis=0)
    tgt_centroid = tgt_matched.mean(axis=0)
    X = src_matched - src_centroid
    Y = tgt_matched - tgt_centroid

    U, _, Vt = np.linalg.svd(X.T @ Y)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1] *= -1
        R_est = Vt.T @ U.T
    t_est = tgt_centroid - R_est @ src_centroid
    return R_est.astype(np.float64), t_est.astype(np.float64)


def rotation_translation_error(R_est, t_est, R_gt, t_gt):
    R_diff = R_est.T @ R_gt
    cos_theta = (np.trace(R_diff) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_error = math.degrees(math.acos(cos_theta))
    trans_error = float(np.linalg.norm(t_est.reshape(3) - t_gt.reshape(3)))
    return rot_error, trans_error


def main():
    """
    Load arguments.
    """
    args = parse_args()
    sample_arg = args.sample_flag or args.sample

    """
    Seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    """
    Load data and model.
    """
    sample_path = Path(sample_arg).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    P_3xn, Q_3xm, R_gt, t_gt = load_pair_npz(sample_path) # [3, N], [3, M], [3, 3], [3,]
    model = load_poli_model(str(checkpoint_path), device)

    P_n3, Q_m3, C_p, C_q = predict_covariances(
        model,
        P_3xn,
        Q_3xm,
        device,
    ) # [N, 3], [M, 3], [N, 3, 3], [M, 3, 3]

    P_dense = densify_points(
        P_n3,
        C_p,
        args.samples_per_point,
        args.n_std,
    ) # [N', 3]

    Q_dense = densify_points(
        Q_m3,
        C_q,
        args.samples_per_point,
        args.n_std,
    ) # [M', 3]

    visualize_densification(
        P_n3.detach().cpu().numpy(),
        Q_m3.detach().cpu().numpy(),
        P_dense,
        Q_dense,
    )

    fpfh_p = compute_fpfh(
        P_dense,
        args.fpfh_normal_radius,
        args.fpfh_feature_radius,
        args.fpfh_normal_max_nn,
        args.fpfh_feature_max_nn,
    ) # [N', 33]

    fpfh_q = compute_fpfh(
        Q_dense,
        args.fpfh_normal_radius,
        args.fpfh_feature_radius,
        args.fpfh_normal_max_nn,
        args.fpfh_feature_max_nn,
    ) # [M', 33]

    src_idx, tgt_idx = estimate_mutual_feature_matches(fpfh_p, fpfh_q) # [K], [K]
    
    src_matched = P_dense[src_idx]                     # [K, 3]
    tgt_matched = Q_dense[tgt_idx]                     # [K, 3]

    src_inliers, tgt_inliers = robin_consensus(src_matched, tgt_matched, args.robin_beta)
    
    visualize_correspondences(
        P_dense,
        Q_dense,
        src_inliers,
        tgt_inliers,
        args.max_vis_correspondences,
        args.seed,
    )

    gnc_noise_bound = args.gnc_noise_bound if args.gnc_noise_bound is not None else args.robin_beta
    R_est, t_est = gnc_registration(
        src_inliers,
        tgt_inliers,
        solver=args.gnc_solver,
        noise_bound=gnc_noise_bound,
    )
    print(f"[GNC] solver={args.gnc_solver} | noise_bound={gnc_noise_bound}")
    rot_error, trans_error = rotation_translation_error(R_est, t_est, R_gt, t_gt)

    visualize_registration(P_dense, Q_dense, R_est, t_est)

    print("\n[Estimated R]")
    print(np.array2string(R_est, precision=6, suppress_small=True))
    print("[Estimated t]")
    print(np.array2string(t_est, precision=6, suppress_small=True))
    print(f"[GT error] rotation={rot_error:.4f} deg | translation={trans_error:.6f}")


if __name__ == "__main__":
    main()
