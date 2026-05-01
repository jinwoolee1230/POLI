import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import numpy as np
import torch

from applications.object_pose_estimation.model import PointPP
from applications.object_pose_estimation.registration_utils import (
    covariance_generation,
    compute_fpfh_batch,
    estimate_correspondences,
    load_checkpoint,
    maximum_consensus,
    normals_from_covariance,
    rotation_translation_error,
    svd_registration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="POLI + FPFH + ROBIN object pose estimation demo."
    )
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fpfh_radius", type=float, default=0.2)
    parser.add_argument("--fpfh_max_nn", type=int, default=90)
    parser.add_argument("--consensus_beta", type=float, default=0.1)
    parser.add_argument(
        "--robin_storage",
        type=str,
        default="ADJ_LIST",
        choices=("ADJ_LIST", "CSR", "ATOMIC_CSR"),
        help="ROBIN compatibility graph storage backend.",
    )
    parser.add_argument(
        "--debug_robin",
        action="store_true",
        help="Print ROBIN graph, max-core, max-clique, and PMC diagnostics.",
    )
    parser.add_argument(
        "--dump_robin_input",
        type=str,
        default=None,
        help="Save the exact ROBIN inputs so the max-clique stage can be replayed.",
    )
    parser.add_argument("--max_lines", type=int, default=1000)
    parser.add_argument("--max_normals", type=int, default=1000)
    parser.add_argument("--normal_length", type=float, default=0.15)
    parser.add_argument("--normal_radius", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7777)
    return parser.parse_args()


def load_npz_sample(path):
    sample = np.load(path)
    required = {"P", "Q", "R_rel", "t_rel"}
    missing = required.difference(sample.files)
    if missing:
        raise KeyError(f"Sample is missing keys: {sorted(missing)}")

    P = sample["P"].astype(np.float32)
    Q = sample["Q"].astype(np.float32)
    if P.shape[0] != 3 or Q.shape[0] != 3:
        raise ValueError(f"Expected P/Q shape (3,N), got P={P.shape}, Q={Q.shape}")

    return {
        "P": P,
        "Q": Q,
        "R_rel": sample["R_rel"].astype(np.float32),
        "t_rel": sample["t_rel"].astype(np.float32).reshape(3),
    }


def make_covariance_normals(model, points_b3n):
    points_centered = points_b3n - points_b3n.mean(dim=2, keepdim=True)
    cov_elements = model(points_centered)
    covariances = covariance_generation(cov_elements.permute(0, 2, 1))
    normals = normals_from_covariance(covariances)
    return covariances, normals


def make_point_cloud(points, color):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def make_output_image_paths(sample_path):
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "source_normals": output_dir / f"{sample_path.stem}_source_neural_normals.png",
        "target_normals": output_dir / f"{sample_path.stem}_target_neural_normals.png",
        "before_registration": output_dir / f"{sample_path.stem}_neural_robin_before_registration.png",
        "after_registration": output_dir / f"{sample_path.stem}_neural_robin_after_registration.png",
    }


def make_line_set(src_points, tgt_points, color):
    import open3d as o3d

    line_points = np.vstack([src_points, tgt_points])
    lines = [[i, i + len(src_points)] for i in range(len(src_points))]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def rotation_matrix_from_vectors(src_vec, dst_vec):
    src = np.asarray(src_vec, dtype=np.float64)
    dst = np.asarray(dst_vec, dtype=np.float64)
    src = src / np.maximum(np.linalg.norm(src), 1e-12)
    dst = dst / np.maximum(np.linalg.norm(dst), 1e-12)

    cross = np.cross(src, dst)
    dot = np.clip(np.dot(src, dst), -1.0, 1.0)
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.maximum(np.linalg.norm(axis), 1e-12)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def make_cylinder_segments(src_points, tgt_points, color, radius, resolution=10):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for start, end in zip(src_points, tgt_points):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        segment = end - start
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-12:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=float(radius),
            height=float(segment_length),
            resolution=int(resolution),
        )
        cylinder.rotate(rotation_matrix_from_vectors(z_axis, segment), center=(0.0, 0.0, 0.0))
        cylinder.translate((start + end) * 0.5)
        cylinder.paint_uniform_color(color)
        mesh += cylinder

    mesh.compute_vertex_normals()
    return mesh


def make_normal_line_set(points, normals, length, color, max_normals, radius):
    n_points = len(points)
    draw_idx = downsample_line_indices(n_points, max_normals)
    starts = points[draw_idx].astype(np.float64)
    normal_vecs = normals[draw_idx].astype(np.float64)
    normal_norms = np.linalg.norm(normal_vecs, axis=1, keepdims=True)
    normal_vecs = normal_vecs / np.maximum(normal_norms, 1e-12)
    ends = starts + normal_vecs * float(length)
    return make_cylinder_segments(starts, ends, color=color, radius=radius)


def orient_normals_outward(points, normals):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    outward_dirs = points - center
    flip_mask = np.sum(normals * outward_dirs, axis=1, keepdims=True) < 0.0
    return np.where(flip_mask, -normals, normals).astype(np.float32)


def downsample_line_indices(num_lines, max_lines):
    if max_lines > 0 and num_lines > max_lines:
        rng = np.random.default_rng(0)
        return rng.choice(num_lines, size=max_lines, replace=False)
    return np.arange(num_lines)


def show_and_save(geoms, image_path, window_name, point_size=9.0, line_width=6.0):
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=1280,
        height=840,
        visible=True,
    )
    for geom in geoms:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = point_size
    opt.line_width = line_width
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(image_path), do_render=True)
    print(f"[Saved image] {image_path}")
    vis.run()
    vis.destroy_window()


def visualize_normals(P_np, Q_np, normals_p_np, normals_q_np, image_paths, args):
    import open3d as o3d

    source_color = [1.0, 0.0, 1.0]
    target_color = [0.0, 0.2, 1.0]
    normal_color = [0.0, 0.9, 0.1]

    source_geoms = [
        make_point_cloud(P_np, source_color),
        make_normal_line_set(
            P_np,
            normals_p_np,
            length=args.normal_length,
            color=normal_color,
            max_normals=args.max_normals,
            radius=args.normal_radius,
        ),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
    ]
    show_and_save(
        source_geoms,
        image_paths["source_normals"],
        "Source Neural Normals: magenta=source, green=estimated normals",
        point_size=9.0,
        line_width=6.0,
    )

    target_geoms = [
        make_point_cloud(Q_np, target_color),
        make_normal_line_set(
            Q_np,
            normals_q_np,
            length=args.normal_length,
            color=normal_color,
            max_normals=args.max_normals,
            radius=args.normal_radius,
        ),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
    ]
    show_and_save(
        target_geoms,
        image_paths["target_normals"],
        "Target Neural Normals: blue=target, green=estimated normals",
        point_size=9.0,
        line_width=6.0,
    )


def visualize_result(P_np, Q_np, src_inliers, tgt_inliers, R_est, t_est, image_paths, args):
    import open3d as o3d

    source_color = [1.0, 0.0, 1.0]
    target_color = [0.0, 0.2, 1.0]
    line_color = [0.0, 0.9, 0.1]

    display_offset = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    P_before_display = P_np + display_offset.reshape(1, 3)
    src_before_display = src_inliers + display_offset.reshape(1, 3)

    before_geoms = [
        make_point_cloud(Q_np, target_color),
        make_point_cloud(P_before_display, source_color),
    ]

    n_lines = len(src_before_display)
    if n_lines > 0:
        draw_idx = downsample_line_indices(n_lines, args.max_lines)
        before_geoms.append(
            make_line_set(src_before_display[draw_idx], tgt_inliers[draw_idx], line_color)
        )

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    before_geoms.append(frame)

    show_and_save(
        before_geoms,
        image_paths["before_registration"],
        "Before Registration: magenta=source offset, blue=target, green=inlier correspondences",
    )

    P_aligned = (P_np @ R_est.T) + t_est.reshape(1, 3)
    src_after = (src_inliers @ R_est.T) + t_est.reshape(1, 3)
    after_geoms = [
        make_point_cloud(Q_np, target_color),
        make_point_cloud(P_aligned, source_color),
    ]
    if n_lines > 0:
        after_geoms.append(make_line_set(src_after[draw_idx], tgt_inliers[draw_idx], line_color))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    after_geoms.append(frame)

    show_and_save(
        after_geoms,
        image_paths["after_registration"],
        "After Registration: magenta=estimated source, blue=target, green=inlier correspondences",
    )


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample_path = Path(args.sample).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Sample] {sample_path}")
    print(f"[Checkpoint] {checkpoint_path}")

    sample = load_npz_sample(sample_path)
    P_np = sample["P"].T
    Q_np = sample["Q"].T
    R_gt = sample["R_rel"]
    t_gt = sample["t_rel"]

    P = torch.from_numpy(sample["P"]).unsqueeze(0).to(device)
    Q = torch.from_numpy(sample["Q"]).unsqueeze(0).to(device)

    model = PointPP().to(device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    with torch.no_grad():
        _, normals_p = make_covariance_normals(model, P)
        _, normals_q = make_covariance_normals(model, Q)
        fpfh_p = compute_fpfh_batch(P, normals_p, radius=args.fpfh_radius, max_nn=args.fpfh_max_nn)
        fpfh_q = compute_fpfh_batch(Q, normals_q, radius=args.fpfh_radius, max_nn=args.fpfh_max_nn)
        matches = estimate_correspondences(fpfh_p[0], fpfh_q[0]).detach().cpu().numpy()

    image_paths = make_output_image_paths(sample_path)
    normals_p_np = orient_normals_outward(P_np, normals_p[0].detach().cpu().numpy())
    normals_q_np = orient_normals_outward(Q_np, normals_q[0].detach().cpu().numpy())
    visualize_normals(P_np, Q_np, normals_p_np, normals_q_np, image_paths, args)

    print(f"[Correspondence] neural FPFH matches={len(matches)}")

    src_matched = P_np
    tgt_matched = Q_np[matches]
    if args.dump_robin_input:
        dump_path = Path(args.dump_robin_input).expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            src_matched=src_matched,
            tgt_matched=tgt_matched,
            matches=matches,
            beta=np.asarray(args.consensus_beta, dtype=np.float64),
            robin_storage=np.asarray(args.robin_storage),
            sample_path=np.asarray(str(sample_path)),
            checkpoint_path=np.asarray(str(checkpoint_path)),
        )
        print(f"[ROBIN] dumped input={dump_path}")

    src_inliers, tgt_inliers = maximum_consensus(
        src_matched,
        tgt_matched,
        beta=args.consensus_beta,
        debug=args.debug_robin,
        storage_type=args.robin_storage,
    )
    print(f"[Consensus] inliers={len(src_inliers)}/{len(src_matched)}")
    if len(src_inliers) < 3:
        raise RuntimeError("Not enough inliers to estimate pose.")

    T_est = svd_registration(src_inliers, tgt_inliers)
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]

    rot_error, trans_error = rotation_translation_error(R_est, t_est, R_gt, t_gt)
    print("\n[Estimated R]")
    print(np.array2string(R_est, precision=6, suppress_small=True))
    print("[Estimated t]")
    print(np.array2string(t_est, precision=6, suppress_small=True))
    print(f"[GT error] rotation={rot_error:.4f} deg | translation={trans_error:.6f}")

    visualize_result(P_np, Q_np, src_inliers, tgt_inliers, R_est, t_est, image_paths, args)


if __name__ == "__main__":
    main(parse_args())
