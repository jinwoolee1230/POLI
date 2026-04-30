import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make object registration samples from one or more PLY point clouds."
    )
    parser.add_argument(
        "--object_ply",
        dest="ply_paths",
        type=str,
        nargs="+",
        required=True,
        help="Input PLY paths.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Directory where sample_*.npz files will be saved.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points sampled for both P and Q.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of P/Q pairs to generate per object.",
    )
    parser.add_argument(
        "--translation_range",
        type=float,
        default=0.0,
        help="Uniform translation range applied independently to x, y, and z.",
    )
    parser.add_argument(
        "--rotation_max_deg",
        type=float,
        default=30.0,
        help="Maximum absolute random rotation angle in degrees.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Target radius after centering and normalizing each object.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_points <= 0:
        raise ValueError("--num_points must be positive.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")
    if args.translation_range < 0:
        raise ValueError("--translation_range must be non-negative.")
    if args.rotation_max_deg < 0:
        raise ValueError("--rotation_max_deg must be non-negative.")


def resolve_ply_paths(ply_paths):
    resolved = []
    for path in ply_paths:
        resolved.append(Path(path).expanduser().resolve())
    return resolved


def load_normalized_points(ply_path, scale):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required to read PLY files. Install it in this environment first."
        ) from exc

    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    center = np.mean(pts, axis=0, keepdims=True)
    pts = pts - center

    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist <= 0:
        raise ValueError(f"Degenerate point cloud: {ply_path}")

    return (pts / max_dist * float(scale)).astype(np.float32)


def axis_angle_to_rotmat(axis, angle):
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    I = np.eye(3, dtype=np.float32)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def sample_random_transform(rng, translation_range=0.0, rotation_max_deg=30.0):
    axis = rng.normal(size=3).astype(np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / axis_norm

    max_rad = float(rotation_max_deg) * np.pi / 180.0
    angle = rng.uniform(-max_rad, max_rad)
    R = axis_angle_to_rotmat(axis, angle).astype(np.float32)

    t = rng.uniform(
        -float(translation_range),
        float(translation_range),
        size=3,
    ).astype(np.float32)
    return R, t.reshape(3, 1)


def apply_transform(points_nx3, R, t_3x1):
    return (points_nx3 @ R.T) + t_3x1.reshape(1, 3)


def compute_indices_dists(P_3xN, Q_3xM, R_rel, t_rel):
    P_trans = (R_rel @ P_3xN) + t_rel
    tree = cKDTree(Q_3xM.T)
    dists, indices = tree.query(P_trans.T, k=1)
    return indices.astype(np.int64), dists.astype(np.float32)


def object_name_from_path(path):
    stem = Path(path).stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def validate_object_size(n_total, ply_path, num_points):
    if num_points > n_total:
        raise ValueError(f"num_points={num_points} > total points={n_total}: {ply_path}")


def build_output_path(output_folder, sample_index, object_name):
    return output_folder / f"sample_{sample_index:06d}_{object_name}.npz"


def create_sample(pts, num_points, rng, translation_range, rotation_max_deg):
    n_total = pts.shape[0]
    sel_P = rng.choice(n_total, size=num_points, replace=False)
    sel_Q = rng.choice(n_total, size=num_points, replace=False)

    P_init = pts[sel_P]
    Q_init = pts[sel_Q]

    R_p, t_p = sample_random_transform(rng, translation_range, rotation_max_deg)
    R_q, t_q = sample_random_transform(rng, translation_range, rotation_max_deg)

    P = apply_transform(P_init, R_p, t_p).astype(np.float32)
    Q = apply_transform(Q_init, R_q, t_q).astype(np.float32)

    R_rel = (R_q @ R_p.T).astype(np.float32)
    t_rel = (t_q - (R_rel @ t_p)).astype(np.float32)

    P_3xN = P.T.astype(np.float32)
    Q_3xM = Q.T.astype(np.float32)
    indices, dists = compute_indices_dists(P_3xN, Q_3xM, R_rel, t_rel)

    return {
        "P": P_3xN,
        "Q": Q_3xM,
        "R_rel": R_rel,
        "t_rel": t_rel,
        "indices": indices,
        "dists": dists,
    }


def make_samples_for_object(pts, ply_path, output_folder, sample_offset, args):
    validate_object_size(
        n_total=pts.shape[0],
        ply_path=ply_path,
        num_points=args.num_points,
    )

    object_name = object_name_from_path(ply_path)
    saved = 0
    for local_idx in range(args.num_samples):
        global_idx = sample_offset + local_idx
        out_name = build_output_path(output_folder, global_idx, object_name)

        rng = np.random.default_rng(global_idx * 1009)
        sample = create_sample(
            pts=pts,
            num_points=args.num_points,
            rng=rng,
            translation_range=args.translation_range,
            rotation_max_deg=args.rotation_max_deg,
        )
        np.savez(out_name, **sample)
        saved += 1

        if saved == 1 or saved % 100 == 0:
            print(f"[Saved] {out_name}")

    return saved


def compute_sample_offset(object_index, num_samples):
    return object_index * num_samples


def main(args):
    validate_args(args)

    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    ply_paths = resolve_ply_paths(args.ply_paths)
    print("[Output]", output_folder)
    print("[Objects]")
    for p in ply_paths:
        print("  -", p)

    total_saved = 0
    for object_index, ply_path in enumerate(ply_paths):
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found: {ply_path}")

        print(f"\n[Load] {ply_path}")
        pts = load_normalized_points(ply_path, args.scale)

        sample_offset = compute_sample_offset(
            object_index=object_index,
            num_samples=args.num_samples,
        )
        print(
            f"\n[Make] object={ply_path.name} "
            f"num_points={args.num_points} samples={args.num_samples}"
        )
        saved = make_samples_for_object(
            pts=pts,
            ply_path=ply_path,
            output_folder=output_folder,
            sample_offset=sample_offset,
            args=args,
        )
        total_saved += saved

    print(f"\nDone. Saved {total_saved} sample(s) in: {output_folder}")


if __name__ == "__main__":
    main(parse_args())
