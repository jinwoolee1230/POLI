import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_OBJECTS = [
    "data/bunny/reconstruction/bun_zipper.ply",
    "data/Armadillo_scans/Armadillo.ply",
    "data/lucy/lucy.ply",
    "data/asian_statuette/xyzrgb_statuette.ply",
    "data/asian_dragon/xyzrgb_dragon.ply",
    "data/happy_recon/happy_vrip.ply",
    "data/dragon_recon/dragon_vrip.ply"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="./",
        help="Root folder containing object ply files.",
    )
    parser.add_argument(
        "--ply_paths",
        type=str,
        nargs="*",
        default=None,
        help=(
            "PLY paths to use. Each path may be absolute or relative to input_root. "
            "If omitted, the default training objects from train_object.py are used."
        ),
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where sample_*.npz files will be saved.",
    )
    parser.add_argument("--P_num_points", type=int, default=1000)
    parser.add_argument(
        "--Q_num_points",
        type=int,
        nargs="+",
        default=[500, 1000],
        help="One or more Q point counts. A separate sample set is generated for each.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of P/Q pairs to generate per object per Q_num_points value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_range", type=float, default=0.0)
    parser.add_argument("--rot_max_deg", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First numeric index used in saved sample filenames.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an output npz if the target filename already exists.",
    )
    return parser.parse_args()


def resolve_ply_paths(input_root, ply_paths):
    root = Path(input_root).expanduser().resolve()
    paths = ply_paths if ply_paths else DEFAULT_OBJECTS

    resolved = []
    for path in paths:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = root / p
        resolved.append(p.resolve())
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


def sample_random_transform(rng, t_range=0.0, rot_max_deg=30.0):
    axis = rng.normal(size=3).astype(np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / axis_norm

    max_rad = float(rot_max_deg) * np.pi / 180.0
    angle = rng.uniform(-max_rad, max_rad)
    R = axis_angle_to_rotmat(axis, angle).astype(np.float32)

    t = rng.uniform(-float(t_range), float(t_range), size=3).astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return R, t.reshape(3, 1), T


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


def make_samples_for_object(ply_path, q_num_points, output_folder, sample_offset, args):
    pts = load_normalized_points(ply_path, args.scale)
    n_total = pts.shape[0]

    if args.P_num_points > n_total:
        raise ValueError(f"P_num_points={args.P_num_points} > total points={n_total}: {ply_path}")
    if q_num_points > n_total:
        raise ValueError(f"Q_num_points={q_num_points} > total points={n_total}: {ply_path}")

    object_name = object_name_from_path(ply_path)
    base_seed = int(args.seed) + sample_offset * 1009 + int(q_num_points) * 9173
    rng = np.random.default_rng(base_seed)

    saved = 0
    for local_idx in range(args.num_samples):
        global_idx = sample_offset + local_idx
        out_name = output_folder / f"sample_{global_idx:06d}_{object_name}_q{q_num_points}.npz"

        if args.skip_existing and out_name.exists():
            print("[Skip]", out_name)
            continue

        sel_P = rng.choice(n_total, size=args.P_num_points, replace=False)
        sel_Q = rng.choice(n_total, size=q_num_points, replace=False)

        P_init = pts[sel_P]
        Q_init = pts[sel_Q]

        _, _, T_p = sample_random_transform(rng, args.t_range, args.rot_max_deg)
        _, _, T_q = sample_random_transform(rng, args.t_range, args.rot_max_deg)

        P = apply_transform(P_init, T_p[:3, :3], T_p[:3, 3:4]).astype(np.float32)
        Q = apply_transform(Q_init, T_q[:3, :3], T_q[:3, 3:4]).astype(np.float32)

        T_rel = T_q @ np.linalg.inv(T_p)
        R_rel = T_rel[:3, :3].astype(np.float32)
        t_rel = T_rel[:3, 3:4].astype(np.float32)

        P_3xN = P.T.astype(np.float32)
        Q_3xM = Q.T.astype(np.float32)
        indices, dists = compute_indices_dists(P_3xN, Q_3xM, R_rel, t_rel)

        np.savez(
            out_name,
            P=P_3xN,
            Q=Q_3xM,
            R_rel=R_rel,
            t_rel=t_rel,
            indices=indices,
            dists=dists,
        )
        saved += 1

        if saved == 1 or saved % 100 == 0:
            print(f"[Saved] {out_name}")

    return args.num_samples


def main(args):
    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    ply_paths = resolve_ply_paths(args.input_root, args.ply_paths)
    print("[Output]", output_folder)
    print("[Objects]")
    for p in ply_paths:
        print("  -", p)

    sample_offset = int(args.start_index)
    for q_num_points in args.Q_num_points:
        for ply_path in ply_paths:
            if not ply_path.exists():
                raise FileNotFoundError(f"PLY not found: {ply_path}")

            print(
                f"\n[Make] object={ply_path.name} "
                f"P={args.P_num_points} Q={q_num_points} samples={args.num_samples}"
            )
            produced = make_samples_for_object(
                ply_path=ply_path,
                q_num_points=int(q_num_points),
                output_folder=output_folder,
                sample_offset=sample_offset,
                args=args,
            )
            sample_offset += produced

    print(f"\nDone. Wrote samples in: {output_folder}")


if __name__ == "__main__":
    main(parse_args())
