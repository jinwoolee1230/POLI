import os
import sys
sys.dont_write_bytecode = True
import time
import argparse

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import covariance_generation, map_z_to_colors, visualize_normals
from visualization.vis_utils import (
    get_sample,
    get_sample_path,
    load_model,
    load_parsed_dataset,
    normalize_inputs,
)


def build_parser(default_mode="ellipsoid", include_mode=True):
    parser = argparse.ArgumentParser(
        description="Visualize parsed .npz samples with ellipsoid, augment, or normal modes"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to parsed .npz dataset directory")

    if include_mode:
        parser.add_argument(
            "--mode",
            type=str,
            choices=["ellipsoid", "augment", "normal"],
            default=default_mode,
            help="Visualization mode",
        )

    parser.add_argument("--mean", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--scaling_factor", type=float, default=100.0)

    parser.add_argument(
        "--n_std",
        type=float,
        default=None,
        help="Number of standard deviations. Defaults to 0.3 in ellipsoid mode and 0.1 in augment mode.",
    )
    parser.add_argument("--min_eig", type=float, default=1e-12, help="Used in ellipsoid mode")
    parser.add_argument(
        "--ellipsoid_resolution",
        type=int,
        default=8,
        help="Sphere resolution used to build ellipsoids. Lower is faster.",
    )

    parser.add_argument("--samples_per_point", type=int, default=300, help="Used in augment mode")
    parser.add_argument(
        "--max_render_points",
        type=int,
        default=100000,
        help="Maximum number of points sent to the renderer after sampling/filtering. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=20000,
        help="Maximum number of input points/covariances to visualize. Set <= 0 to disable.",
    )

    parser.add_argument(
        "--uncertainty_colormap",
        type=str,
        default="jet",
        help="Used in normal mode",
    )
    parser.add_argument("--normal_length", type=float, default=3.0, help="Used in normal mode")
    parser.add_argument(
        "--normal_color",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        help="Used in normal mode",
    )
    return parser


def parse_args(default_mode="ellipsoid", include_mode=True):
    parser = build_parser(default_mode=default_mode, include_mode=include_mode)
    args = parser.parse_args()
    if not include_mode:
        args.mode = default_mode
    return args


def resolve_n_std(args):
    if args.n_std is not None:
        return args.n_std
    if args.mode == "augment":
        return 0.1
    return 0.3


def infer_covariances(model, P, Q, args):
    normalized_P, _ = normalize_inputs(P, Q, args.mean, args.scaling_factor)
    with torch.no_grad():
        C = model(normalized_P)
        return covariance_generation(C.permute(0, 2, 1))


def load_first_sample(dataset, device):
    sample_idx = 0
    sample = get_sample(dataset, sample_idx, device)
    sample_path = get_sample_path(dataset, sample_idx)
    return sample, sample_path


def print_camera(vis):
    vc = vis.get_view_control()
    cam = vc.convert_to_pinhole_camera_parameters()

    np.set_printoptions(precision=6, suppress=True)

    print("\n==============================")
    print("[Camera Extrinsic]")
    print(cam.extrinsic)

    fx, fy = cam.intrinsic.get_focal_length()
    cx, cy = cam.intrinsic.get_principal_point()

    print("\n[Camera Intrinsic]")
    print(f"width={cam.intrinsic.width}, height={cam.intrinsic.height}")
    print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print("==============================\n")


def apply_camera(vis, extrinsic, intrinsic_cfg=None):
    if extrinsic is None:
        return

    vc = vis.get_view_control()
    cam = vc.convert_to_pinhole_camera_parameters()

    if intrinsic_cfg is not None:
        cam.intrinsic.set_intrinsics(
            int(intrinsic_cfg["width"]),
            int(intrinsic_cfg["height"]),
            float(intrinsic_cfg["fx"]),
            float(intrinsic_cfg["fy"]),
            float(intrinsic_cfg["cx"]),
            float(intrinsic_cfg["cy"]),
        )

    cam.extrinsic = extrinsic.copy()
    vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)


def get_camera_params(vis):
    vc = vis.get_view_control()
    return vc.convert_to_pinhole_camera_parameters()


def choose_subset_indices(num_points, max_points):
    if max_points is None or max_points <= 0 or num_points <= max_points:
        return np.arange(num_points)
    return np.linspace(0, num_points - 1, num=max_points, dtype=np.int64)


def subsample_points_and_covariances(points, covariances, max_points):
    indices = choose_subset_indices(points.shape[0], max_points)
    return points[indices], covariances[indices]


def limit_render_points(points, max_points):
    indices = choose_subset_indices(points.shape[0], max_points)
    return points[indices]


def filter_z_outliers(points, covariances=None):
    if points.shape[0] < 4:
        if covariances is None:
            return points
        return points, covariances

    z = points[:, 2]
    q1, q3 = np.percentile(z, [25.0, 75.0])
    iqr = q3 - q1

    if iqr <= 1e-8:
        if covariances is None:
            return points
        return points, covariances

    lower = q1 - 8.0 * iqr
    upper = q3 + 8.0 * iqr
    mask = (z >= lower) & (z <= upper)

    if not np.any(mask):
        if covariances is None:
            return points
        return points, covariances

    filtered_points = points[mask]
    if covariances is None:
        return filtered_points
    return filtered_points, covariances[mask]


def build_ellipsoid_geometries(points, covariances, n_std, min_eig, ellipsoid_resolution):
    colors = map_z_to_colors(points, colormap="jet")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    merged_ellipsoids = o3d.geometry.TriangleMesh()

    for idx in range(points.shape[0]):
        cov = covariances[idx]
        pos = points[idx]

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_eig)
        scales = n_std * np.sqrt(eigvals)
        if np.any(scales <= 0):
            continue

        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(
            radius=1.0,
            resolution=max(3, ellipsoid_resolution),
        )
        transform = np.eye(4)
        transform[:3, :3] = eigvecs @ np.diag(scales)
        ellipsoid.transform(transform)
        ellipsoid.translate(pos)
        ellipsoid.paint_uniform_color(colors[idx])
        merged_ellipsoids += ellipsoid

    geometries = [pcd]
    if len(merged_ellipsoids.vertices) > 0:
        geometries.append(merged_ellipsoids)
    return geometries


def sample_points_from_covariance(points, covariances, n_samples, n_std):
    sampled_points = []
    for idx in range(points.shape[0]):
        mean = points[idx]
        cov = covariances[idx]

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)
        safe_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        pts = np.random.multivariate_normal(mean, safe_cov * (n_std ** 2), n_samples)
        sampled_points.append(pts)

    return np.vstack(sampled_points)


def run_ellipsoid_mode(args, model, dataset, device):
    sample, sample_path = load_first_sample(dataset, device)
    print(f"Visualizing first sample in ellipsoid mode: {sample_path}")

    P = sample["P"]
    Q = sample["Q"]
    C = infer_covariances(model, P, Q, args)

    points = P[0].permute(1, 0).cpu().numpy()
    covariances = C[0].cpu().numpy()
    points, covariances = subsample_points_and_covariances(points, covariances, args.max_input_points)
    points, covariances = filter_z_outliers(points, covariances)
    geometries = build_ellipsoid_geometries(
        points,
        covariances,
        resolve_n_std(args),
        args.min_eig,
        args.ellipsoid_resolution,
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Covariance Ellipsoids", 1280, 800)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.0, 0.0, 0.0])

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.reset_view_point(True)
    vis.register_key_callback(ord("v"), lambda viewer: (print_camera(viewer), False)[1])
    vis.register_key_callback(ord("V"), lambda viewer: (print_camera(viewer), False)[1])

    vis.run()
    vis.destroy_window()


def run_normal_mode(args, model, dataset, device):
    sample, sample_path = load_first_sample(dataset, device)
    print(f"Visualizing first sample in normal mode: {sample_path}")

    P = sample["P"]
    Q = sample["Q"]
    C = infer_covariances(model, P, Q, args)

    points = P[0].permute(1, 0).cpu().numpy()
    covariances = C[0].cpu().numpy()
    points, covariances = subsample_points_and_covariances(points, covariances, args.max_input_points)
    points, covariances = filter_z_outliers(points, covariances)

    visualize_normals(
        points,
        covariances,
        colormap=args.uncertainty_colormap,
        show_normals=True,
        normal_length=args.normal_length,
        normal_color=tuple(args.normal_color),
    )


def run_augment_mode(args, model, dataset, device):
    window_width = 960
    window_height = 720
    n_std = resolve_n_std(args)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Gaussian Samples (q: next, v: print cam, r: reapply cam)",
        width=window_width,
        height=window_height,
    )

    render_option = vis.get_render_option()
    render_option.point_size = 0.3
    render_option.background_color = np.array([0.0, 0.0, 0.0])

    point_cloud = o3d.geometry.PointCloud()
    geometry_added = False
    next_sample = False

    init_extrinsic = np.array([
        [0.903388, -0.428701, 0.010265, -1.831778],
        [-0.201837, -0.446202, -0.871875, -3.076615],
        [0.378353, 0.785570, -0.489621, 120.292488],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    init_intrinsic = None

    def next_cb(_vis):
        nonlocal next_sample
        next_sample = True
        return False

    def print_cam_cb(_vis):
        cam = get_camera_params(_vis)

        np.set_printoptions(precision=6, suppress=True)

        print("\n==============================")
        print("[Current Camera Extrinsic]")
        print(cam.extrinsic)

        fx, fy = cam.intrinsic.get_focal_length()
        cx, cy = cam.intrinsic.get_principal_point()

        print("\n[Current Camera Intrinsic]")
        print(f"width={cam.intrinsic.width}, height={cam.intrinsic.height}")
        print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        print("\n[Copy-Paste Ready]")
        print("INIT_EXTRINSIC = np.array([")
        for row in cam.extrinsic:
            print(f"    [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}, {row[3]: .6f}],")
        print("], dtype=np.float64)\n")

        print("INIT_INTRINSIC = {")
        print(f'    "width": {cam.intrinsic.width},')
        print(f'    "height": {cam.intrinsic.height},')
        print(f'    "fx": {fx},')
        print(f'    "fy": {fy},')
        print(f'    "cx": {cx},')
        print(f'    "cy": {cy},')
        print("}")
        print("==============================\n")
        return False

    def reapply_cam_cb(_vis):
        apply_camera(_vis, init_extrinsic, init_intrinsic)
        return False

    vis.register_key_callback(ord("q"), next_cb)
    vis.register_key_callback(ord("Q"), next_cb)
    vis.register_key_callback(ord("v"), print_cam_cb)
    vis.register_key_callback(ord("V"), print_cam_cb)
    vis.register_key_callback(ord("r"), reapply_cam_cb)
    vis.register_key_callback(ord("R"), reapply_cam_cb)

    with torch.no_grad():
        for sample_idx in tqdm(range(len(dataset))):
            sample = get_sample(dataset, sample_idx, device)
            print(f"[Sample {sample_idx}] {get_sample_path(dataset, sample_idx)}")
            print("press 'q' next, 'v' print, 'r' reapply")

            P = sample["P"]
            Q = sample["Q"]
            C = infer_covariances(model, P, Q, args)

            points = P[0].permute(1, 0).cpu().numpy()
            covariances = C[0].cpu().numpy()
            points, covariances = subsample_points_and_covariances(points, covariances, args.max_input_points)
            points, covariances = filter_z_outliers(points, covariances)

            sampled_points = sample_points_from_covariance(
                points,
                covariances,
                n_samples=args.samples_per_point,
                n_std=n_std,
            )
            sampled_points = filter_z_outliers(sampled_points)
            sampled_points = limit_render_points(sampled_points, args.max_render_points)

            colors = map_z_to_colors(sampled_points, colormap="jet")

            point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            if not geometry_added:
                vis.add_geometry(point_cloud)
                geometry_added = True
            else:
                vis.update_geometry(point_cloud)

            vis.poll_events()
            vis.update_renderer()
            vis.reset_view_point(True)

            next_sample = False
            while not next_sample:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)

    vis.destroy_window()


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    dataset = load_parsed_dataset(args.dataset_dir)
    print(f"Loaded {len(dataset)} parsed samples from {args.dataset_dir}")

    if args.mode == "ellipsoid":
        run_ellipsoid_mode(args, model, dataset, device)
        return
    if args.mode == "augment":
        run_augment_mode(args, model, dataset, device)
        return
    if args.mode == "normal":
        run_normal_mode(args, model, dataset, device)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main(parse_args())
