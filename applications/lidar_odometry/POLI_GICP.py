import argparse
from collections import deque
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


from data_preprocess.pcd_data_class import PCD_Dataset
from model.pointnetpp_core import PointPP
from utils import covariance_generation, scale_aware_normalization


VISUALIZE_POINT_STRIDE = 1
VISUALIZE_HISTORY_SIZE = 5
VISUALIZE_POINT_SIZE = 0.1
VISUALIZE_SHOW_COVARIANCES = True
VISUALIZE_COV_STRIDE = 1
VISUALIZE_COV_SCALE = 0.01


def load_icp_solver():
    try:
        from applications.lidar_odometry.icp_family import icp_generalized_gauss_newton
    except ImportError as exc:
        try:
            from icp_family import icp_generalized_gauss_newton
        except ImportError:
            raise ImportError(
                "Could not import `icp_generalized_gauss_newton` from "
                "`applications/lidar_odometry/icp_family.py`. Make sure that "
                "module and its dependencies are available before running POLI_GICP."
            ) from exc
    return icp_generalized_gauss_newton


def load_iridescence():
    try:
        import pyridescence.guik as guik
        import pyridescence.glk as glk
    except ImportError as exc:
        raise ImportError(
            "Visualization requires `pyridescence`. Install it in the environment "
            "you use to run POLI_GICP."
        ) from exc
    return guik, glk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run POLI-GICP on a single parsed odometry dataset directory"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to a parsed odometry dataset directory containing .npz files",
    )
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Normalization mean used before covariance inference.",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=100.0,
        help="Normalization scaling factor used before covariance inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run the model on.",
    )
    return parser.parse_args()


def dataset_name_from_path(dataset_dir):
    return os.path.basename(os.path.normpath(dataset_dir))


def relative_pose_to_matrix(R_rel, t_rel):
    T_rel = torch.eye(4, dtype=R_rel.dtype, device=R_rel.device)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = t_rel.squeeze(-1)
    return T_rel


def transform_points(points, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3:]
    return rotation @ points + translation


def point_tensor_to_numpy(points, stride=1):
    stride = max(1, int(stride))
    return points[:, ::stride].transpose(0, 1).contiguous().detach().cpu().numpy().astype(np.float32)


def covariance_tensor_to_numpy(covariances, stride=1):
    stride = max(1, int(stride))
    return covariances[::stride].contiguous().detach().cpu().numpy().astype(np.float32)


def positions_to_numpy(positions):
    if not positions:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(positions).astype(np.float32)


def make_color_array(num_points, color):
    if num_points <= 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.repeat(np.asarray(color, dtype=np.float32)[None, :], num_points, axis=0)


class POLIGICPViewer:
    def __init__(self, dataset_name, point_stride, history_size, point_size, show_covariances, cov_stride, cov_scale):
        self.guik, self.glk = load_iridescence()
        self.viewer = self.guik.LightViewer.instance(title=f"POLI-GICP | {dataset_name}")
        self.viewer.set_draw_xy_grid(True)
        self.viewer.set_point_shape(point_size=point_size, metric=True, circle=True)
        self.viewer.use_orbit_camera_control(distance=80.0, theta=0.35, phi=-1.0)

        self.dataset_name = dataset_name
        self.point_stride = max(1, int(point_stride))
        self.history_size = max(1, int(history_size))
        self.point_size = float(point_size)
        self.show_covariances = bool(show_covariances)
        self.cov_stride = max(1, int(cov_stride))
        self.cov_scale = float(cov_scale)
        self.target_history = deque(maxlen=self.history_size)
        self.est_positions = []
        self.gt_positions = []
        self.is_closed = False
        self.map_shader = self._build_vertex_shader()
        self.target_shader = self._build_flat_shader((0.20, 0.75, 1.00), alpha=1.0)
        self.source_shader = self._build_flat_shader((1.00, 0.58, 0.16), alpha=1.0)
        self.est_shader = self._build_flat_shader((1.00, 0.32, 0.24), alpha=1.0)
        self.gt_shader = self._build_flat_shader((0.18, 0.95, 0.52), alpha=1.0)
        self.source_cov_shader = self._build_flat_shader((1.00, 0.58, 0.16), alpha=0.22).make_transparent()
        self.coord_shader = self.guik.VertexColor()

    def _build_flat_shader(self, color, alpha=1.0):
        shader = self.guik.FlatColor(float(color[0]), float(color[1]), float(color[2]), float(alpha))
        shader.set_point_shape_circle()
        shader.set_point_scale_metric()
        shader.set_point_size(self.point_size)
        return shader

    def _build_vertex_shader(self):
        shader = self.guik.VertexColor()
        shader.set_point_shape_circle()
        shader.set_point_scale_metric()
        shader.set_point_size(self.point_size)
        return shader

    def _world_points(self, local_points, world_transform):
        return transform_points(local_points, world_transform)

    def _world_covariances(self, covariances, rotation):
        return rotation @ covariances @ rotation.transpose(0, 1)

    def _covariance_payload(self, points, covariances):
        point_np = point_tensor_to_numpy(points, self.cov_stride)
        cov_np = covariance_tensor_to_numpy(covariances, self.cov_stride)
        means = [point for point in point_np]
        covs = [cov for cov in cov_np]
        return means, covs

    def _jet_colors_from_z(self, points):
        if points.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32)

        z = points[:, 2]
        z_mean = float(z.mean())
        z_std = float(z.std())
        z_min = z_mean - 3.0 * z_std
        z_max = z_mean + 3.0 * z_std

        if z_std < 1e-8 or z_max - z_min < 1e-8:
            z_norm = np.full(points.shape[0], 0.5, dtype=np.float32)
        else:
            z_clipped = np.clip(z, z_min, z_max)
            z_norm = ((z_clipped - z_min) / (z_max - z_min)).astype(np.float32)

        r = np.clip(1.5 - np.abs(4.0 * z_norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * z_norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * z_norm - 1.0), 0.0, 1.0)
        a = np.ones_like(r, dtype=np.float32)
        return np.stack([r, g, b, a], axis=1).astype(np.float32)

    def update(self, frame_idx, sample_path, P, Q, C_p, C_q, R_est, t_est, T_world_est, T_world_gt):
        if self.is_closed:
            return

        if self.viewer.closed():
            self.is_closed = True
            return

        Q_world = self._world_points(Q, T_world_est)
        P_aligned_world = self._world_points(R_est @ P + t_est, T_world_est)

        Q_world_np = point_tensor_to_numpy(Q_world, self.point_stride)
        P_aligned_world_np = point_tensor_to_numpy(P_aligned_world, self.point_stride)
        self.target_history.append(Q_world_np)

        map_points = np.vstack(list(self.target_history)) if self.target_history else np.empty((0, 3), dtype=np.float32)
        map_colors = self._jet_colors_from_z(map_points)
        world_rot_q = T_world_est[:3, :3]
        world_rot_p = world_rot_q @ R_est
        C_q_world = self._world_covariances(C_q, world_rot_q)
        C_p_world = self._world_covariances(C_p, world_rot_p)

        est_position = T_world_est[:3, 3].detach().cpu().numpy().astype(np.float32)
        gt_position = T_world_gt[:3, 3].detach().cpu().numpy().astype(np.float32)
        self.est_positions.append(est_position)
        self.gt_positions.append(gt_position)

        map_buffer = self.glk.PointCloudBuffer(map_points)
        map_buffer.add_color(map_colors)
        self.viewer.update_drawable(
            "target_history",
            map_buffer,
            self.map_shader,
        )
        self.viewer.update_points(
            "target_current",
            Q_world_np,
            self.target_shader,
        )
        self.viewer.update_points(
            "source_aligned",
            P_aligned_world_np,
            self.source_shader,
        )
        self.viewer.update_points(
            "traj_est_nodes",
            positions_to_numpy(self.est_positions),
            self.est_shader,
        )
        self.viewer.update_points(
            "traj_gt_nodes",
            positions_to_numpy(self.gt_positions),
            self.gt_shader,
        )

        if self.show_covariances:
            source_means, source_covs = self._covariance_payload(P_aligned_world, C_p_world)
            self.viewer.update_normal_dists(
                "source_covariances",
                source_means,
                source_covs,
                self.cov_scale,
                self.source_cov_shader,
            )

        if len(self.est_positions) >= 2:
            est_vertices = positions_to_numpy(self.est_positions)
            est_lines = self.viewer.update_thin_lines(
                "traj_est",
                est_vertices,
                colors=make_color_array(est_vertices.shape[0], (1.00, 0.32, 0.24, 1.00)),
                line_strip=True,
                shader_setting=self.est_shader,
            )
            est_lines.set_line_width(2.5)

        if len(self.gt_positions) >= 2:
            gt_vertices = positions_to_numpy(self.gt_positions)
            gt_lines = self.viewer.update_thin_lines(
                "traj_gt",
                gt_vertices,
                colors=make_color_array(gt_vertices.shape[0], (0.18, 0.95, 0.52, 1.00)),
                line_strip=True,
                shader_setting=self.gt_shader,
            )
            gt_lines.set_line_width(2.5)

        self.viewer.update_coord("world_axes", self.coord_shader)
        self.viewer.lookat(est_position.astype(np.float32))
        self.viewer.clear_text()
        self.viewer.append_text(
            "\n".join(
                [
                    f"POLI-GICP | {self.dataset_name}",
                    f"frame: {frame_idx}",
                    f"sample: {os.path.basename(sample_path)}",
                    f"history clouds: {len(self.target_history)}",
                    f"render points: map={map_points.shape[0]} current={Q_world_np.shape[0]}",
                    (
                        f"covariances: source only, stride={self.cov_stride}, "
                        f"source={len(source_means)}"
                    ) if self.show_covariances else "covariances: off",
                ]
            )
        )
        if not self.viewer.spin_once():
            self.is_closed = True

    def wait(self):
        if self.is_closed:
            return
        self.viewer.spin()


def save_trajectory_pair(results, output_folder, name):
    tum_est = []
    tum_gt = []
    T_est = torch.eye(4, dtype=torch.float32)
    T_gt = torch.eye(4, dtype=torch.float32)

    for pred, gt in results:
        R_est, t_est = pred
        R_gt, t_gt = gt

        T_est = relative_pose_to_matrix(R_est, t_est) @ T_est
        T_gt = relative_pose_to_matrix(R_gt, t_gt) @ T_gt

        tum_est.append(torch.linalg.inv(T_est.clone()))
        tum_gt.append(torch.linalg.inv(T_gt.clone()))

    os.makedirs(output_folder, exist_ok=True)
    save_tum_traj(tum_est, os.path.join(output_folder, f"{name}_est.txt"))
    save_tum_traj(tum_gt, os.path.join(output_folder, f"{name}_gt.txt"))


def save_tum_traj(traj, filename):
    with open(filename, "w") as f:
        for idx, T in enumerate(traj):
            line = matrix_to_tum(T, idx)
            f.write(line + "\n")

    print(f"Trajectory saved to {filename}")


def matrix_to_tum(T, idx):
    rot = T[:3, :3]
    trans = T[:3, 3]

    rot_np = rot.detach().cpu().numpy()
    trans_np = trans.detach().cpu().numpy()

    quat = R.from_matrix(rot_np).as_quat()
    return (
        f"{idx} {trans_np[0]} {trans_np[1]} {trans_np[2]} "
        f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
    )


def evaluate_dataset(dataloader, model, icp_solver, mean, scaling_factor, device, viewer=None):
    R_prev = torch.eye(3, device=device)
    t_prev = torch.zeros((3, 1), device=device)
    dgicp_result = []
    T_est_chain = torch.eye(4, device=device)
    T_gt_chain = torch.eye(4, device=device)
    sample_files = getattr(dataloader.dataset, "sample_files", None)

    with torch.no_grad():
        for frame_idx, batch in enumerate(tqdm(dataloader, desc="POLI-GICP")):
            P = batch["P"].to(device)
            Q = batch["Q"].to(device)
            R_rel = batch["R_rel"].to(device).squeeze(0)
            t_rel = batch["t_rel"].to(device).squeeze(0)

            normalized_P, normalized_Q = scale_aware_normalization(P, Q, mean, scaling_factor)

            C_p = model(normalized_P)
            C_q = model(normalized_Q)
            C_p = covariance_generation(C_p.permute(0, 2, 1)).squeeze(0)
            C_q = covariance_generation(C_q.permute(0, 2, 1)).squeeze(0)

            R_dgicp, t_dgicp = icp_solver(
                P.squeeze(0),
                Q.squeeze(0),
                R_prev,
                t_prev,
                C_p,
                C_q,
            )
            R_prev = R_dgicp
            t_prev = t_dgicp

            T_est_chain = relative_pose_to_matrix(R_dgicp, t_dgicp) @ T_est_chain
            T_gt_chain = relative_pose_to_matrix(R_rel, t_rel) @ T_gt_chain
            T_world_est = torch.linalg.inv(T_est_chain)
            T_world_gt = torch.linalg.inv(T_gt_chain)

            if viewer is not None:
                sample_path = sample_files[frame_idx] if sample_files is not None else f"frame_{frame_idx:06d}"
                viewer.update(
                    frame_idx=frame_idx,
                    sample_path=sample_path,
                    P=P.squeeze(0),
                    Q=Q.squeeze(0),
                    C_p=C_p,
                    C_q=C_q,
                    R_est=R_dgicp,
                    t_est=t_dgicp,
                    T_world_est=T_world_est,
                    T_world_gt=T_world_gt,
                )

            dgicp_result.append(
                (
                    (R_dgicp.detach().cpu(), t_dgicp.detach().cpu()),
                    (R_rel.detach().cpu(), t_rel.detach().cpu()),
                )
            )

    return dgicp_result


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")

    device = torch.device(args.device)
    dataset = PCD_Dataset(args.dataset_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No parsed '.npz' files found in dataset_dir: {args.dataset_dir}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = PointPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = torch.tensor(args.mean, dtype=torch.float32, device=device).view(1, 3, 1)
    icp_solver = load_icp_solver()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(dataset)}")
    viewer = POLIGICPViewer(
        dataset_name=dataset_name_from_path(args.dataset_dir),
        point_stride=VISUALIZE_POINT_STRIDE,
        history_size=VISUALIZE_HISTORY_SIZE,
        point_size=VISUALIZE_POINT_SIZE,
        show_covariances=VISUALIZE_SHOW_COVARIANCES,
        cov_stride=VISUALIZE_COV_STRIDE,
        cov_scale=VISUALIZE_COV_SCALE,
    )

    results = evaluate_dataset(
        dataloader=dataloader,
        model=model,
        icp_solver=icp_solver,
        mean=mean,
        scaling_factor=args.scaling_factor,
        device=device,
        viewer=viewer,
    )

    save_trajectory_pair(results, args.output_folder, dataset_name_from_path(args.dataset_dir))

    print("Close the pyridescence viewer to finish.")
    viewer.wait()


if __name__ == "__main__":
    main(parse_args())
