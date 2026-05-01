#!/usr/bin/env python3
import argparse
import atexit
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from registration_utils import (
    compute_fpfh_batch,
    estimate_correspondences,
    load_checkpoint,
    maximum_consensus,
    rotation_translation_error,
    svd_registration,
)


STAGES = [
    ("covariance", "1. Neural Covariance"),
    ("normals", "2. Surface Normals"),
    ("correspondence", "3. FPFH Consensus"),
    ("registration", "4. Registration"),
]

# Keep overlapping defaults aligned with applications/object_pose_estimation/demo.py.
DEMO_DEFAULTS = {
    "device": "cuda",
    "fpfh_radius": 0.2,
    "fpfh_max_nn": 90,
    "consensus_beta": 0.1,
    "max_lines": 1000,
    "max_normals": 1000,
    "normal_length": 0.15,
    "seed": 7777,
}


def cleanup_imgui_ini():
    candidate_paths = {
        os.path.join(os.getcwd(), "imgui.ini"),
        os.path.join(str(SCRIPT_DIR), "imgui.ini"),
    }
    for path in candidate_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


atexit.register(cleanup_imgui_ini)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iridescence-based POLI object pose pipeline viewer."
    )
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=DEMO_DEFAULTS["device"])
    parser.add_argument("--seed", type=int, default=DEMO_DEFAULTS["seed"])

    parser.add_argument("--fpfh_radius", type=float, default=DEMO_DEFAULTS["fpfh_radius"])
    parser.add_argument("--fpfh_max_nn", type=int, default=DEMO_DEFAULTS["fpfh_max_nn"])
    parser.add_argument("--consensus_beta", type=float, default=DEMO_DEFAULTS["consensus_beta"])

    parser.add_argument("--scan_separation", type=float, default=2.2)
    parser.add_argument("--max_points", type=int, default=2400)
    parser.add_argument("--max_covariances", type=int, default=420)
    parser.add_argument("--max_normals", type=int, default=DEMO_DEFAULTS["max_normals"])
    parser.add_argument("--max_lines", type=int, default=DEMO_DEFAULTS["max_lines"])

    parser.add_argument("--normal_length", type=float, default=DEMO_DEFAULTS["normal_length"])
    parser.add_argument("--covariance_scale", type=float, default=0.06)
    parser.add_argument("--point_size", type=float, default=0.018)
    parser.add_argument("--line_width", type=float, default=1.9)
    parser.add_argument("--normal_width", type=float, default=1.4)

    parser.add_argument("--camera_theta", type=float, default=0.36)
    parser.add_argument("--camera_phi", type=float, default=-0.92)
    parser.add_argument("--camera_distance", type=float, default=None)
    parser.add_argument("--grid", action="store_true")

    parser.add_argument("--covariance_seconds", type=float, default=1.7)
    parser.add_argument("--normal_seconds", type=float, default=1.7)
    parser.add_argument("--correspondence_seconds", type=float, default=1.7)
    parser.add_argument("--registration_seconds", type=float, default=2.4)
    parser.add_argument("--hold_seconds", type=float, default=1.8)
    parser.add_argument("--playback_rate", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")

    return parser.parse_args()


def load_iridescence():
    try:
        import pyridescence.glk as glk
        import pyridescence.guik as guik
    except ImportError as exc:
        raise ImportError(
            "This viewer requires `pyridescence`. Install it with `pip install pyridescence`."
        ) from exc
    return guik, glk


def choose_indices(n_items, max_items, seed):
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=max_items, replace=False))


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clean_rotation(R_mat):
    u, _, vh = np.linalg.svd(R_mat)
    R_clean = u @ vh
    if np.linalg.det(R_clean) < 0:
        vh[2, :] *= -1
        R_clean = u @ vh
    return R_clean


def make_slerp(R_est):
    rotations = Rotation.from_matrix([np.eye(3), R_est])
    return Slerp([0.0, 1.0], rotations)


def make_turbo_colors_from_z(points, anchor_z=None, alpha=1.0):
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    if anchor_z is None:
        anchor_z = float(np.median(points[:, 2]))

    relative_z = points[:, 2] - float(anchor_z)
    z_min = -0.25 * max(np.ptp(points[:, 2]), 1e-6)
    z_max = float(np.percentile(relative_z, 95.0))
    z_max = max(z_max, 0.25 * max(np.ptp(points[:, 2]), 1e-6))

    if z_max - z_min < 1e-8:
        return np.full((points.shape[0], 4), (0.5, 0.5, 0.5, alpha), dtype=np.float32)

    z_clipped = np.clip(relative_z, z_min, z_max)
    z_norm = ((z_clipped - z_min) / (z_max - z_min)).astype(np.float32)
    turbo_t = 0.08 + 0.92 * z_norm

    z2 = turbo_t * turbo_t
    z3 = z2 * turbo_t
    z4 = z3 * turbo_t
    z5 = z4 * turbo_t

    r = 0.13572138 + 4.61539260 * turbo_t - 42.66032258 * z2 + 132.13108234 * z3 - 152.94239396 * z4 + 59.28637943 * z5
    g = 0.09140261 + 2.19418839 * turbo_t + 4.84296658 * z2 - 14.18503333 * z3 + 4.27729857 * z4 + 2.82956604 * z5
    b = 0.10667330 + 12.64194608 * turbo_t - 60.58204836 * z2 + 110.36276771 * z3 - 89.90310912 * z4 + 27.34824973 * z5

    colors = np.stack([r, g, b], axis=1)
    colors = np.clip(colors, 0.0, 1.0)
    alphas = np.full((points.shape[0], 1), alpha, dtype=np.float32)
    return np.concatenate([colors.astype(np.float32), alphas], axis=1)


def make_iridescent_colors(points, phase=0.0, alpha=1.0):
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    scale = max(float(np.percentile(radius, 90.0)), 1e-6)
    p = centered / scale
    wave = 2.9 * p[:, 0] - 2.2 * p[:, 1] + 3.7 * p[:, 2] + float(phase)

    r = 0.58 + 0.34 * np.sin(wave + 0.0)
    g = 0.62 + 0.28 * np.sin(wave + 2.1)
    b = 0.72 + 0.26 * np.sin(wave + 4.2)

    gloss = np.clip(0.75 + 0.35 * p[:, 2], 0.55, 1.10)
    colors = np.stack([r, g, b], axis=1) * gloss[:, None]
    colors = np.clip(colors, 0.0, 1.0)
    colors = 0.82 * colors + 0.18

    alphas = np.full((points.shape[0], 1), alpha, dtype=np.float32)
    return np.concatenate([colors.astype(np.float32), alphas], axis=1)


def transform_covariances(covariances, rotation):
    return rotation @ covariances @ rotation.T


def make_segment_vertices(starts, ends):
    vertices = np.empty((starts.shape[0] * 2, 3), dtype=np.float32)
    vertices[0::2] = starts.astype(np.float32)
    vertices[1::2] = ends.astype(np.float32)
    return vertices


def make_scene_center_distance(data, scan_separation):
    offset = np.array([scan_separation, 0.0, 0.0], dtype=np.float32)
    aligned = (data["P_np"] @ data["R_est"].T) + data["t_est"].reshape(1, 3)
    scene = np.vstack([data["P_np"] + offset, data["Q_np"], aligned]).astype(np.float32)

    center = np.mean(np.vstack([data["P_np"] + offset, data["Q_np"]]), axis=0).astype(np.float32)
    center[0] += 0.08 * max(np.ptp(scene[:, 0]), 1e-6)

    extent = scene.max(axis=0) - scene.min(axis=0)
    distance = max(float(np.max(extent)) * 2.6, 0.6)
    return center, distance


def load_npz_sample_local(path):
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


def make_covariance_normals_local(model, points_b3n):
    from registration_utils import covariance_generation, normals_from_covariance

    points_centered = points_b3n - points_b3n.mean(dim=2, keepdim=True)
    cov_elements = model(points_centered)
    covariances = covariance_generation(cov_elements.permute(0, 2, 1))
    normals = normals_from_covariance(covariances)
    return covariances, normals


def orient_normals_outward_local(points, normals):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    outward_dirs = points - center
    flip_mask = np.sum(normals * outward_dirs, axis=1, keepdims=True) < 0.0
    return np.where(flip_mask, -normals, normals).astype(np.float32)


def load_pipeline_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample_path = Path(args.sample).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    sample = load_npz_sample_local(sample_path)
    P_np = sample["P"].T
    Q_np = sample["Q"].T
    R_gt = sample["R_rel"]
    t_gt = sample["t_rel"]

    P = torch.from_numpy(sample["P"]).unsqueeze(0).to(device)
    Q = torch.from_numpy(sample["Q"]).unsqueeze(0).to(device)

    from model import PointPP

    model = PointPP().to(device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    with torch.no_grad():
        cov_p, normals_p = make_covariance_normals_local(model, P)
        cov_q, normals_q = make_covariance_normals_local(model, Q)

        fpfh_p = compute_fpfh_batch(
            P,
            normals_p,
            radius=args.fpfh_radius,
            max_nn=args.fpfh_max_nn,
        )
        fpfh_q = compute_fpfh_batch(
            Q,
            normals_q,
            radius=args.fpfh_radius,
            max_nn=args.fpfh_max_nn,
        )
        matches = estimate_correspondences(fpfh_p[0], fpfh_q[0]).detach().cpu().numpy()

    src_matched = P_np
    tgt_matched = Q_np[matches]
    src_inliers, tgt_inliers = maximum_consensus(
        src_matched,
        tgt_matched,
        beta=args.consensus_beta,
    )
    if len(src_inliers) < 3:
        raise RuntimeError("Not enough inliers to estimate pose.")

    T_est = svd_registration(src_inliers, tgt_inliers)
    R_est = clean_rotation(T_est[:3, :3])
    t_est = T_est[:3, 3]

    rot_error, trans_error = rotation_translation_error(
        R_est,
        t_est,
        R_gt,
        t_gt,
    )

    normals_p_np = orient_normals_outward_local(
        P_np,
        normals_p[0].detach().cpu().numpy(),
    )
    normals_q_np = orient_normals_outward_local(
        Q_np,
        normals_q[0].detach().cpu().numpy(),
    )

    return {
        "sample_path": sample_path,
        "P_np": P_np.astype(np.float32),
        "Q_np": Q_np.astype(np.float32),
        "cov_p": cov_p[0].detach().cpu().numpy().astype(np.float32),
        "cov_q": cov_q[0].detach().cpu().numpy().astype(np.float32),
        "normals_p": normals_p_np.astype(np.float32),
        "normals_q": normals_q_np.astype(np.float32),
        "src_inliers": src_inliers.astype(np.float32),
        "tgt_inliers": tgt_inliers.astype(np.float32),
        "R_est": R_est.astype(np.float32),
        "t_est": t_est.astype(np.float32),
        "rot_error": float(rot_error),
        "trans_error": float(trans_error),
    }


class IridescencePipelineViewer:
    def __init__(self, data, args):
        self.guik, self.glk = load_iridescence()
        self.data = data
        self.args = args

        self.center, auto_distance = make_scene_center_distance(data, args.scan_separation)
        self.camera_distance = args.camera_distance if args.camera_distance is not None else auto_distance
        self.offset = np.array([args.scan_separation, 0.0, 0.0], dtype=np.float32)

        self.stage_durations = [
            ("covariance", float(args.covariance_seconds)),
            ("normals", float(args.normal_seconds)),
            ("correspondence", float(args.correspondence_seconds)),
            ("registration", float(args.registration_seconds)),
            ("hold", float(args.hold_seconds)),
        ]
        self.total_duration = sum(duration for _, duration in self.stage_durations)

        self.state = {
            "point_idx_p": choose_indices(len(data["P_np"]), args.max_points, args.seed + 1),
            "point_idx_q": choose_indices(len(data["Q_np"]), args.max_points, args.seed + 2),
            "cov_idx_p": choose_indices(len(data["P_np"]), args.max_covariances, args.seed + 3),
            "cov_idx_q": choose_indices(len(data["Q_np"]), args.max_covariances, args.seed + 4),
            "normal_idx_p": choose_indices(len(data["P_np"]), args.max_normals, args.seed + 5),
            "normal_idx_q": choose_indices(len(data["Q_np"]), args.max_normals, args.seed + 6),
            "line_idx": choose_indices(
                len(data["src_inliers"]),
                min(args.max_lines, len(data["src_inliers"])),
                args.seed + 7,
            ),
            "slerp": make_slerp(data["R_est"]),
        }

        self.viewer = self.guik.LightViewer.instance(
            title=f"POLI Iridescence | {data['sample_path'].stem}"
        )
        self.viewer.set_clear_color(np.array((0.01, 0.015, 0.035, 1.0), dtype=np.float32))
        self.viewer.set_draw_xy_grid(bool(args.grid))
        self.viewer.set_point_shape(point_size=args.point_size, metric=True, circle=True)
        self.viewer.use_orbit_camera_control(
            distance=self.camera_distance,
            theta=args.camera_theta,
            phi=args.camera_phi,
        )

        self.vertex_shader = self.guik.VertexColor()
        self.vertex_shader.set_point_shape_circle()
        self.vertex_shader.set_point_scale_metric()
        self.vertex_shader.set_point_size(args.point_size)

        self.target_cov_shader = self._make_flat_shader((0.14, 0.55, 1.0), 0.11).make_transparent()
        self.source_cov_shader = self._make_flat_shader((1.0, 0.42, 0.92), 0.14).make_transparent()
        self.normal_target_shader = self._make_flat_shader((0.18, 0.98, 0.92), 0.96)
        self.normal_source_shader = self._make_flat_shader((1.0, 0.78, 0.32), 0.96)
        self.corr_shader = self._make_flat_shader((0.18, 1.0, 0.72), 0.95)
        self.coord_shader = self.guik.VertexColor()

    def _make_flat_shader(self, color, alpha):
        shader = self.guik.FlatColor(
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(alpha),
        )
        shader.set_point_shape_circle()
        shader.set_point_scale_metric()
        shader.set_point_size(self.args.point_size)
        return shader

    def _scene_time(self, elapsed):
        if self.total_duration <= 0.0:
            return 0.0
        if self.args.loop:
            return elapsed % self.total_duration
        return min(elapsed, self.total_duration)

    def _stage_progress(self, elapsed):
        t = self._scene_time(elapsed)
        cursor = 0.0
        for stage, duration in self.stage_durations:
            next_cursor = cursor + duration
            if t <= next_cursor or stage == self.stage_durations[-1][0]:
                progress = 1.0 if duration <= 1e-8 else (t - cursor) / duration
                return stage, np.clip(progress, 0.0, 1.0)
            cursor = next_cursor
        return "hold", 1.0

    def _update_points(self, name, points, colors):
        buffer = self.glk.PointCloudBuffer(points.astype(np.float32))
        buffer.add_color(colors.astype(np.float32))
        self.viewer.update_drawable(name, buffer, self.vertex_shader)

    def _update_covariances(self, name, points, covariances, indices, count, shader):
        if count <= 0:
            return

        idx = indices[:count]
        means = [point.astype(np.float32) for point in points[idx]]
        covs = [cov.astype(np.float32) for cov in covariances[idx]]
        self.viewer.update_normal_dists(
            name,
            means,
            covs,
            self.args.covariance_scale,
            shader,
        )

    def _update_segments(self, name, starts, ends, shader, width):
        if starts.shape[0] == 0:
            return

        vertices = make_segment_vertices(starts, ends)

        lines = self.viewer.update_thin_lines(
            name,
            vertices,
            line_strip=False,
            shader_setting=shader,
        )
        if lines is not None:
            lines.set_line_width(float(width))

    def _normal_segments(self, points, normals, indices, count):
        if count <= 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        idx = indices[:count]
        starts = points[idx]
        vecs = normals[idx]
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        ends = starts + vecs * float(self.args.normal_length)
        return starts.astype(np.float32), ends.astype(np.float32)

    def _correspondence_segments(self, src_points, tgt_points, count):
        if count <= 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        idx = self.state["line_idx"][:count]
        return src_points[idx].astype(np.float32), tgt_points[idx].astype(np.float32)

    def _update_hud(self, stage):
        lines = [
            "POLI Iridescence",
            "",
        ]
        for name, label in STAGES:
            prefix = ">" if name == stage or (stage == "hold" and name == "registration") else " "
            lines.append(f"{prefix} {label}")

        lines.extend(
            [
                "",
                f"sample: {self.data['sample_path'].name}",
                f"inliers: {len(self.data['src_inliers'])}",
                f"rot err: {self.data['rot_error']:.3f} deg",
                f"trans err: {self.data['trans_error']:.5f}",
                "",
                "theme: turbo target / iridescent source",
                "controls: drag orbit, wheel zoom",
            ]
        )

        self.viewer.clear_text()
        self.viewer.append_text("\n".join(lines))

    def render(self, elapsed):
        stage, progress = self._stage_progress(elapsed)
        reveal = smoothstep(progress)
        shimmer_phase = 1.35 * elapsed

        P_np = self.data["P_np"]
        Q_np = self.data["Q_np"]

        if stage in {"covariance", "normals", "correspondence"}:
            P_display = P_np + self.offset
            normals_p_display = self.data["normals_p"]
            cov_p_display = self.data["cov_p"]
            src_corr_display = self.data["src_inliers"] + self.offset
        else:
            alpha = smoothstep(progress if stage == "registration" else 1.0)
            R_alpha = self.state["slerp"]([alpha]).as_matrix()[0].astype(np.float32)
            t_alpha = alpha * self.data["t_est"].reshape(1, 3)
            offset_alpha = (1.0 - alpha) * self.offset.reshape(1, 3)

            P_display = (P_np @ R_alpha.T) + t_alpha + offset_alpha
            normals_p_display = self.data["normals_p"] @ R_alpha.T
            cov_p_display = np.stack(
                [transform_covariances(cov, R_alpha) for cov in self.data["cov_p"]],
                axis=0,
            ).astype(np.float32)
            src_corr_display = (self.data["src_inliers"] @ R_alpha.T) + t_alpha + offset_alpha

        point_idx_p = self.state["point_idx_p"]
        point_idx_q = self.state["point_idx_q"]

        source_points = P_display[point_idx_p]
        target_points = Q_np[point_idx_q]

        source_colors = make_iridescent_colors(
            source_points,
            phase=shimmer_phase,
            alpha=0.98,
        )
        target_colors = make_turbo_colors_from_z(
            target_points,
            anchor_z=float(np.median(Q_np[:, 2])),
            alpha=0.96,
        )

        self._update_points("source_points", source_points, source_colors)
        self._update_points("target_points", target_points, target_colors)

        if stage == "covariance":
            cov_count = int(np.ceil(len(self.state["cov_idx_p"]) * reveal))
            self._update_covariances(
                "source_covariances",
                P_display,
                cov_p_display,
                self.state["cov_idx_p"],
                cov_count,
                self.source_cov_shader,
            )
            self._update_covariances(
                "target_covariances",
                Q_np,
                self.data["cov_q"],
                self.state["cov_idx_q"],
                int(np.ceil(len(self.state["cov_idx_q"]) * reveal)),
                self.target_cov_shader,
            )
        else:
            self._update_covariances(
                "source_covariances",
                P_display,
                cov_p_display,
                self.state["cov_idx_p"],
                len(self.state["cov_idx_p"]),
                self.source_cov_shader,
            )
            self._update_covariances(
                "target_covariances",
                Q_np,
                self.data["cov_q"],
                self.state["cov_idx_q"],
                len(self.state["cov_idx_q"]),
                self.target_cov_shader,
            )

        if stage == "normals":
            normal_count_p = int(np.ceil(len(self.state["normal_idx_p"]) * reveal))
            normal_count_q = int(np.ceil(len(self.state["normal_idx_q"]) * reveal))
        elif stage in {"correspondence", "registration", "hold"}:
            normal_count_p = len(self.state["normal_idx_p"])
            normal_count_q = len(self.state["normal_idx_q"])
        else:
            normal_count_p = 0
            normal_count_q = 0

        normal_src_starts, normal_src_ends = self._normal_segments(
            P_display,
            normals_p_display,
            self.state["normal_idx_p"],
            normal_count_p,
        )
        normal_tgt_starts, normal_tgt_ends = self._normal_segments(
            Q_np,
            self.data["normals_q"],
            self.state["normal_idx_q"],
            normal_count_q,
        )
        self._update_segments(
            "source_normals",
            normal_src_starts,
            normal_src_ends,
            self.normal_source_shader,
            self.args.normal_width,
        )
        self._update_segments(
            "target_normals",
            normal_tgt_starts,
            normal_tgt_ends,
            self.normal_target_shader,
            self.args.normal_width,
        )

        if stage == "correspondence":
            corr_count = int(np.ceil(len(self.state["line_idx"]) * reveal))
        elif stage in {"registration", "hold"}:
            corr_count = len(self.state["line_idx"])
        else:
            corr_count = 0

        corr_starts, corr_ends = self._correspondence_segments(
            src_corr_display,
            self.data["tgt_inliers"],
            corr_count,
        )
        self._update_segments(
            "correspondences",
            corr_starts,
            corr_ends,
            self.corr_shader,
            self.args.line_width,
        )

        self.viewer.update_coord("world_axes", self.coord_shader)
        self.viewer.lookat(self.center.astype(np.float32))
        self._update_hud(stage)
        alive = self.viewer.spin_once()
        cleanup_imgui_ini()
        return alive

    def run(self):
        start_time = time.perf_counter()
        while True:
            if self.viewer.closed():
                break

            elapsed = (time.perf_counter() - start_time) * float(self.args.playback_rate)
            if not self.render(elapsed):
                break


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    data = load_pipeline_data(args)
    viewer = IridescencePipelineViewer(data, args)
    viewer.run()


if __name__ == "__main__":
    main(parse_args())
