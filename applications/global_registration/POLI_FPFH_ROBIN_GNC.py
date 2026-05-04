#!/usr/bin/env python3
import argparse
import atexit
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from applications.object_pose_estimation.registration_utils import (
    compute_fpfh,
    estimate_mutual_feature_matches,
    gnc_registration,
    maximum_consensus,
    rotation_translation_error,
)
from utils import covariance_generation, scale_aware_normalization
from visualization.iridescence_utils import (
    choose_indices,
    clean_rotation,
    cleanup_imgui_ini,
    load_iridescence,
    make_iridescent_colors,
    make_segment_vertices,
    make_slerp,
    make_turbo_colors_from_z,
    smoothstep,
)
from visualization.vis_utils import load_model

STAGES = [
    ("raw",            "1. Raw Point Clouds"),
    ("densification",  "2. Gaussian Densification"),
    ("correspondence", "3. FPFH + ROBIN"),
    ("registration",   "4. GNC Registration"),
]

STAGE_DURATIONS = [
    ("raw",            1.5),
    ("densification",  2.2),
    ("correspondence", 2.0),
    ("registration",   2.5),
    ("hold",           1.8),
]

DEMO_DEFAULTS = {
    "device":              "cuda",
    "seed":                7777,
    "samples_per_point":   100,
    "n_std":               0.5,
    "fpfh_normal_radius":  3.0,
    "fpfh_feature_radius": 5.0,
    "fpfh_normal_max_nn":  30,
    "fpfh_feature_max_nn": 90,
    "robin_beta":          1.0,
    "gnc_solver":          "tls",
    "gnc_noise_bound":     None,
}

NORMALIZATION_MEAN = [-0.6547287702560425, 0.36115387082099915, 0.7911769151687622]
SCALING_FACTOR = 100.0


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


@torch.no_grad()
def predict_covariances(model, P_3xn, Q_3xm, device):
    P = torch.from_numpy(P_3xn).unsqueeze(0).to(device=device, dtype=torch.float32)
    Q = torch.from_numpy(Q_3xm).unsqueeze(0).to(device=device, dtype=torch.float32)
    P_norm, Q_norm = scale_aware_normalization(P, Q, NORMALIZATION_MEAN, SCALING_FACTOR)

    C_p = covariance_generation(model(P_norm).permute(0, 2, 1))[0]
    C_q = covariance_generation(model(Q_norm).permute(0, 2, 1))[0]

    return P[0].T.contiguous(), Q[0].T.contiguous(), C_p, C_q


@torch.no_grad()
def densify_points(points_n3, covariances_n33, samples_per_point, n_std):
    device = points_n3.device
    eps = 1e-6
    C = covariances_n33 + torch.eye(3, device=device).unsqueeze(0) * eps

    try:
        L = torch.linalg.cholesky(C)
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
    )
    keep = torch.linalg.norm(z, dim=-1) <= n_std
    delta = z @ L.transpose(1, 2)
    sampled = (points_n3.unsqueeze(1) + delta)[keep]
    sampled = torch.cat([points_n3, sampled], dim=0)
    return sampled.detach().cpu().numpy().astype(np.float32)


atexit.register(cleanup_imgui_ini, SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Iridescence POLI global registration viewer.")
    parser.add_argument("--sample",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="/home/jiwoo/Desktop/POLI/weights/HeLiPR/vlp_helipr_0.2m.pth")
    parser.add_argument("--device",     type=str, default=DEMO_DEFAULTS["device"])
    parser.add_argument("--seed",       type=int, default=DEMO_DEFAULTS["seed"])

    parser.add_argument("--samples_per_point",   type=int,   default=DEMO_DEFAULTS["samples_per_point"])
    parser.add_argument("--n_std",               type=float, default=DEMO_DEFAULTS["n_std"])
    parser.add_argument("--fpfh_normal_radius",  type=float, default=DEMO_DEFAULTS["fpfh_normal_radius"])
    parser.add_argument("--fpfh_feature_radius", type=float, default=DEMO_DEFAULTS["fpfh_feature_radius"])
    parser.add_argument("--fpfh_normal_max_nn",  type=int,   default=DEMO_DEFAULTS["fpfh_normal_max_nn"])
    parser.add_argument("--fpfh_feature_max_nn", type=int,   default=DEMO_DEFAULTS["fpfh_feature_max_nn"])
    parser.add_argument("--robin_beta",          type=float, default=DEMO_DEFAULTS["robin_beta"])
    parser.add_argument("--gnc_solver",          choices=("tls", "gm"), default=DEMO_DEFAULTS["gnc_solver"])
    parser.add_argument("--gnc_noise_bound",     type=float, default=DEMO_DEFAULTS["gnc_noise_bound"])

    parser.add_argument("--max_points", type=int,   default=8000,  help="Raw points shown per cloud.")
    parser.add_argument("--max_dense",  type=int,   default=50000, help="Max total dense points shown per cloud.")
    parser.add_argument("--max_lines",  type=int,   default=300,   help="Max correspondence lines.")

    parser.add_argument("--scan_separation", type=float, default=None, help="X offset between clouds (auto if None).")
    parser.add_argument("--camera_theta",    type=float, default=0.5)
    parser.add_argument("--camera_phi",      type=float, default=-0.9)
    parser.add_argument("--camera_distance", type=float, default=None)
    parser.add_argument("--grid",            action="store_true")
    parser.add_argument("--point_size",      type=float, default=0.12)
    parser.add_argument("--line_width",      type=float, default=1.0)

    return parser.parse_args()


def compute_scene_layout(P_dense, Q_dense, scan_separation_override=None):
    all_pts = np.vstack([P_dense, Q_dense])
    extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    if scan_separation_override is not None:
        sep = float(scan_separation_override)
    else:
        sep = max(float(extent[0]) * 0.75, float(np.linalg.norm(extent)) * 0.25, 1.0)
    offset = np.array([sep, 0.0, 0.0], dtype=np.float32)
    src_shifted = P_dense + offset
    center = ((src_shifted.mean(axis=0) + Q_dense.mean(axis=0)) * 0.5).astype(np.float32)
    scene = np.vstack([src_shifted, Q_dense])
    distance = max(float(np.max(scene.max(axis=0) - scene.min(axis=0))) * 2.8, 10.0)
    return offset, center, distance


def load_pipeline_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    sample_path = Path(args.sample).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    print(f"[Device] {device}")
    print(f"[Sample] {sample_path}")

    P_3xn, Q_3xm, R_gt, t_gt = load_pair_npz(sample_path)
    model = load_model(str(checkpoint_path), device)

    print("[Covariances] predicting...")
    P_n3, Q_m3, C_p, C_q = predict_covariances(model, P_3xn, Q_3xm, device)
    N_raw_p, N_raw_q = P_n3.shape[0], Q_m3.shape[0]
    print(f"[Raw] P={N_raw_p} Q={N_raw_q}")

    print("[Densification] sampling...")
    P_dense = densify_points(P_n3, C_p, args.samples_per_point, args.n_std)
    Q_dense = densify_points(Q_m3, C_q, args.samples_per_point, args.n_std)
    print(f"[Dense] P={len(P_dense)} Q={len(Q_dense)}")

    print("[FPFH] computing...")
    fpfh_p = compute_fpfh(P_dense, args.fpfh_normal_radius, args.fpfh_feature_radius, args.fpfh_normal_max_nn, args.fpfh_feature_max_nn)
    fpfh_q = compute_fpfh(Q_dense, args.fpfh_normal_radius, args.fpfh_feature_radius, args.fpfh_normal_max_nn, args.fpfh_feature_max_nn)

    print("[Match] mutual nearest neighbors...")
    src_idx, tgt_idx = estimate_mutual_feature_matches(fpfh_p, fpfh_q)
    src_matched = P_dense[src_idx]
    tgt_matched = Q_dense[tgt_idx]
    print(f"[Match] matches={len(src_idx)}")

    print("[ROBIN] consensus...")
    src_inliers, tgt_inliers = maximum_consensus(src_matched, tgt_matched, args.robin_beta)
    print(f"[ROBIN] inliers={len(src_inliers)}/{len(src_idx)}")

    gnc_noise_bound = args.gnc_noise_bound if args.gnc_noise_bound is not None else args.robin_beta
    print(f"[GNC] solver={args.gnc_solver} noise_bound={gnc_noise_bound}")
    R_est, t_est = gnc_registration(src_inliers, tgt_inliers, solver=args.gnc_solver, noise_bound=gnc_noise_bound)

    rot_err, trans_err = rotation_translation_error(R_est, t_est, R_gt, t_gt)
    print(f"[Error] rot={rot_err:.4f} deg  trans={trans_err:.6f}")

    return {
        "sample_path":    sample_path,
        "P_dense":        P_dense,
        "Q_dense":        Q_dense,
        "N_raw_p":        N_raw_p,
        "N_raw_q":        N_raw_q,
        "src_inliers":    src_inliers.astype(np.float32),
        "tgt_inliers":    tgt_inliers.astype(np.float32),
        "R_est":          clean_rotation(R_est),
        "t_est":          t_est.astype(np.float32),
        "rot_error":      float(rot_err),
        "trans_error":    float(trans_err),
        "gnc_solver":     args.gnc_solver,
        "gnc_noise_bound":float(gnc_noise_bound),
    }


class IridescencePipelineViewer:
    def __init__(self, data, args):
        self.guik, self.glk = load_iridescence()
        self.data = data
        self.args = args

        self.offset, self.center, auto_dist = compute_scene_layout(
            data["P_dense"], data["Q_dense"], args.scan_separation
        )
        self.camera_distance = args.camera_distance if args.camera_distance is not None else auto_dist

        self.stage_durations = STAGE_DURATIONS
        self.total_duration = sum(d for _, d in self.stage_durations)

        N_raw_p   = data["N_raw_p"]
        N_raw_q   = data["N_raw_q"]
        n_extra_p = len(data["P_dense"]) - N_raw_p
        n_extra_q = len(data["Q_dense"]) - N_raw_q
        max_extra = max(0, args.max_dense - args.max_points)

        raw_idx_p   = choose_indices(N_raw_p,   args.max_points, args.seed + 1)
        raw_idx_q   = choose_indices(N_raw_q,   args.max_points, args.seed + 2)
        extra_idx_p = choose_indices(n_extra_p, max_extra,       args.seed + 3) + N_raw_p
        extra_idx_q = choose_indices(n_extra_q, max_extra,       args.seed + 4) + N_raw_q

        n_inl      = len(data["src_inliers"])
        raw_line   = choose_indices(n_inl, min(args.max_lines, n_inl), args.seed + 7)
        sweep_ord  = np.argsort(data["src_inliers"][raw_line, 0])

        self.state = {
            "raw_idx_p":    raw_idx_p,
            "raw_idx_q":    raw_idx_q,
            "extra_idx_p":  extra_idx_p,
            "extra_idx_q":  extra_idx_q,
            "line_idx":     raw_line[sweep_ord],
            "slerp":        make_slerp(data["R_est"]),
        }
        self.anchor_z_q = float(np.median(data["Q_dense"][:, 2]))

        self.viewer = self.guik.LightViewer.instance(
            title=f"POLI Global Reg | {data['sample_path'].stem}"
        )
        self.viewer.set_clear_color(np.array((0.01, 0.015, 0.035, 1.0), dtype=np.float32))
        self.viewer.set_draw_xy_grid(bool(args.grid))
        self.viewer.set_point_shape(point_size=args.point_size, metric=True, circle=True)
        self.viewer.use_arcball_camera_control(
            distance=self.camera_distance,
            theta=args.camera_theta,
            phi=args.camera_phi,
        )

        self.vertex_shader = self.guik.VertexColor()
        self.vertex_shader.set_point_shape_circle()
        self.vertex_shader.set_point_scale_metric()
        self.vertex_shader.set_point_size(args.point_size)

        self.corr_shader = self.guik.FlatColor(0.18, 1.0, 0.72, 0.95)

    def _scene_time(self, elapsed):
        if self.total_duration <= 0.0:
            return 0.0
        return min(elapsed, self.total_duration)

    def _stage_progress(self, elapsed):
        t = self._scene_time(elapsed)
        cursor = 0.0
        for stage, dur in self.stage_durations:
            nxt = cursor + dur
            if t <= nxt or stage == self.stage_durations[-1][0]:
                prog = 1.0 if dur <= 1e-8 else (t - cursor) / dur
                return stage, float(np.clip(prog, 0.0, 1.0))
            cursor = nxt
        return "hold", 1.0

    def _update_points(self, name, points, colors):
        if points.shape[0] == 0:
            return
        buf = self.glk.PointCloudBuffer(points.astype(np.float32))
        buf.add_color(colors.astype(np.float32))
        self.viewer.update_drawable(name, buf, self.vertex_shader)

    def _update_segments(self, name, starts, ends, width, shader=None):
        if starts.shape[0] == 0:
            return
        lines = self.viewer.update_thin_lines(
            name, make_segment_vertices(starts, ends),
            line_strip=False, shader_setting=shader or self.corr_shader,
        )
        if lines is not None:
            lines.set_line_width(float(width))

    def _update_hud(self, stage):
        lines = ["POLI Global Registration", ""]
        for name, label in STAGES:
            prefix = ">" if name == stage or (stage == "hold" and name == "registration") else " "
            lines.append(f"{prefix} {label}")
        d = self.data
        lines += [
            "",
            f"sample: {d['sample_path'].name}",
            f"raw:  P={d['N_raw_p']}  Q={d['N_raw_q']}",
            f"inliers: {len(d['src_inliers'])}",
            f"gnc: {d['gnc_solver']} | noise_bound: {d['gnc_noise_bound']:.3f}",
            f"rot err:   {d['rot_error']:.3f} deg",
            f"trans err: {d['trans_error']:.5f}",
            "",
            "controls: drag arcball, wheel zoom",
        ]
        self.viewer.clear_text()
        self.viewer.append_text("\n".join(lines))

    def render(self, elapsed):
        stage, progress = self._stage_progress(elapsed)
        reveal  = smoothstep(progress)
        shimmer = 1.35 * elapsed

        P_dense = self.data["P_dense"]
        Q_dense = self.data["Q_dense"]

        # Registration animation: source moves from offset position to aligned
        if stage == "registration":
            reg_a = reveal
        elif stage == "hold":
            reg_a = 1.0
        else:
            reg_a = 0.0

        if reg_a > 0.0:
            R_a = self.state["slerp"]([reg_a]).as_matrix()[0].astype(np.float32)
            t_a = reg_a * self.data["t_est"].reshape(1, 3)
            off_a = (1.0 - reg_a) * self.offset.reshape(1, 3)
            P_src      = (P_dense @ R_a.T) + t_a + off_a
            inl_src    = (self.data["src_inliers"] @ R_a.T) + t_a + off_a
        else:
            P_src   = P_dense + self.offset
            inl_src = self.data["src_inliers"] + self.offset

        # Source: raw always visible; extra fades in during densification
        raw_p   = self.state["raw_idx_p"]
        extra_p = self.state["extra_idx_p"]
        if stage == "raw":
            src_idx = raw_p
        elif stage == "densification":
            n_extra = int(np.ceil(len(extra_p) * reveal))
            src_idx = np.concatenate([raw_p, extra_p[:n_extra]])
        else:
            src_idx = np.concatenate([raw_p, extra_p])

        raw_q   = self.state["raw_idx_q"]
        extra_q = self.state["extra_idx_q"]
        if stage == "raw":
            tgt_idx = raw_q
        elif stage == "densification":
            n_extra_q = int(np.ceil(len(extra_q) * reveal))
            tgt_idx = np.concatenate([raw_q, extra_q[:n_extra_q]])
        else:
            tgt_idx = np.concatenate([raw_q, extra_q])

        src_pts = P_src[src_idx]
        tgt_pts = Q_dense[tgt_idx]

        self._update_points("source", src_pts,
                            make_iridescent_colors(src_pts, phase=shimmer, alpha=0.98))
        self._update_points("target", tgt_pts,
                            make_turbo_colors_from_z(tgt_pts, anchor_z=self.anchor_z_q, alpha=0.96))

        # Correspondence lines: sweep in → hold → fade out during registration
        line_idx = self.state["line_idx"]
        if stage == "correspondence":
            n_lines    = int(np.ceil(len(line_idx) * reveal))
            corr_alpha = 0.95
        elif stage == "registration":
            n_lines    = len(line_idx)
            corr_alpha = 0.95 * (1.0 - reveal)
        else:
            n_lines    = 0
            corr_alpha = 0.0

        if n_lines > 0 and corr_alpha > 0.01:
            idx = line_idx[:n_lines]
            fade = self.guik.FlatColor(0.18, 1.0, 0.72, float(corr_alpha))
            self._update_segments(
                "correspondences",
                inl_src[idx],
                self.data["tgt_inliers"][idx],
                self.args.line_width,
                shader=fade,
            )
        else:
            try:
                self.viewer.remove_drawable("correspondences")
            except Exception:
                pass

        self.viewer.lookat(self.center)
        self._update_hud(stage)
        alive = self.viewer.spin_once()
        cleanup_imgui_ini(SCRIPT_DIR)
        return alive

    def run(self):
        start = time.perf_counter()
        while not self.viewer.closed():
            elapsed = time.perf_counter() - start
            if not self.render(elapsed):
                break


def main():
    args = parse_args()
    data = load_pipeline_data(args)
    viewer = IridescencePipelineViewer(data, args)
    viewer.run()


if __name__ == "__main__":
    main()
