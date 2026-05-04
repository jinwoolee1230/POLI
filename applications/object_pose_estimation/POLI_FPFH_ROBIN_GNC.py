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
for _p in (SCRIPT_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from registration_utils import (
    compute_fpfh_batch,
    estimate_correspondences,
    gnc_registration,
    load_checkpoint,
    maximum_consensus,
    rotation_translation_error,
)
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

STAGES = [
    ("covariance",    "1. Neural Covariance"),
    ("normals",       "2. Surface Normals"),
    ("correspondence","3. FPFH Consensus"),
    ("registration",  "4. Registration"),
]

DEMO_DEFAULTS = {
    "device":          "cuda",
    "fpfh_radius":     0.2,
    "fpfh_max_nn":     90,
    "consensus_beta":  0.1,
    "gnc_solver":      "tls",
    "gnc_noise_bound": 0.05,
    "max_lines":       1000,
    "max_normals":     1000,
    "normal_length":   0.15,
    "seed":            7777,
}


atexit.register(cleanup_imgui_ini, SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Iridescence-based POLI object pose pipeline viewer.")
    parser.add_argument("--sample",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device",     type=str, default=DEMO_DEFAULTS["device"])
    parser.add_argument("--seed",       type=int, default=DEMO_DEFAULTS["seed"])

    parser.add_argument("--fpfh_radius",    type=float, default=DEMO_DEFAULTS["fpfh_radius"])
    parser.add_argument("--fpfh_max_nn",    type=int,   default=DEMO_DEFAULTS["fpfh_max_nn"])
    parser.add_argument("--consensus_beta", type=float, default=DEMO_DEFAULTS["consensus_beta"])
    parser.add_argument("--gnc_solver",     choices=("tls", "gm"), default=DEMO_DEFAULTS["gnc_solver"])
    parser.add_argument("--gnc_noise_bound",type=float, default=DEMO_DEFAULTS["gnc_noise_bound"])

    parser.add_argument("--scan_separation",  type=float, default=2.2)
    parser.add_argument("--max_points",       type=int,   default=2400)
    parser.add_argument("--max_covariances",  type=int,   default=420)
    parser.add_argument("--max_normals",      type=int,   default=DEMO_DEFAULTS["max_normals"])
    parser.add_argument("--max_lines",        type=int,   default=DEMO_DEFAULTS["max_lines"])
    parser.add_argument("--normal_length",    type=float, default=DEMO_DEFAULTS["normal_length"])
    parser.add_argument("--covariance_scale", type=float, default=0.06)

    parser.add_argument("--point_size",   type=float, default=0.009)
    parser.add_argument("--line_width",   type=float, default=0.95)
    parser.add_argument("--normal_width", type=float, default=0.7)

    parser.add_argument("--camera_theta",    type=float, default=0.36)
    parser.add_argument("--camera_phi",      type=float, default=-0.92)
    parser.add_argument("--camera_distance", type=float, default=None)
    parser.add_argument("--grid",            action="store_true")

    parser.add_argument("--covariance_seconds",    type=float, default=1.7)
    parser.add_argument("--normal_seconds",        type=float, default=1.7)
    parser.add_argument("--correspondence_seconds",type=float, default=1.7)
    parser.add_argument("--registration_seconds",  type=float, default=2.4)
    parser.add_argument("--hold_seconds",          type=float, default=1.8)
    parser.add_argument("--playback_rate",         type=float, default=1.0)
    parser.add_argument("--loop",                  action="store_true")

    return parser.parse_args()


def transform_covariances(covariances, R):
    return R @ covariances @ R.T


def load_npz_sample(path):
    sample = np.load(path)
    missing = {"P", "Q", "R_rel", "t_rel"}.difference(sample.files)
    if missing:
        raise KeyError(f"Sample missing keys: {sorted(missing)}")
    P = sample["P"].astype(np.float32)
    Q = sample["Q"].astype(np.float32)
    if P.shape[0] != 3 or Q.shape[0] != 3:
        raise ValueError(f"Expected (3,N), got P={P.shape} Q={Q.shape}")
    return {"P": P, "Q": Q,
            "R_rel": sample["R_rel"].astype(np.float32),
            "t_rel": sample["t_rel"].astype(np.float32).reshape(3)}


def make_covariance_normals(model, points_b3n):
    from registration_utils import covariance_generation, normals_from_covariance
    centered = points_b3n - points_b3n.mean(dim=2, keepdim=True)
    cov_elems = model(centered)
    covariances = covariance_generation(cov_elems.permute(0, 2, 1))
    normals = normals_from_covariance(covariances)
    return covariances, normals


def orient_normals_outward(points, normals):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    flip = np.sum(normals * (points - center), axis=1, keepdims=True) < 0.0
    return np.where(flip, -normals, normals).astype(np.float32)


def make_scene_center_distance(P_np, Q_np, offset):
    src_off = P_np + offset.reshape(1, 3)
    center = ((src_off.mean(axis=0) + Q_np.mean(axis=0)) * 0.5).astype(np.float32)
    center[0] += 0.08 * max(np.ptp(np.vstack([src_off, Q_np])[:, 0]), 1e-6)
    scene = np.vstack([src_off, Q_np])
    extent = scene.max(axis=0) - scene.min(axis=0)
    distance = max(float(np.max(extent)) * 2.6, 0.6)
    return center, distance


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

    sample = load_npz_sample(sample_path)
    P_np = sample["P"].T
    Q_np = sample["Q"].T

    P = torch.from_numpy(sample["P"]).unsqueeze(0).to(device)
    Q = torch.from_numpy(sample["Q"]).unsqueeze(0).to(device)

    from model.pointnetpp_obj import PointPP
    model = PointPP().to(device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    with torch.no_grad():
        cov_p, normals_p = make_covariance_normals(model, P)
        cov_q, normals_q = make_covariance_normals(model, Q)
        fpfh_p = compute_fpfh_batch(P, normals_p, radius=args.fpfh_radius, max_nn=args.fpfh_max_nn)
        fpfh_q = compute_fpfh_batch(Q, normals_q, radius=args.fpfh_radius, max_nn=args.fpfh_max_nn)
        matches = estimate_correspondences(fpfh_p[0], fpfh_q[0]).detach().cpu().numpy()

    src_matched = P_np
    tgt_matched = Q_np[matches]
    src_inliers, tgt_inliers = maximum_consensus(src_matched, tgt_matched, beta=args.consensus_beta)
    if len(src_inliers) < 3:
        raise RuntimeError("Not enough inliers.")

    gnc_nb = args.gnc_noise_bound if args.gnc_noise_bound is not None else args.consensus_beta
    R_raw, t_est = gnc_registration(src_inliers, tgt_inliers, solver=args.gnc_solver, noise_bound=gnc_nb)
    R_est = clean_rotation(R_raw)

    rot_err, trans_err = rotation_translation_error(R_est, t_est, sample["R_rel"], sample["t_rel"])

    normals_p_np = orient_normals_outward(P_np, normals_p[0].detach().cpu().numpy())
    normals_q_np = orient_normals_outward(Q_np, normals_q[0].detach().cpu().numpy())

    return {
        "sample_path": sample_path,
        "P_np":        P_np.astype(np.float32),
        "Q_np":        Q_np.astype(np.float32),
        "cov_p":       cov_p[0].detach().cpu().numpy().astype(np.float32),
        "cov_q":       cov_q[0].detach().cpu().numpy().astype(np.float32),
        "normals_p":   normals_p_np.astype(np.float32),
        "normals_q":   normals_q_np.astype(np.float32),
        "src_inliers": src_inliers.astype(np.float32),
        "tgt_inliers": tgt_inliers.astype(np.float32),
        "R_est":       R_est,
        "t_est":       t_est.astype(np.float32),
        "rot_error":   float(rot_err),
        "trans_error": float(trans_err),
        "gnc_solver":  args.gnc_solver,
        "gnc_noise_bound": float(gnc_nb),
    }


class IridescencePipelineViewer:
    def __init__(self, data, args):
        self.guik, self.glk = load_iridescence()
        self.data = data
        self.args = args

        self.offset = np.array([args.scan_separation, 0.0, 0.0], dtype=np.float32)
        self.center, auto_dist = make_scene_center_distance(data["P_np"], data["Q_np"], self.offset)
        self.camera_distance = args.camera_distance if args.camera_distance is not None else auto_dist

        self.stage_durations = [
            ("covariance",    float(args.covariance_seconds)),
            ("normals",       float(args.normal_seconds)),
            ("correspondence",float(args.correspondence_seconds)),
            ("registration",  float(args.registration_seconds)),
            ("hold",          float(args.hold_seconds)),
        ]
        self.total_duration = sum(d for _, d in self.stage_durations)

        n_inl     = len(data["src_inliers"])
        raw_line  = choose_indices(n_inl, min(args.max_lines, n_inl), args.seed + 7)
        sweep_ord = np.argsort(data["src_inliers"][raw_line, 0])

        self.state = {
            "point_idx_p":  choose_indices(len(data["P_np"]), args.max_points, args.seed + 1),
            "point_idx_q":  choose_indices(len(data["Q_np"]), args.max_points, args.seed + 2),
            "cov_idx_p":    choose_indices(len(data["P_np"]), args.max_covariances, args.seed + 3),
            "cov_idx_q":    choose_indices(len(data["Q_np"]), args.max_covariances, args.seed + 4),
            "normal_idx_p": choose_indices(len(data["P_np"]), args.max_normals, args.seed + 5),
            "normal_idx_q": choose_indices(len(data["Q_np"]), args.max_normals, args.seed + 6),
            "line_idx":     raw_line[sweep_ord],
            "slerp":        make_slerp(data["R_est"]),
        }
        self.anchor_z_q = float(np.median(data["Q_np"][:, 2]))

        self.viewer = self.guik.LightViewer.instance(
            title=f"POLI Iridescence | {data['sample_path'].stem}"
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

        self.target_cov_shader = self._flat(0.14, 0.55, 1.0, 0.11).make_transparent()
        self.source_cov_shader = self._flat(1.0,  0.42, 0.92, 0.14).make_transparent()
        self.normal_tgt_shader = self._flat(0.18, 0.98, 0.92, 0.96)
        self.normal_src_shader = self._flat(1.0,  0.78, 0.32, 0.96)
        self.corr_shader       = self._flat(0.18, 1.0,  0.72, 0.95)

    def _flat(self, r, g, b, a):
        s = self.guik.FlatColor(r, g, b, a)
        s.set_point_shape_circle()
        s.set_point_scale_metric()
        s.set_point_size(self.args.point_size)
        return s

    def _scene_time(self, elapsed):
        if self.total_duration <= 0.0:
            return 0.0
        if self.args.loop:
            return elapsed % self.total_duration
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
        buf = self.glk.PointCloudBuffer(points.astype(np.float32))
        buf.add_color(colors.astype(np.float32))
        self.viewer.update_drawable(name, buf, self.vertex_shader)

    def _update_covariances(self, name, points, covs, indices, count, shader):
        if count <= 0:
            return
        idx = indices[:count]
        self.viewer.update_normal_dists(
            name,
            [p.astype(np.float32) for p in points[idx]],
            [c.astype(np.float32) for c in covs[idx]],
            self.args.covariance_scale,
            shader,
        )

    def _update_segments(self, name, starts, ends, shader, width):
        if starts.shape[0] == 0:
            return
        lines = self.viewer.update_thin_lines(
            name, make_segment_vertices(starts, ends),
            line_strip=False, shader_setting=shader,
        )
        if lines is not None:
            lines.set_line_width(float(width))

    def _normal_segments(self, points, normals, indices, count):
        if count <= 0:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        idx = indices[:count]
        vecs = normals[idx]
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        ends = points[idx] + vecs * float(self.args.normal_length)
        return points[idx].astype(np.float32), ends.astype(np.float32)

    def _update_hud(self, stage):
        lines = ["POLI Iridescence", ""]
        for name, label in STAGES:
            prefix = ">" if name == stage or (stage == "hold" and name == "registration") else " "
            lines.append(f"{prefix} {label}")
        d = self.data
        lines += [
            "",
            f"sample: {d['sample_path'].name}",
            f"inliers: {len(d['src_inliers'])}",
            f"gnc: {d['gnc_solver']} | noise_bound: {d['gnc_noise_bound']}",
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

        P_np = self.data["P_np"]
        Q_np = self.data["Q_np"]

        if stage in {"covariance", "normals", "correspondence"}:
            P_disp      = P_np + self.offset
            normals_disp = self.data["normals_p"]
            cov_p_disp  = self.data["cov_p"]
            inl_disp    = self.data["src_inliers"] + self.offset
        else:
            alpha = smoothstep(progress if stage == "registration" else 1.0)
            R_a   = self.state["slerp"]([alpha]).as_matrix()[0].astype(np.float32)
            t_a   = alpha * self.data["t_est"].reshape(1, 3)
            off_a = (1.0 - alpha) * self.offset.reshape(1, 3)
            P_disp      = (P_np @ R_a.T) + t_a + off_a
            normals_disp = self.data["normals_p"] @ R_a.T
            cov_p_disp  = np.stack(
                [transform_covariances(c, R_a) for c in self.data["cov_p"]], axis=0
            ).astype(np.float32)
            inl_disp = (self.data["src_inliers"] @ R_a.T) + t_a + off_a

        src_pts = P_disp[self.state["point_idx_p"]]
        tgt_pts = Q_np[self.state["point_idx_q"]]

        self._update_points("source_points", src_pts,
                            make_iridescent_colors(src_pts, phase=shimmer, alpha=0.98))
        self._update_points("target_points", tgt_pts,
                            make_turbo_colors_from_z(tgt_pts, anchor_z=self.anchor_z_q, alpha=0.96))

        # Covariances
        if stage == "covariance":
            cov_cnt_p = int(np.ceil(len(self.state["cov_idx_p"]) * reveal))
            cov_cnt_q = int(np.ceil(len(self.state["cov_idx_q"]) * reveal))
        else:
            cov_cnt_p = len(self.state["cov_idx_p"])
            cov_cnt_q = len(self.state["cov_idx_q"])

        self._update_covariances("source_covariances", P_disp, cov_p_disp,
                                 self.state["cov_idx_p"], cov_cnt_p, self.source_cov_shader)
        self._update_covariances("target_covariances", Q_np, self.data["cov_q"],
                                 self.state["cov_idx_q"], cov_cnt_q, self.target_cov_shader)

        # Normals
        if stage == "normals":
            n_cnt_p = int(np.ceil(len(self.state["normal_idx_p"]) * reveal))
            n_cnt_q = int(np.ceil(len(self.state["normal_idx_q"]) * reveal))
        elif stage in {"correspondence", "registration", "hold"}:
            n_cnt_p = len(self.state["normal_idx_p"])
            n_cnt_q = len(self.state["normal_idx_q"])
        else:
            n_cnt_p = n_cnt_q = 0

        ns_s, ne_s = self._normal_segments(P_disp, normals_disp, self.state["normal_idx_p"], n_cnt_p)
        nt_s, nt_e = self._normal_segments(Q_np, self.data["normals_q"], self.state["normal_idx_q"], n_cnt_q)
        self._update_segments("source_normals", ns_s, ne_s, self.normal_src_shader, self.args.normal_width)
        self._update_segments("target_normals", nt_s, nt_e, self.normal_tgt_shader, self.args.normal_width)

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
            idx  = line_idx[:n_lines]
            fade = self._flat(0.18, 1.0, 0.72, float(corr_alpha))
            self._update_segments("correspondences",
                                  inl_disp[idx], self.data["tgt_inliers"][idx],
                                  fade, self.args.line_width)
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
            elapsed = (time.perf_counter() - start) * float(self.args.playback_rate)
            if not self.render(elapsed):
                break


def main():
    args = parse_args()
    data = load_pipeline_data(args)
    viewer = IridescencePipelineViewer(data, args)
    viewer.run()


if __name__ == "__main__":
    main()
