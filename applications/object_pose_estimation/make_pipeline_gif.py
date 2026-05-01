import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation, Slerp

from applications.object_pose_estimation.demo import (
    load_npz_sample,
    make_covariance_normals,
    orient_normals_outward,
)
from applications.object_pose_estimation.model import PointPP
from applications.object_pose_estimation.registration_utils import (
    compute_fpfh_batch,
    estimate_correspondences,
    load_checkpoint,
    maximum_consensus,
    rotation_translation_error,
    svd_registration,
)


# ============================================================
# Colors
# ============================================================

SOURCE_COLOR = np.array([1.0, 0.0, 0.85])      # magenta
TARGET_COLOR = np.array([0.0, 0.32, 1.0])      # blue
NORMAL_COLOR = np.array([1.0, 0.85, 0.0])      # yellow
CORR_COLOR = np.array([0.0, 0.78, 0.2])        # green

COV_SOURCE_COLOR = np.array([0.95, 0.35, 0.10])
COV_TARGET_COLOR = np.array([0.10, 0.45, 0.95])

BG_COLOR = "white"


# ============================================================
# Pipeline labels
# ============================================================

STEP_LABELS = [
    "1. POLI Predicts Local Geometry",
    "2. Normal Estimation",
    "3. FPFH Correspondence",
    "4. Registration",
]

STAGE_TO_STEP = {
    "covariance": 0,
    "normals": 1,
    "correspondence": 2,
    "registration": 3,
}


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a GIF/MP4 of the POLI object registration pipeline."
    )

    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7777)

    parser.add_argument("--fpfh_radius", type=float, default=0.2)
    parser.add_argument("--fpfh_max_nn", type=int, default=90)
    parser.add_argument("--consensus_beta", type=float, default=0.1)

    parser.add_argument("--scan_separation", type=float, default=2.2)

    parser.add_argument("--max_points", type=int, default=800)
    parser.add_argument("--max_covariances", type=int, default=300)
    parser.add_argument("--max_normals", type=int, default=300)
    parser.add_argument("--max_lines", type=int, default=300)

    parser.add_argument("--normal_length", type=float, default=0.06)
    parser.add_argument("--covariance_scale", type=float, default=0.06)

    # Larger value means more zoom-in.
    parser.add_argument("--view_zoom", type=float, default=0.9)

    parser.add_argument("--covariance_frames", type=int, default=30)
    parser.add_argument("--normal_frames", type=int, default=30)
    parser.add_argument("--correspondence_frames", type=int, default=30)
    parser.add_argument("--registration_frames", type=int, default=70)
    parser.add_argument("--hold_frames", type=int, default=14)

    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)

    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================

def choose_indices(n_items, max_items, seed):
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=max_items, replace=False))


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clean_rotation(R):
    u, _, vh = np.linalg.svd(R)
    R_clean = u @ vh

    if np.linalg.det(R_clean) < 0:
        vh[2, :] *= -1
        R_clean = u @ vh

    return R_clean


def make_output_path(sample_path, output):
    if output is not None:
        return Path(output).expanduser().resolve()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"{sample_path.stem}_poli_pipeline.gif"


# ============================================================
# Load data and run pipeline
# ============================================================

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
        cov_p, normals_p = make_covariance_normals(model, P)
        cov_q, normals_q = make_covariance_normals(model, Q)

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

        matches = estimate_correspondences(
            fpfh_p[0],
            fpfh_q[0],
        ).detach().cpu().numpy()

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

    normals_p_np = orient_normals_outward(
        P_np,
        normals_p[0].detach().cpu().numpy(),
    )

    normals_q_np = orient_normals_outward(
        Q_np,
        normals_q[0].detach().cpu().numpy(),
    )

    print(f"[Correspondence] neural FPFH matches = {len(matches)}")
    print(f"[Consensus] inliers = {len(src_inliers)} / {len(src_matched)}")
    print(f"[GT error] rotation = {rot_error:.4f} deg | translation = {trans_error:.6f}")

    return {
        "sample_path": sample_path,
        "P_np": P_np,
        "Q_np": Q_np,
        "cov_p": cov_p[0].detach().cpu().numpy(),
        "cov_q": cov_q[0].detach().cpu().numpy(),
        "normals_p": normals_p_np,
        "normals_q": normals_q_np,
        "matches": matches,
        "src_inliers": src_inliers,
        "tgt_inliers": tgt_inliers,
        "R_est": R_est,
        "t_est": t_est,
        "rot_error": rot_error,
        "trans_error": trans_error,
    }


# ============================================================
# Camera / view
# ============================================================

def compute_view_limits(data, args):
    offset = np.array([args.scan_separation, 0.0, 0.0], dtype=np.float64)

    P_np = data["P_np"]
    Q_np = data["Q_np"]

    P_aligned = (P_np @ data["R_est"].T) + data["t_est"].reshape(1, 3)

    display_pts = np.vstack([
        P_np + offset,
        Q_np,
    ])

    all_pts = np.vstack([
        display_pts,
        P_aligned,
    ])

    center = display_pts.mean(axis=0)

    # Slightly shift the object to the right visual region.
    center[0] += 0.05

    pad = 0.02

    def half(pts, axis):
        lo, hi = pts[:, axis].min(), pts[:, axis].max()
        span = hi - lo
        return span * (0.5 + pad) / max(float(args.view_zoom), 1e-6)

    x_half = half(display_pts, 0)
    y_half = half(display_pts, 1)
    z_half = half(all_pts, 2)

    return center, x_half, y_half, z_half


def setup_axis(ax, center, x_half, y_half, z_half):
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor(BG_COLOR)

    ax.set_xlim(center[0] - x_half, center[0] + x_half)
    ax.set_ylim(center[1] - y_half, center[1] + y_half)
    ax.set_zlim(center[2] - z_half, center[2] + z_half)

    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.view_init(elev=120, azim=-90)
    ax.set_proj_type("persp")
    ax.set_axis_off()

# ============================================================
# Drawing functions
# ============================================================

def draw_step_labels(fig, current_step):
    # fig.text(
    #     0.055,
    #     0.955,
    #     "POLI Registration Pipeline",
    #     color="#222222",
    #     fontsize=18,
    #     fontweight="bold",
    #     ha="left",
    #     va="top",
    # )

    y_start = 0.84
    y_gap = 0.115

    for i, label in enumerate(STEP_LABELS):
        active = i == current_step

        fig.text(
            0.055,
            y_start - i * y_gap,
            label,
            color="#111111" if active else "#c6c6c6",
            fontsize=25 if active else 18,
            fontweight="bold" if active else "normal",
            ha="left",
            va="center",
        )


def scatter_cloud(ax, points, color, size, alpha=0.93):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=size,
        c=[color],
        alpha=alpha,
        depthshade=False,
        linewidths=0,
    )


def draw_line_collection(
    ax,
    src_points,
    tgt_points,
    color,
    linewidth=1.5,
    alpha=0.85,
):
    if len(src_points) == 0:
        return

    segments = np.stack([src_points, tgt_points], axis=1)

    collection = Line3DCollection(
        segments,
        colors=[(*color, alpha)],
        linewidths=linewidth,
    )

    ax.add_collection3d(collection)


def covariance_norm(covariances):
    vals = np.linalg.eigvalsh(covariances)
    radii = np.sqrt(np.clip(vals, 1e-12, None))

    norm = np.percentile(radii, 75)
    return max(float(norm), 1e-6)


def draw_covariance_ellipsoid(
    ax,
    center,
    covariance,
    color,
    cov_norm,
    scale,
    alpha=0.80,
):
    vals, vecs = np.linalg.eigh(covariance)

    radii = np.sqrt(np.clip(vals, 1e-12, None))
    radii = radii / cov_norm * scale
    radii = np.clip(radii, scale * 0.18, scale * 2.2)

    u = np.linspace(0.0, 2.0 * np.pi, 12)
    v = np.linspace(0.0, np.pi, 7)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    xyz = np.stack([x, y, z], axis=0).reshape(3, -1)
    ellipsoid = (vecs @ xyz).T + center.reshape(1, 3)

    X = ellipsoid[:, 0].reshape(len(u), len(v))
    Y = ellipsoid[:, 1].reshape(len(u), len(v))
    Z = ellipsoid[:, 2].reshape(len(u), len(v))

    ax.plot_wireframe(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        color=color,
        linewidth=1.4,
        alpha=alpha,
    )


def draw_covariances(
    ax,
    points,
    covariances,
    indices,
    color,
    offset,
    progress,
    cov_norm,
    scale,
):
    n_show = int(np.ceil(len(indices) * smoothstep(progress)))

    if n_show <= 0:
        return

    for idx in indices[:n_show]:
        draw_covariance_ellipsoid(
            ax,
            points[idx] + offset,
            covariances[idx],
            color=color,
            cov_norm=cov_norm,
            scale=scale,
        )


def draw_normals(
    ax,
    points,
    normals,
    indices,
    color,
    offset,
    progress,
    length,
):
    n_show = int(np.ceil(len(indices) * smoothstep(progress)))

    if n_show <= 0:
        return

    idx = indices[:n_show]

    starts = points[idx] + offset
    vecs = normals[idx]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    ends = starts + vecs * length

    draw_line_collection(
        ax,
        starts,
        ends,
        color=color,
        linewidth=2.7,
        alpha=0.95,
    )


def make_slerp(R_est):
    rotations = Rotation.from_matrix([
        np.eye(3),
        R_est,
    ])

    return Slerp([0.0, 1.0], rotations)


# ============================================================
# Rendering
# ============================================================

def render_frame(data, args, stage, progress, state):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)

    # Right-side large 3D view.
    # [left, bottom, width, height] in figure coordinates.
    ax = fig.add_axes([0.40, 0.03, 0.58, 0.92], projection="3d")

    setup_axis(
        ax,
        state["center"],
        state["x_half"],
        state["y_half"],
        state["z_half"],
    )

    current_step = STAGE_TO_STEP[stage]
    draw_step_labels(fig, current_step)

    P_np = data["P_np"]
    Q_np = data["Q_np"]

    offset = np.array([args.scan_separation, 0.0, 0.0], dtype=np.float64)

    point_idx_p = state["point_idx_p"]
    point_idx_q = state["point_idx_q"]

    if stage in {"covariance", "normals", "correspondence"}:
        P_display = P_np + offset
    else:
        alpha = smoothstep(progress)
        R_alpha = state["slerp"]([alpha]).as_matrix()[0]

        P_display = (P_np @ R_alpha.T) + alpha * data["t_est"].reshape(1, 3)
        P_display = P_display + (1.0 - alpha) * offset.reshape(1, 3)

    scatter_cloud(
        ax,
        Q_np[point_idx_q],
        TARGET_COLOR,
        size=30.0,
        alpha=0.94,
    )

    scatter_cloud(
        ax,
        P_display[point_idx_p],
        SOURCE_COLOR,
        size=30.0,
        alpha=0.94,
    )

    # ------------------------------------------------------------
    # Stage 1: covariance
    # ------------------------------------------------------------
    if stage in {"covariance", "normals", "correspondence"}:
        cov_progress = progress if stage == "covariance" else 1.0

        draw_covariances(
            ax,
            P_np,
            data["cov_p"],
            state["cov_idx_p"],
            COV_SOURCE_COLOR,
            offset,
            cov_progress,
            state["cov_norm"],
            args.covariance_scale,
        )

        draw_covariances(
            ax,
            Q_np,
            data["cov_q"],
            state["cov_idx_q"],
            COV_TARGET_COLOR,
            np.zeros(3),
            cov_progress,
            state["cov_norm"],
            args.covariance_scale,
        )

    # ------------------------------------------------------------
    # Stage 2: normals
    # ------------------------------------------------------------
    if stage in {"normals", "correspondence"}:
        normal_progress = progress if stage == "normals" else 1.0

        draw_normals(
            ax,
            P_np,
            data["normals_p"],
            state["normal_idx_p"],
            NORMAL_COLOR,
            offset,
            normal_progress,
            args.normal_length,
        )

        draw_normals(
            ax,
            Q_np,
            data["normals_q"],
            state["normal_idx_q"],
            NORMAL_COLOR,
            np.zeros(3),
            normal_progress,
            args.normal_length,
        )

    # ------------------------------------------------------------
    # Stage 3: correspondence
    # ------------------------------------------------------------
    if stage == "correspondence":
        n_show = int(np.ceil(len(state["line_idx"]) * smoothstep(progress)))

        if n_show > 0:
            idx = state["line_idx"][:n_show]

            draw_line_collection(
                ax,
                data["src_inliers"][idx] + offset,
                data["tgt_inliers"][idx],
                CORR_COLOR,
                linewidth=1.65,
                alpha=0.82,
            )

    # ------------------------------------------------------------
    # Stage 4: registration
    # ------------------------------------------------------------
    elif stage == "registration":
        alpha = smoothstep(progress)

        R_alpha = state["slerp"]([alpha]).as_matrix()[0]
        t_alpha = alpha * data["t_est"].reshape(1, 3)
        offset_alpha = (1.0 - alpha) * offset.reshape(1, 3)

        P_moved = (P_np @ R_alpha.T) + t_alpha + offset_alpha
        normals_p_rot = data["normals_p"] @ R_alpha.T

        draw_normals(
            ax,
            P_moved,
            normals_p_rot,
            state["normal_idx_p"],
            NORMAL_COLOR,
            np.zeros(3),
            1.0,
            args.normal_length,
        )

        draw_normals(
            ax,
            Q_np,
            data["normals_q"],
            state["normal_idx_q"],
            NORMAL_COLOR,
            np.zeros(3),
            1.0,
            args.normal_length,
        )

        src_lines = (data["src_inliers"] @ R_alpha.T) + t_alpha + offset_alpha
        idx = state["line_idx"]

        draw_line_collection(
            ax,
            src_lines[idx],
            data["tgt_inliers"][idx],
            CORR_COLOR,
            linewidth=1.65,
            alpha=0.75,
        )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    plt.close(fig)

    return frame


# ============================================================
# Render state and frame schedule
# ============================================================

def make_render_state(data, args):
    center, x_half, y_half, z_half = compute_view_limits(data, args)

    P_np = data["P_np"]
    Q_np = data["Q_np"]

    line_count = len(data["src_inliers"])

    cov_norm = covariance_norm(
        np.concatenate(
            [
                data["cov_p"],
                data["cov_q"],
            ],
            axis=0,
        )
    )

    return {
        "center": center,
        "x_half": x_half,
        "y_half": y_half,
        "z_half": z_half,
        "point_idx_p": choose_indices(len(P_np), args.max_points, args.seed + 1),
        "point_idx_q": choose_indices(len(Q_np), args.max_points, args.seed + 2),
        "cov_idx_p": choose_indices(len(P_np), args.max_covariances, args.seed + 3),
        "cov_idx_q": choose_indices(len(Q_np), args.max_covariances, args.seed + 4),
        "normal_idx_p": choose_indices(len(P_np), args.max_normals, args.seed + 5),
        "normal_idx_q": choose_indices(len(Q_np), args.max_normals, args.seed + 6),
        "line_idx": choose_indices(
            line_count,
            min(args.max_lines, line_count),
            args.seed + 7,
        ),
        "cov_norm": cov_norm,
        "slerp": make_slerp(data["R_est"]),
    }


def frame_schedule(args):
    schedule = []

    for i in range(max(args.covariance_frames, 1)):
        progress = i / max(args.covariance_frames - 1, 1)
        schedule.append(("covariance", progress))

    for i in range(max(args.normal_frames, 1)):
        progress = i / max(args.normal_frames - 1, 1)
        schedule.append(("normals", progress))

    for i in range(max(args.correspondence_frames, 1)):
        progress = i / max(args.correspondence_frames - 1, 1)
        schedule.append(("correspondence", progress))

    for i in range(max(args.registration_frames, 1)):
        progress = i / max(args.registration_frames - 1, 1)
        schedule.append(("registration", progress))

    for _ in range(max(args.hold_frames, 0)):
        schedule.append(("registration", 1.0))

    return schedule


# ============================================================
# Write animation
# ============================================================

def write_animation(data, args):
    output_path = make_output_path(data["sample_path"], args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = make_render_state(data, args)
    schedule = frame_schedule(args)

    print(f"[Render] frames = {len(schedule)} | fps = {args.fps}")
    print(f"[Output] {output_path}")

    suffix = output_path.suffix.lower()

    if suffix == ".mp4":
        writer_kwargs = {
            "fps": args.fps,
            "macro_block_size": 1,
        }
    else:
        writer_kwargs = {
            "mode": "I",
            "duration": 1.0 / max(args.fps, 1),
            "loop": 0,
        }

    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        for frame_idx, (stage, progress) in enumerate(schedule, start=1):
            frame = render_frame(
                data,
                args,
                stage,
                progress,
                state,
            )

            writer.append_data(frame)

            if frame_idx == 1 or frame_idx == len(schedule) or frame_idx % 10 == 0:
                print(f"[Render] {frame_idx} / {len(schedule)}")


# ============================================================
# Main
# ============================================================

def main(args):
    data = load_pipeline_data(args)
    write_animation(data, args)


if __name__ == "__main__":
    main(parse_args())
