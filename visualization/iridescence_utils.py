from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def cleanup_imgui_ini(script_dir=None):
    candidate_paths = {Path.cwd() / "imgui.ini"}
    if script_dir is not None:
        candidate_paths.add(Path(script_dir) / "imgui.ini")

    for path in candidate_paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def load_iridescence():
    try:
        import pyridescence.glk as glk
        import pyridescence.guik as guik
    except ImportError as exc:
        raise ImportError("This viewer requires `pyridescence`. Install it first.") from exc
    return guik, glk


def choose_indices(n_items, max_items, seed):
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=max_items, replace=False))


def smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clean_rotation(R):
    R = np.asarray(R, dtype=np.float64)
    u, _, vh = np.linalg.svd(R)
    Rc = u @ vh
    if np.linalg.det(Rc) < 0:
        vh[2] *= -1
        Rc = u @ vh
    return Rc.astype(np.float32)


def make_slerp(R_est):
    return Slerp([0.0, 1.0], Rotation.from_matrix([np.eye(3), R_est.astype(np.float64)]))


def make_turbo_colors_from_z(points, anchor_z=None, alpha=1.0):
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)
    if anchor_z is None:
        anchor_z = float(np.median(points[:, 2]))
    rel = points[:, 2] - anchor_z
    z_min = -0.25 * max(np.ptp(points[:, 2]), 1e-6)
    z_max = max(float(np.percentile(rel, 95.0)), 0.25 * max(np.ptp(points[:, 2]), 1e-6))
    if z_max - z_min < 1e-8:
        return np.full((points.shape[0], 4), (0.5, 0.5, 0.5, alpha), dtype=np.float32)
    t = (0.08 + 0.92 * np.clip((rel - z_min) / (z_max - z_min), 0.0, 1.0)).astype(np.float32)
    t2, t3, t4, t5 = t**2, t**3, t**4, t**5
    r = np.clip(0.13572138 + 4.61539260*t - 42.66032258*t2 + 132.13108234*t3 - 152.94239396*t4 + 59.28637943*t5, 0, 1)
    g = np.clip(0.09140261 + 2.19418839*t +  4.84296658*t2 -  14.18503333*t3 +   4.27729857*t4 +  2.82956604*t5, 0, 1)
    b = np.clip(0.10667330 +12.64194608*t - 60.58204836*t2 + 110.36276771*t3 -  89.90310912*t4 + 27.34824973*t5, 0, 1)
    colors = np.stack([r, g, b], axis=1).astype(np.float32)
    return np.concatenate([colors, np.full((len(t), 1), alpha, dtype=np.float32)], axis=1)


def make_iridescent_colors(points, phase=0.0, alpha=1.0):
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)
    c = points - points.mean(axis=0, keepdims=True)
    scale = max(float(np.percentile(np.linalg.norm(c, axis=1), 90.0)), 1e-6)
    p = c / scale
    w = 2.9*p[:, 0] - 2.2*p[:, 1] + 3.7*p[:, 2] + float(phase)
    r = 0.58 + 0.34*np.sin(w)
    g = 0.62 + 0.28*np.sin(w + 2.1)
    b = 0.72 + 0.26*np.sin(w + 4.2)
    gloss = np.clip(0.75 + 0.35*p[:, 2], 0.55, 1.10)
    colors = np.clip(np.stack([r, g, b], axis=1) * gloss[:, None], 0, 1)
    colors = (0.82 * colors + 0.18).astype(np.float32)
    return np.concatenate([colors, np.full((len(w), 1), alpha, dtype=np.float32)], axis=1)


def make_segment_vertices(starts, ends):
    v = np.empty((starts.shape[0] * 2, 3), dtype=np.float32)
    v[0::2] = starts.astype(np.float32)
    v[1::2] = ends.astype(np.float32)
    return v
