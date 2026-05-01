import math
from pathlib import Path

import numpy as np
import torch


def covariance_generation(elements):
    """
    elements: [B, N, 6]
    return: [B, N, 3, 3]
    """
    B, N, _ = elements.shape
    L = torch.zeros(B, N, 3, 3, device=elements.device, dtype=elements.dtype)
    L[:, :, 0, 0] = elements[..., 0]
    L[:, :, 1, 0] = elements[..., 3]
    L[:, :, 1, 1] = elements[..., 1]
    L[:, :, 2, 0] = elements[..., 4]
    L[:, :, 2, 1] = elements[..., 5]
    L[:, :, 2, 2] = elements[..., 2]
    return L @ L.transpose(-2, -1)


def normals_from_covariance(covariance):
    eig_vals, eig_vecs = torch.linalg.eigh(covariance)
    return eig_vecs[..., 0]


def compute_fpfh_batch(points_b3n, normals_bn3, radius=0.3, max_nn=90):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for FPFH feature extraction.") from exc

    batch_size = points_b3n.shape[0]
    points_np = points_b3n.transpose(1, 2).detach().cpu().numpy()
    normals_np = normals_bn3.detach().cpu().numpy()
    fpfh_list = []

    for i in range(batch_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np[i].astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals_np[i].astype(np.float64))
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)),
        )
        fpfh_list.append(np.asarray(feat.data, dtype=np.float32))

    return torch.tensor(np.stack(fpfh_list, axis=0), dtype=torch.float32, device=points_b3n.device)


def estimate_correspondences(feat_src, feat_tgt):
    """
    feat_src: [D, Ns]
    feat_tgt: [D, Nt]
    return matches: [Ns], source i -> target matches[i]
    """
    dists = torch.cdist(feat_src.T, feat_tgt.T, p=2)
    return torch.argmin(dists, dim=1)


def svd_registration(src_matched, tgt_matched):
    src = np.asarray(src_matched, dtype=np.float64)
    tgt = np.asarray(tgt_matched, dtype=np.float64)
    if src.shape != tgt.shape:
        raise ValueError(f"src/tgt shape mismatch: {src.shape} vs {tgt.shape}")
    if src.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required.")

    src_centroid = np.mean(src, axis=0)
    tgt_centroid = np.mean(tgt, axis=0)
    src_demean = src - src_centroid
    tgt_demean = tgt - tgt_centroid

    H = src_demean.T @ tgt_demean
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tgt_centroid - R @ src_centroid

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _short_array_hash(array):
    import hashlib

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _file_sha256(path):
    import hashlib

    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_degree_stats(graph):
    degrees = np.asarray(
        [graph.GetVertexDegree(i) for i in range(graph.VertexCount())],
        dtype=np.int64,
    )
    if degrees.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(degrees.min()),
        "max": int(degrees.max()),
        "mean": float(degrees.mean()),
    }


def _validate_clique(graph, indices):
    indices = [int(i) for i in indices]
    selected = set(indices)
    for i in indices:
        neighbors = {int(graph.GetVertexEdge(i, k)) for k in range(graph.GetVertexDegree(i))}
        missing = sorted(selected.difference(neighbors).difference({i}))
        if missing:
            return False, (i, missing[:10])
    return True, None


def _run_clique_solver(spark_robin, graph, solver_mode):
    params = spark_robin.MaxCliqueSolver.Params()
    params.solver_mode = solver_mode
    solver = spark_robin.MaxCliqueSolver(params)
    return np.asarray(solver.FindMaxClique(graph), dtype=np.int64)


def _print_robin_result(label, graph, indices, validate=True):
    indices = np.asarray(indices, dtype=np.int64)
    sorted_indices = np.sort(indices)
    if validate:
        valid, invalid_detail = _validate_clique(graph, indices)
        valid_text = str(valid)
    else:
        valid, invalid_detail = None, None
        valid_text = "n/a"
    print(f"[ROBIN:{label}] count={len(indices)} valid_clique={valid_text}")
    print(f"[ROBIN:{label}] sha256_ordered={_short_array_hash(indices)}")
    print(f"[ROBIN:{label}] sha256_sorted={_short_array_hash(sorted_indices)}")
    print(f"[ROBIN:{label}] first50={sorted_indices[:50].tolist()}")
    print(f"[ROBIN:{label}] last50={sorted_indices[-50:].tolist()}")
    if validate and not valid:
        print(f"[ROBIN:{label}] invalid_detail={invalid_detail}")


def maximum_consensus(src_matched, tgt_matched, beta=0.1, debug=False, storage_type="ADJ_LIST"):
    """
    ROBIN maximum clique consensus.
    """
    src = np.asarray(src_matched, dtype=np.float64)
    tgt = np.asarray(tgt_matched, dtype=np.float64)

    import spark_robin

    storage = getattr(spark_robin.GraphStorageType, storage_type)
    graph = spark_robin.Make3dRegInvGraph(
        src.T,
        tgt.T,
        float(beta),
        storage,
    )
    indices = spark_robin.FindInlierStructure(
        graph,
        spark_robin.InlierGraphStructure.MAX_CLIQUE,
    )

    if debug:
        module_path = Path(spark_robin.__file__).resolve()
        print(f"[ROBIN] module={module_path}")
        print(f"[ROBIN] module_sha256={_file_sha256(module_path)}")
        for lib_name in ("libpmc.a", "librobin.a"):
            lib_path = module_path.parent / "lib" / lib_name
            print(f"[ROBIN] {lib_name}_sha256={_file_sha256(lib_path)}")
        print(f"[ROBIN] beta={float(beta)} storage={storage_type}")
        print(f"[ROBIN] src_shape={src.shape} tgt_shape={tgt.shape}")
        print(f"[ROBIN] src_sha256={_short_array_hash(src)}")
        print(f"[ROBIN] tgt_sha256={_short_array_hash(tgt)}")
        print(
            f"[ROBIN] graph_vertices={graph.VertexCount()} "
            f"graph_edges={graph.EdgeCount()} degree_stats={_graph_degree_stats(graph)}"
        )
        max_core = np.asarray(
            spark_robin.FindInlierStructure(
                graph,
                spark_robin.InlierGraphStructure.MAX_CORE,
            ),
            dtype=np.int64,
        )
        _print_robin_result("max_core", graph, max_core, validate=False)
        _print_robin_result("find_max_clique", graph, indices)
        exact = _run_clique_solver(
            spark_robin,
            graph,
            spark_robin.MaxCliqueSolver.CLIQUE_SOLVER_MODE.PMC_EXACT,
        )
        heuristic = _run_clique_solver(
            spark_robin,
            graph,
            spark_robin.MaxCliqueSolver.CLIQUE_SOLVER_MODE.PMC_HEU,
        )
        _print_robin_result("direct_exact", graph, exact)
        _print_robin_result("direct_heuristic", graph, heuristic)

    return src[indices, :], tgt[indices, :]


def rotation_translation_error(R_est, t_est, R_gt, t_gt):
    R_diff = R_est.T @ R_gt
    cos_theta = (np.trace(R_diff) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_error = math.degrees(math.acos(cos_theta))
    trans_error = float(np.linalg.norm(t_est.reshape(3) - t_gt.reshape(3)))
    return rot_error, trans_error


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")
    return checkpoint
