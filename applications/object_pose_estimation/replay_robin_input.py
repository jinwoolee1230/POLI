import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from applications.object_pose_estimation.registration_utils import (
    _file_sha256,
    _graph_degree_stats,
    _print_robin_result,
    _run_clique_solver,
    _short_array_hash,
    _validate_clique,
    rotation_translation_error,
    svd_registration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay only the ROBIN max-clique stage from dumped demo inputs."
    )
    parser.add_argument("dump", type=str)
    parser.add_argument(
        "--storage",
        type=str,
        default="ALL",
        choices=("ALL", "ADJ_LIST", "CSR", "ATOMIC_CSR"),
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional sample npz with R_rel/t_rel for pose-error reporting.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat ROBIN from the same dumped input to detect nondeterministic PMC output.",
    )
    return parser.parse_args()


def load_sample(path):
    sample = np.load(path)
    return {
        "R_rel": sample["R_rel"].astype(np.float64),
        "t_rel": sample["t_rel"].astype(np.float64).reshape(3),
    }


def pose_error_text(src, tgt, indices, sample):
    if sample is None or len(indices) < 3:
        return ""
    T_est = svd_registration(src[indices], tgt[indices])
    rot_error, trans_error = rotation_translation_error(
        T_est[:3, :3],
        T_est[:3, 3],
        sample["R_rel"],
        sample["t_rel"],
    )
    return f" pose_rot={rot_error:.6f}deg pose_trans={trans_error:.9f}"


def run_storage_once(spark_robin, src, tgt, beta, storage_name, sample, detailed):
    storage = getattr(spark_robin.GraphStorageType, storage_name)
    graph = spark_robin.Make3dRegInvGraph(src.T, tgt.T, float(beta), storage)

    max_core = np.asarray(
        spark_robin.FindInlierStructure(graph, spark_robin.InlierGraphStructure.MAX_CORE),
        dtype=np.int64,
    )
    find_max_clique = np.asarray(
        spark_robin.FindInlierStructure(graph, spark_robin.InlierGraphStructure.MAX_CLIQUE),
        dtype=np.int64,
    )
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

    if detailed:
        _print_robin_result("max_core", graph, max_core, validate=False)
        _print_robin_result("find_max_clique", graph, find_max_clique)
        _print_robin_result("direct_exact", graph, exact)
        _print_robin_result("direct_heuristic", graph, heuristic)
        if sample is not None and len(find_max_clique) >= 3:
            print(f"[ROBIN]{pose_error_text(src, tgt, find_max_clique, sample)}")
    else:
        find_valid, _ = _validate_clique(graph, find_max_clique)
        exact_valid, _ = _validate_clique(graph, exact)
        heuristic_valid, _ = _validate_clique(graph, heuristic)
        print(
            "[ROBIN:repeat] "
            f"find_count={len(find_max_clique)} "
            f"find_sorted={_short_array_hash(np.sort(find_max_clique))} "
            f"find_valid={find_valid}"
            f"{pose_error_text(src, tgt, find_max_clique, sample)}"
        )
        print(
            "[ROBIN:repeat] "
            f"exact_count={len(exact)} "
            f"exact_sorted={_short_array_hash(np.sort(exact))} "
            f"exact_valid={exact_valid}"
            f"{pose_error_text(src, tgt, exact, sample)}"
        )
        print(
            "[ROBIN:repeat] "
            f"heuristic_count={len(heuristic)} "
            f"heuristic_sorted={_short_array_hash(np.sort(heuristic))} "
            f"heuristic_valid={heuristic_valid}"
            f"{pose_error_text(src, tgt, heuristic, sample)}"
        )

    return graph


def run_storage(spark_robin, src, tgt, beta, storage_name, sample, repeat):
    print(f"\n[ROBIN] storage={storage_name}")
    if repeat <= 1:
        graph = run_storage_once(
            spark_robin,
            src,
            tgt,
            beta,
            storage_name,
            sample,
            detailed=True,
        )
        print(
            f"[ROBIN] graph_vertices={graph.VertexCount()} "
            f"graph_edges={graph.EdgeCount()} degree_stats={_graph_degree_stats(graph)}"
        )
        return

    for i in range(repeat):
        print(f"[ROBIN:repeat] iteration={i + 1}/{repeat}")
        graph = run_storage_once(
            spark_robin,
            src,
            tgt,
            beta,
            storage_name,
            sample,
            detailed=False,
        )
        print(f"[ROBIN:repeat] vertices={graph.VertexCount()} edges={graph.EdgeCount()}")


def main(args):
    import spark_robin

    dump = np.load(Path(args.dump).expanduser().resolve())
    src = dump["src_matched"].astype(np.float64)
    tgt = dump["tgt_matched"].astype(np.float64)
    beta = float(dump["beta"])
    sample_path = args.sample
    if sample_path is None and "sample_path" in dump.files:
        sample_path = str(dump["sample_path"].item())
    sample = load_sample(sample_path) if sample_path else None

    module_path = Path(spark_robin.__file__).resolve()
    print(f"[ROBIN] module={module_path}")
    print(f"[ROBIN] module_sha256={_file_sha256(module_path)}")
    for lib_name in ("libpmc.a", "librobin.a"):
        lib_path = module_path.parent / "lib" / lib_name
        print(f"[ROBIN] {lib_name}_sha256={_file_sha256(lib_path)}")
    print(f"[ROBIN] dump={Path(args.dump).expanduser().resolve()}")
    print(f"[ROBIN] src_shape={src.shape} tgt_shape={tgt.shape} beta={beta}")

    storage_names = ("ADJ_LIST", "CSR", "ATOMIC_CSR") if args.storage == "ALL" else (args.storage,)
    for storage_name in storage_names:
        run_storage(spark_robin, src, tgt, beta, storage_name, sample, max(args.repeat, 1))


if __name__ == "__main__":
    main(parse_args())
