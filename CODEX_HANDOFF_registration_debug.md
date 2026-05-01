# Codex Handoff: POLI Object Pose Registration Divergence

## Goal

We are debugging a cross-machine divergence where:

- the same codebase is believed to be used,
- neural normal estimation appears to match,
- but the final post-normal registration result diverges on this machine vs another machine.

This note is written so another Codex instance on the working machine can pick up the investigation with minimal context loss.

## Short Symptom Summary

The user reports:

- normal estimation is the same,
- but the later "optimization" or final registration result differs,
- another computer produces much better results with apparently the same code.

Important clarification:

- In the current inference/demo pipeline, the post-normal stage is not Adam/LBFGS style gradient optimization.
- In `applications/object_pose_estimation/demo.py`, the pipeline after normal estimation is:
  1. neural covariance -> normals
  2. Open3D FPFH feature extraction
  3. nearest-neighbor feature matching via `torch.cdist(...); torch.argmin(...)`
  4. ROBIN maximum clique consensus
  5. closed-form SVD rigid registration
- The only obvious gradient optimizer in this repo is the training-time `Adam` in `train.py`, not the demo inference path.

This means the divergence is most likely in one of:

- Open3D FPFH behavior/version,
- match tie-breaking from small floating differences,
- ROBIN consensus result,
- SVD input correspondences,
- or "same code / same weights / same env" actually being false.

## Relevant Code Path

### Demo / inference path

File: `applications/object_pose_estimation/demo.py`

- seeding: lines 311-313
- model loading: lines 336-338
- neural normal generation: lines 340-342
- FPFH extraction: lines 343-344
- feature matching: line 345
- consensus: lines 356-360
- final rigid solve: lines 365-367
- error print: lines 369-374

### Registration utilities

File: `applications/object_pose_estimation/registration_utils.py`

- covariance assembly: lines 7-20
- normals from covariance via `torch.linalg.eigh`: lines 23-25
- Open3D FPFH: lines 28-49
- correspondence estimation via `torch.cdist` + `argmin`: lines 52-59
- ROBIN consensus: lines 89-108
- SVD rigid registration: lines 62-86

### Training path

File: `train.py`

- `Adam` optimizer: line 43
- SDPR pose block: lines 45-47 and 117-123

This is likely not the path the user means when they say normals are same but "optimization result" differs, unless they are debugging training rather than `demo.py`.

## Current Repo State On This Machine

Repo root: `/home/jinwoo/POLI`

Git HEAD:

- `05695cac0e620a0f8bc63c1b3e67e4d80f133b5e`

Working tree is dirty.

Relevant modified files:

- `applications/object_pose_estimation/demo.py`
- `applications/object_pose_estimation/registration_utils.py`
- `utils.py`

Diff summary:

- `demo.py`: import path changes to use repo-root-qualified imports and `sys.path` insertion
- `registration_utils.py`: temporary debug prints added inside `maximum_consensus`
- `utils.py`: visualization-related color mapping changes

This already weakens the assumption that both computers are using literally identical code.

## File Integrity Snapshot

Use these hashes to confirm that both machines are really using the same artifacts.

### Code hashes

- `train.py`
  - `f6f8b1c028d1b3954dc63b7937660324622b8ad600b197b0e1ba940a64ed0201`
- `utils.py`
  - `f7bf32013d94b77665539998c2b2cdf09d25e4795afcf89a16633540b038746e`
- `applications/object_pose_estimation/demo.py`
  - `87fae3e273ac4882c28c8899c4cc487ac696688d40a905104f3d823cc0cc7795`
- `applications/object_pose_estimation/registration_utils.py`
  - `e7b42be578c3154cc17da10260c93e3c3c84e494adeb6c33cbd9f810ce444215`
- `applications/object_pose_estimation/model/pointnetpp_core.py`
  - `fc5df4b9dc3ca475d0d419cad339f580972f0d17427156f8c44d9a498a47c6b2`
- `applications/object_pose_estimation/model/pointnetpp_utils.py`
  - `0181fd242e406742dca991052f88cdb4b50580f7f80c7e81adaaf3648b762e1b`

### Data / weight hashes

- `applications/object_pose_estimation/bunny.npz`
  - `31631c5a6bf9f1967512c9042d4473cfa835be2df6362a077279f95a9503c6c4`
- `applications/object_pose_estimation/dragon.npz`
  - `f66593ecdcf0f188703f6e6d42170cc25b4303155d6a6c5097ead564a159c012`
- `weights/object/object_500n1000points.pth`
  - `ef7e357afb90d012224840a1053b880403b1295f50f2f7365c7cc63f75f8b764`

## Sample / Checkpoint Metadata

### Sample files

`bunny.npz`

- `P`: shape `(3, 800)`, dtype `float32`
- `Q`: shape `(3, 800)`, dtype `float32`
- `R_rel`: shape `(3, 3)`, dtype `float32`
- `t_rel`: shape `(3, 1)`, dtype `float32`
- `indices`: shape `(800,)`, dtype `int64`
- `dists`: shape `(800,)`, dtype `float32`

`dragon.npz`

- `P`: shape `(3, 800)`, dtype `float32`
- `Q`: shape `(3, 800)`, dtype `float32`
- `R_rel`: shape `(3, 3)`, dtype `float32`
- `t_rel`: shape `(3, 1)`, dtype `float32`
- `indices`: shape `(800,)`, dtype `int64`
- `dists`: shape `(800,)`, dtype `float32`

### Object checkpoint

`weights/object/object_500n1000points.pth`

- top-level keys:
  - `epoch`
  - `model_state_dict`
  - `optimizer_state_dict`
  - `step`
  - `valid_step`
- model state dict entries: `158`

## Environment Survey On This Machine

One major finding: the environment is fragmented. Different required packages exist in different conda envs, and some are editable installs pointing to local source trees outside this repo.

That means "same conda env" may already be false even before version differences.

### Shell default env observed by Codex

Executable:

- `/home/jinwoo/anaconda3/bin/python`

Observed imports:

- Python `3.8.19`
- torch `2.4.1+cu121`
- numpy import reported `1.24.4`
- scipy `1.10.0`
- open3d `0.19.0`
- `spark_robin`: missing
- `pointnet2_ops`: missing
- `sdprlayers`: missing
- `torch.cuda.is_available()`: `False` inside sandbox

Important mismatch:

- `pip show numpy` reported `1.23.2`, while Python import reported `1.24.4`
- this is another sign that environment state is not clean/simple

### `diffphysdrone` env

Executable:

- `/home/jinwoo/anaconda3/envs/diffphysdrone/bin/python`

Observed imports:

- Python `3.11.15`
- torch `2.2.2+cu121`
- numpy `1.26.4`
- scipy missing
- open3d missing
- `pointnet2_ops` missing
- `sdprlayers` missing
- `spark_robin` missing
- `torch.cuda.is_available()`: `True` inside this probe, but core deps are absent

This does not look like the full working env for this pipeline.

### `radar` env

Executable:

- `/home/jinwoo/anaconda3/envs/radar/bin/python`

Observed imports:

- Python `3.10.19`
- torch `2.3.0+cu121`
- numpy `2.1.2`
- scipy `1.9.0`
- open3d import failed due NumPy ABI issue
- `pointnet2_ops` import failed with undefined symbol:
  - `_ZNK3c105Error4whatEv`
- `sdprlayers` import failed because SciPy/Open3D stack is broken under NumPy 2.x
- `spark_robin` import succeeded

This env is clearly inconsistent / ABI-broken:

- SciPy 1.9.0 is not compatible with NumPy 2.1.2
- Open3D also fails there
- `pointnet2_ops` appears compiled against a different PyTorch ABI

If this machine is using `radar` for runs, environment issues are very plausible.

### `radar2` env

Executable:

- `/home/jinwoo/anaconda3/envs/radar2/bin/python`

Observed imports:

- Python `3.10.18`
- torch `2.9.0+cu128`
- numpy `1.24.4`
- scipy `1.9.0`
- open3d `0.18.0`
- `pointnet2_ops` `3.3.0`
- `sdprlayers` import succeeded
- `spark_robin` import succeeded

But there is an important caveat:

- `pointnet2_ops` is not from a normal conda package path
- it resolves to local editable source:
  - `/home/jinwoo/Downloads/pointnet2_ops/pointnet2_ops/__init__.py`
- `sdprlayers` is also from local editable source:
  - `/home/jinwoo/2026_RSS/DGICP/sdprlayer/src/sdprlayers/__init__.py`
- `spark_robin` is from env site-packages:
  - `/home/jinwoo/anaconda3/envs/radar2/lib/python3.10/site-packages/spark_robin.cpython-310-x86_64-linux-gnu.so`

This env is the closest thing found to a plausible fully assembled runtime for the pipeline.

### `pointnet_dgicp` env

Executable:

- `/home/jinwoo/anaconda3/envs/pointnet_dgicp/bin/python`

Observed imports:

- Python `3.9.21`
- torch `2.5.1`
- numpy `2.0.1`
- scipy `1.13.1`
- open3d `0.19.0`
- `pointnet2_ops` `3.3.0`
- `sdprlayers`: missing
- `spark_robin`: missing

This env has `pointnet2_ops`, but not the full registration stack.

## External Editable / Local Dependency State

This is very important because another machine may be "using the same repo" but not the same external editable sources.

### pointnet2_ops local source

Path:

- `/home/jinwoo/Downloads/pointnet2_ops`

Git HEAD:

- `8afb2b39d4cbbd9d4414edc04ed2529e211f799e`

Git status:

- untracked `pointnet2_ops.egg-info/`
- untracked `pointnet2_ops/__pycache__/`
- untracked `pointnet2_ops/_ext.cpython-310-x86_64-linux-gnu.so`

### sdprlayers local source

Path:

- `/home/jinwoo/2026_RSS/DGICP/sdprlayer`

Git HEAD:

- `cb3aed2d70766d2cb612a92a6a014346b53f34a6`

Git status is dirty:

- modified `.gitignore`
- modified `src/sdprlayers/layers/pose_est.py`
- modified `src/sdprlayers/layers/sdprlayer.py`
- deleted `src/submodules/cert_tools`
- modified `src/submodules/poly_matrix`
- untracked `mosek_output.tmp`
- untracked `src/submodules/certifiable-tools/`

### ROBIN source inside this repo

Path:

- `/home/jinwoo/POLI/ROBIN`

Git HEAD:

- `52bab5f2bb6dd68a5febb6bd06bb2ffdff2f75c2`

Git status is dirty:

- modified `CMakeLists.txt`
- modified `cmake/DownloadProject.CMakeLists.cmake.in`
- modified `matlab/CMakeLists.txt`
- modified `python/CMakeLists.txt`

## Reproduction Attempt Inside Codex Sandbox

I tried a non-visual probe under `radar2` using the current repo code and object checkpoint.

Result:

- model forward failed at `pointnet2_ops` furthest point sampling on CPU
- error:
  - `RuntimeError: points must be a CUDA tensor`

This means:

- this pipeline likely requires a CUDA-visible runtime in practice
- Codex sandbox results about `torch.cuda.is_available()` should not be over-interpreted
- full reproduction on this machine needs the actual GPU-enabled user runtime, not just the sandbox

## Highest-Probability Divergence Points

Ranked from most suspicious to less suspicious.

### 1. The two computers are not actually using the same runtime env

This is the strongest finding so far.

Reasons:

- core deps are spread across multiple envs
- some key packages are editable installs from local source trees
- ABI-sensitive compiled extensions are involved
- several envs are obviously broken or incomplete

### 2. Open3D version difference changes FPFH descriptors

Current observed versions on this machine vary across envs:

- Open3D `0.18.0`
- Open3D `0.19.0`

If normals truly match but later outputs diverge, FPFH generation is a prime suspect.

### 3. `torch.cdist` + `argmin` match selection is sensitive to tiny feature differences

Correspondence estimation is:

- full pairwise distance matrix
- hard nearest-neighbor selection

If multiple targets have near-identical distances, tiny floating differences can flip match assignments, and that will amplify downstream in ROBIN and SVD.

### 4. ROBIN maximum clique may return a different valid inlier set

If the compatibility graph differs even slightly, or if multiple equal-size cliques exist, the selected consensus set can change.

The current local `registration_utils.py` was already modified to print:

- inlier count
- first 50 sorted indices
- last 50 sorted indices

That is a good stage to compare directly across machines.

### 5. Compiled extension / ABI mismatch in `pointnet2_ops`

This is especially suspicious if one machine has a custom-built extension against a different torch/CUDA combo.

We already observed one env where `pointnet2_ops` fails with a missing C++ symbol.

### 6. Local repo modifications or different weights/sample files

This is boring but still essential to rule out.

## What The Other Machine's Codex Should Report Back

Please ask the working machine to send back all of the following.

### A. Exact runtime identity

- `which python`
- `python -V`
- `python -c "import sys; print(sys.executable)"`
- active conda env name
- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- GPU model

### B. Exact package origins

For each package below, report both `__version__` and `__file__`:

- `torch`
- `numpy`
- `scipy`
- `open3d`
- `pointnet2_ops`
- `sdprlayers`
- `spark_robin`

This is critical because editable installs can hide local source dependencies.

### C. Exact code/data identity

- `git rev-parse HEAD` in `POLI`
- `git status --short`
- hashes of the same code/data files listed above

### D. Stage-by-stage outputs

The working machine should compare:

- normals hash
- FPFH hash
- match vector hash
- number of unique matched targets
- ROBIN inlier count
- ROBIN inlier index hash
- first/last sorted inlier indices
- final estimated rotation
- final estimated translation
- rotation/translation error vs GT

## Recommended Probe Script For The Working Machine

The other Codex can run something like this from repo root in the actual working env.

```python
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/jinwoo/POLI")
sys.path.insert(0, str(REPO))

from applications.object_pose_estimation.demo import load_npz_sample, make_covariance_normals
from applications.object_pose_estimation.model import PointPP
from applications.object_pose_estimation.registration_utils import (
    compute_fpfh_batch,
    estimate_correspondences,
    load_checkpoint,
    rotation_translation_error,
    svd_registration,
)
import spark_robin


def arr_sha(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.view(np.uint8)).hexdigest()


def t_sha(x):
    return arr_sha(x.detach().cpu().numpy())


np.random.seed(7777)
torch.manual_seed(7777)
torch.cuda.manual_seed_all(7777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("This setup expects CUDA because pointnet2_ops FPS requires CUDA.")

sample = load_npz_sample(REPO / "applications/object_pose_estimation/bunny.npz")
P = torch.from_numpy(sample["P"]).unsqueeze(0).to(device)
Q = torch.from_numpy(sample["Q"]).unsqueeze(0).to(device)

model = PointPP().to(device)
load_checkpoint(model, str(REPO / "weights/object/object_500n1000points.pth"), device)
model.eval()

with torch.no_grad():
    cov_p, normals_p = make_covariance_normals(model, P)
    cov_q, normals_q = make_covariance_normals(model, Q)
    fpfh_p = compute_fpfh_batch(P, normals_p, radius=0.2, max_nn=90)
    fpfh_q = compute_fpfh_batch(Q, normals_q, radius=0.2, max_nn=90)
    matches = estimate_correspondences(fpfh_p[0], fpfh_q[0]).detach().cpu().numpy()

src = sample["P"].T.astype(np.float64)
tgt = sample["Q"].T[matches].astype(np.float64)

graph = spark_robin.Make3dRegInvGraph(
    src.T,
    tgt.T,
    float(0.1),
    spark_robin.GraphStorageType.ADJ_LIST,
)
indices = np.asarray(
    spark_robin.FindInlierStructure(
        graph,
        spark_robin.InlierGraphStructure.MAX_CLIQUE,
    )
)

src_inliers = src[indices]
tgt_inliers = tgt[indices]
T_est = svd_registration(src_inliers, tgt_inliers)
R_est = T_est[:3, :3]
t_est = T_est[:3, 3]
rot_error, trans_error = rotation_translation_error(
    R_est, t_est, sample["R_rel"], sample["t_rel"]
)

print("device", device)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("normals_p_sha", t_sha(normals_p))
print("normals_q_sha", t_sha(normals_q))
print("fpfh_p_sha", t_sha(fpfh_p))
print("fpfh_q_sha", t_sha(fpfh_q))
print("matches_sha", arr_sha(matches))
print("matches_unique", len(np.unique(matches)))
print("indices_count", len(indices))
print("indices_sha", arr_sha(indices.astype(np.int64)))
print("indices_first50", np.sort(indices)[:50].tolist())
print("indices_last50", np.sort(indices)[-50:].tolist())
print("R_est", np.array2string(R_est, precision=6, suppress_small=True))
print("t_est", np.array2string(t_est, precision=6, suppress_small=True))
print("rot_error_deg", rot_error)
print("trans_error", trans_error)
```

If the normals hash already differs, the problem is earlier than expected.

If the normals hash matches but the FPFH hash differs, focus on Open3D and dtype/path issues.

If FPFH matches but `matches_sha` differs, focus on `torch.cdist` / `argmin` / dtype / feature layout.

If matches match but `indices_sha` differs, focus on ROBIN.

If indices match but final pose differs, focus on SVD input ordering / numeric handling.

## Concrete Questions For The Other Codex

1. Which exact conda env successfully runs the pose demo on the working machine?
2. Are `pointnet2_ops`, `sdprlayers`, and `spark_robin` regular installs or editable installs?
3. What are the exact `__file__` paths for those modules?
4. Do the code/data hashes match this machine?
5. At which exact stage do the hashes first diverge:
   - normals
   - FPFH
   - matches
   - ROBIN inliers
   - final pose
6. Is the working machine on Open3D `0.18.x`, `0.19.x`, or something else?
7. What are the torch + CUDA versions there, and was `pointnet2_ops` compiled against that exact torch version?

## Current Best Hypothesis

Best current hypothesis:

- the problem is primarily environment drift, not pure repo logic drift
- especially around:
  - custom compiled `pointnet2_ops`
  - editable `sdprlayers`
  - `spark_robin`
  - Open3D version
  - and possibly GPU-visible runtime differences

The strongest candidate env on this machine is `radar2`, but even that env depends on local editable source outside the repo, so "same repo" is not enough to guarantee the same runtime behavior.

