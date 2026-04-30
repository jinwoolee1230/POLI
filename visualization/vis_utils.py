import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


from data_preprocess.pcd_data_class import PCD_Dataset
from utils import scale_aware_normalization


def load_parsed_dataset(dataset_dir):
    dataset = PCD_Dataset(dataset_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No parsed '.npz' files found in dataset_dir: {dataset_dir}")
    return dataset


def load_model(checkpoint, device):
    from model.pointnetpp_core import PointPP

    model = PointPP().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def validate_sample_idx(dataset, sample_idx):
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx {sample_idx} is out of range for dataset with {len(dataset)} samples"
        )


def get_sample(dataset, sample_idx, device):
    validate_sample_idx(dataset, sample_idx)

    sample = dataset[sample_idx]
    batched_sample = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batched_sample[key] = value.unsqueeze(0).to(device)
        else:
            batched_sample[key] = value
    return batched_sample


def get_sample_path(dataset, sample_idx):
    validate_sample_idx(dataset, sample_idx)
    return dataset.sample_files[sample_idx]


def normalize_inputs(P, Q, mean, scaling_factor):
    mean_tensor = torch.tensor(mean, dtype=P.dtype, device=P.device).view(1, 3, 1)
    return scale_aware_normalization(P, Q, mean_tensor, scaling_factor)
