from torch.utils.data import Dataset
import torch
import os
import numpy as np

class PCD_Dataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.sample_files = sorted(
            [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.npz')]
        )
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        sample = np.load(self.sample_files[idx])
        sample_data = {
            'P': torch.from_numpy(sample['P'].astype(np.float32, copy=False)),
            'Q': torch.from_numpy(sample['Q'].astype(np.float32, copy=False)),
            'R_rel': torch.from_numpy(sample['R_rel'].astype(np.float32, copy=False)),
            't_rel': torch.from_numpy(sample['t_rel'].astype(np.float32, copy=False)),
            'dists': torch.from_numpy(sample['dists'].astype(np.float32, copy=False)),
            'indices': torch.from_numpy(sample['indices'].astype(np.int64, copy=False)),
        }

        return sample_data
