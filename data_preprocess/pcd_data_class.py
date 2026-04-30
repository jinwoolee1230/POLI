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
            'P': torch.tensor(sample['P'], dtype=torch.float32),
            'Q': torch.tensor(sample['Q'], dtype=torch.float32),
            'R_rel': torch.tensor(sample['R_rel'], dtype=torch.float32),
            't_rel': torch.tensor(sample['t_rel'], dtype=torch.float32),
            'dists': torch.tensor(sample['dists'], dtype=torch.float32),
            'indices': torch.tensor(sample['indices'], dtype=torch.int64),
        }

        return sample_data
