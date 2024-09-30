# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:33:06 2024

@author: Dell
"""
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, mel_specs, labels):
        self.mel_specs = torch.from_numpy(mel_specs).float()      
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

        # Check the original size of the dataset
        self.num_samples = self.mel_specs.size(0)
        
    def __len__(self):
        return len(self.mel_specs)
    
    def __getitem__(self, idx):
        # Use clone().detach() to avoid the warning
        return self.mel_specs[idx].clone().detach(), self.labels[idx].clone().detach()
