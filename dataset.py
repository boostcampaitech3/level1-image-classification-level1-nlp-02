import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from PIL import Image



class GenderDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, index: torch.tensor, transform=None):
        self.dataframe = dataframe
        self.index = index
        self.transform = transform
    
        super().__init__()

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        x = Image.open(f'../input/data/train/images/{self.dataframe.detail_path[idx]}')

        if self.transform:
            x = self.transform(x)

        y = self.dataframe.gender[idx]

        return x, y


class MaskDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, index: torch.tensor, transform=None):
        self.dataframe = dataframe
        self.index = index
        self.transform = transform
    
        super().__init__()

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        x = Image.open(f'../input/data/train/images/{self.dataframe.detail_path[idx]}')

        if self.transform:
            x = self.transform(x)

        y = self.dataframe.mask_type[idx]

        return x, y