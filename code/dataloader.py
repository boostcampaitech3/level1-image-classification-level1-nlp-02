import torch
import pandas as pd
import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import augmentation

CSV_PATH = '../input/data/train/train_meta.csv'
#IMAGE_PATH = '../input/data/train/images/'

class MaskDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, transform=None):
        self.dataset = dataset
        self.labels = torch.tensor(dataset['label'].values)
        self.transform = transform

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        x = Image.open('../'+row['path'])
        
        if self.transform:
            x = self.transform(x)

        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)


def get_loader(config):

    # set random seed
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    g = torch.Generator()
    g.manual_seed(config.random_seed)

    dataset : pd.DataFrame = pd.read_csv(CSV_PATH)
    
    train_cnt : int = int(len(dataset) * config.train_ratio)
    valid_cnt : int = len(dataset) - train_cnt
    
    indices = torch.randperm(len(dataset))

    # random shuffle
    dataset  = dataset.iloc[indices]

    # split
    train_dataset = dataset.iloc[:train_cnt]
    valid_dataset = dataset.iloc[train_cnt:]
    
    train_loader = DataLoader(
        dataset = MaskDataset(train_dataset, transform=augmentation.basic_transforms),
        batch_size = config.batch_size,
        shuffle=True,
    )
    
    valid_loader = DataLoader(
        dataset = MaskDataset(valid_dataset, transform=augmentation.basic_transforms),
        batch_size = config.batch_size,
        shuffle=False,
    )
    
    return train_loader, valid_loader