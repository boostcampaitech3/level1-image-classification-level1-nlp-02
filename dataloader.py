import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import random

# Need to edit when dataset changed
from dataset import GenderDataset, MaskDataset
from augmentation import basic_transforms


def get_gender_loaders(config):

    # set random seed
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    g = torch.Generator()
    g.manual_seed(config.random_seed)


    dataframe: pd.DataFrame = pd.read_csv('../input/data/pre_processed_train.csv')

    train_cnt: int = int(dataframe.shape[0] * config.train_valid_ratio)
    valid_cnt: int = dataframe.shape[0] - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices: torch.tensor = torch.randperm(dataframe.shape[0])
    train_idx, valid_idx = torch.index_select(
        torch.tensor([i for i in range(dataframe.shape[0])]),
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader= DataLoader(
        # Need to edit when dataset changed
        dataset=GenderDataset(dataframe, train_idx, basic_transforms),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        generator=g,
    )

    valid_loader = DataLoader(
        # Need to edit when dataset changed
        dataset=GenderDataset(dataframe, valid_idx, basic_transforms),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
        generator=g,
    )

    return train_loader, valid_loader


def get_mask_loaders(config):

    # set random seed
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    g = torch.Generator()
    g.manual_seed(config.random_seed)


    dataframe: pd.DataFrame = pd.read_csv('../input/data/pre_processed_train.csv')

    train_cnt: int = int(dataframe.shape[0] * config.train_valid_ratio)
    valid_cnt: int = dataframe.shape[0] - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices: torch.tensor = torch.randperm(dataframe.shape[0])
    train_idx, valid_idx = torch.index_select(
        torch.tensor([i for i in range(dataframe.shape[0])]),
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader= DataLoader(
        # Need to edit when dataset changed
        dataset=MaskDataset(dataframe, train_idx, basic_transforms),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        generator=g,
    )

    valid_loader = DataLoader(
        # Need to edit when dataset changed
        dataset=MaskDataset(dataframe, valid_idx, basic_transforms),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        generator=g,
    )

    return train_loader, valid_loader