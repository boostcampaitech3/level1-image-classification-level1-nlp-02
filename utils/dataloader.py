import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from utils.processing import *
import PIL.Image as Image
import os
import numpy as np
import pandas as pd
import random

class MaskedFaceDataset(Dataset):
    def __init__(self, csv_file:pd.DataFrame, folder:str, transforms=None):
        self.csv_file = csv_file
        self.transforms = transforms
        self.folder = folder

    def __len__(self):
        return len(self.csv_file)
        
    def __getitem__(self, idx:int) -> dict:
        sample = self.csv_file.loc[idx]
        gender = sample['gender_label']
        age = sample['age_label']
        mask = sample['mask_label']
        label = sample['label']
        image = Image.open(os.path.join(self.folder, sample['detail_path']))

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))['image']

        
        sample = {'image':image, 'labels':{'gender':gender,
                                            'age':age, 'mask':mask,
                                            'label':label}}

        return sample

class MaskedFaceDataset_Test(Dataset):
    def __init__(self, csv_file:pd.DataFrame, folder:str, transforms=None):
        self.csv_file = csv_file
        self.paths = self.csv_file['ImageID'].values
        self.folder = folder
        self.transforms = transforms
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx:int) -> Image:
        path = os.path.join(self.folder, self.paths[idx])
        image = Image.open(path)

        if self.transforms is not None:
            image = self.transforms(image=np.array(image))['image']

        return image

class TrainLoaderWrapper(object):
    def __init__(self, config:dict, train_df, valid_df):
        self.config = config
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = self.config['batch_size']
        self.valid_size = self.config['valid_size']
        self.num_workers = self.config['num_workers']
        
    def _make_dataset(self) -> torch.utils.data.Dataset:
        '''
        return dataset
        '''
        train_T, valid_T = get_augmentation('train')

        train_dataset = MaskedFaceDataset(self.train_df, self.config['dir']['image_dir'].format('train'),
                                             transforms=train_T)
        valid_dataset = MaskedFaceDataset(self.valid_df, self.config['dir']['image_dir'].format('train'),
                                             transforms=valid_T)
        
        return train_dataset, valid_dataset

    def make_dataloader(self) -> torch.utils.data.DataLoader:
        '''
        return dataloader
        '''
        train_dataset, valid_dataset = self._make_dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                        num_workers=self.num_workers, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                        num_workers=self.num_workers, shuffle=False)

        return train_dataloader, valid_dataloader
                
class TestLoaderWrapper():
    def __init__(self, config:dict, info:pd.DataFrame):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']
        self.info = info
    
    def _make_dataset(self) -> torch.utils.data.Dataset:
        test_T = get_augmentation('test')
        dataset = MaskedFaceDataset_Test(self.info, self.config['dir']['image_dir'].format('eval'),
                                                transforms=test_T)

        return dataset

    def make_dataloader(self) -> torch.utils.data.DataLoader:
        test_dataset = self._make_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                                    num_workers=self.num_workers, shuffle=False)

        return test_dataloader
        
def get_augmentation(mode) -> torchvision.transforms:
    '''
    return torchvision.transforms
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = A.Compose([
        #A.Crop(y_min=24, y_max=460, x_max=384, p=1.0), #[24:460,:,:] / H,W,C
        A.ToGray(p=1),
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Rotate(limit=(-20,20), p=1),
            A.PiecewiseAffine(p=0.8, scale=(0.008, 0.01))
        ], p=1),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),  
            A.MotionBlur(p=1, blur_limit=5)
        ], p=1),
        A.OneOf([
            A.GridDropout(p=1,ratio=0.3),
            A.Cutout(p=1, num_holes=20), 
        ], p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        #A.Crop(y_min=24, y_max=460, x_max=384, p=1.0),
        A.ToGray(p=1),
        A.Resize(256,256),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    if mode == 'train':
        return train_transforms, test_transforms 

    elif mode == 'test':
        return test_transforms
        







