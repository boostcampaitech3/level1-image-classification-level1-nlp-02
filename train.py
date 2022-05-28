from train.trainer import Trainer
from utils.dataloader import TrainLoaderWrapper
from utils.processing import processing_df
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import os 
import yaml
import time
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

def _make_checkpoint_dir(model_checkpoint_dir):
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    shutil.copy('./config/config.yaml', model_checkpoint_dir)
    
def main():
    config = yaml.load(open("./config/" + "config.yaml", "r"), Loader=yaml.FullLoader)
    train_df = pd.read_csv(os.path.join(config['dir']['input_dir'], 'train/train.csv')) 
    
    if config['experiment']['debugging']:
        train_df = train_df.loc[:100]
        print('debugging mode')

    if config['experiment']['cross_validation']:
        fold = config['experiment']['fold']
    else:
        fold = 1

    now = time.localtime()
    date = f'{now.tm_mday}d-{now.tm_hour}h-{now.tm_min}m'
    model_checkpoint_dir = os.path.join(config['dir']['checkpoint_dir'], date)
    _make_checkpoint_dir(model_checkpoint_dir)

    sys.stdout = open(model_checkpoint_dir +'/training_log.txt', 'w')

    mskf = MultilabelStratifiedShuffleSplit(n_splits=fold, test_size=config['valid_size'], random_state=config['random_seed'])
    for fold, (train_idx, valid_idx) in enumerate(mskf.split(train_df, train_df[['gender','age']]), 1):
        train_df = processing_df(train_df.loc[train_idx].reset_index(drop=True), config)
        valid_df = processing_df(train_df.loc[valid_idx].reset_index(drop=True), config)
        feeder = TrainLoaderWrapper(config, train_df, valid_df)
        trainer = Trainer(config, feeder, model_checkpoint_dir, fold)
        trainer.train()

    sys.stdout.close()

if __name__ == '__main__':
    main()