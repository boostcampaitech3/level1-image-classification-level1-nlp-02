import wandb
import warnings
warnings.filterwarnings(action='ignore')

import torch

import numpy as np
import pandas as pd

from model import GenderClassifier, MaskClassifier, AgeClassifier
from dataloader import get_eval_loader



PROJECT = 'mask-classification'
ENTITY = 'naramalsami'
VERSION = 0
GENDER_MODEL_NAME = 'gender_classifier_basic_cnn_bceloss_epoch14_0.00215.pt'
MASK_MODEL_NAME = 'mask_classifier_basic_cnn_cross-entropy-loss_epoch10_0.00438.pt'
AGE_MODEL_NAME = 'age_classifier_basic_cnn_cross-entropy-loss_epoch12_0.01709.pt'
BATCH_SIZE = 64
CLASSES = 2

# Download from wandb
# run = wandb.init()
# artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/model:v{VERSION}', type='model')
# artifact_dir = artifact.download()
# run.join()

gender_data_dict = torch.load(f'checkpoint/{GENDER_MODEL_NAME}')
mask_data_dict = torch.load(f'checkpoint/{MASK_MODEL_NAME}')
age_data_dict = torch.load(f'checkpoint/{AGE_MODEL_NAME}')

eval_loader = get_eval_loader(batch_size=BATCH_SIZE)
device = torch.device(0)

gender_predictions = []
mask_predictions = []
age_predictions = []

gender_model = GenderClassifier().to(device)
mask_model = MaskClassifier().to(device)
age_model = AgeClassifier().to(device)

gender_model.load_state_dict(gender_data_dict['model_state_dict'])
mask_model.load_state_dict(mask_data_dict['model_state_dict'])
age_model.load_state_dict(age_data_dict['model_state_dict'])


gender_model.eval() # not to learn
with torch.no_grad():
    for _, (X) in enumerate(eval_loader):
        X = X.to(device).float()

        y_hat = gender_model(X)
        y_hat = torch.squeeze(y_hat)
        y_hat = y_hat.cpu().numpy()
        y_hat = np.where(y_hat > 0.5, 1, 0)

        gender_predictions.extend(y_hat)


mask_model.eval() # not to learn
with torch.no_grad():
    for _, (X) in enumerate(eval_loader):
        X = X.to(device).float()

        y_hat = mask_model(X)
        y_hat = torch.squeeze(y_hat)
        y_hat = y_hat.cpu().numpy().argmax(axis=1)

        mask_predictions.extend(y_hat)


age_model.eval() # not to learn
with torch.no_grad():
    for _, (X) in enumerate(eval_loader):
        X = X.to(device).float()

        y_hat = age_model(X)
        y_hat = torch.squeeze(y_hat)
        y_hat = y_hat.cpu().numpy().argmax(axis=1)

        age_predictions.extend(y_hat)


tmp_submission = pd.read_csv('../input/data/eval/info.csv')
tmp_submission['gender'] = gender_predictions
tmp_submission['mask'] = mask_predictions
tmp_submission['age'] = age_predictions

tmp_submission.to_csv('../input/data/eval/tmp_submission.csv', index=False)