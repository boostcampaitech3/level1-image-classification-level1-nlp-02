from utils.dataloader import TestLoaderWrapper
from utils.utils import get_device
from models.model import Network
import torch
import yaml
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_class(gender, age, mask):
    a = (gender >= 0.5).float().squeeze()
    b = age.argmax(dim=1)
    c = mask.argmax(dim=1)
    d = torch.stack((a,b,c))
    return [(i[0]*3 + i[1] + i[2]*6).item() for i in d.T]

def main():
    checkpoint_dir = './checkpoint/2d-13h-56m'
    config = yaml.load(open(checkpoint_dir + "/config.yaml", "r"), Loader=yaml.FullLoader)
    info = pd.read_csv(os.path.join(config['dir']['input_dir'], 'eval/info.csv')) 
    feeder = TestLoaderWrapper(config, info)
    dataloader = feeder.make_dataloader()
    device = get_device()
    model = Network(config)

    if config['experiment']['cross_validation']:
        num_fold = config['experiment']['fold']
    else:
        num_fold = 1

    total_preds, total_gender, total_age, total_mask = [], [], [], []

    for fold in tqdm(range(1, num_fold+1)):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, config['model']['model_name'] + f'_{fold}.pt'))['model'])
        model = model.to(device)
        model.eval()
        preds, preds_gender, preds_age, preds_mask = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

        if config['model']['output_structure'] == 'single':
            for image in dataloader:
                image = image.to(device)
                pred = model(image) 
                pred = pred.cpu().detach()
                preds = torch.cat((preds, pred))
            total_preds.append(preds)

        elif config['model']['output_structure'] == 'multiple':
            for image in dataloader:
                image = image.to(device)
                pred = model(image) 
                gender, age, mask = model(image)
                preds_gender = torch.cat((preds_gender, gender.cpu().detach()))
                preds_age = torch.cat((preds_age, age.cpu().detach()))
                preds_mask = torch.cat((preds_mask, mask.cpu().detach()))

            total_gender.append(preds_gender)
            total_age.append(preds_age)
            total_mask.append(preds_mask)
            
    if config['model']['output_structure'] == 'single':
        preds = sum(total_preds)/num_fold
        preds = preds.argmax(dim=-1)

    elif config['model']['output_structure'] == 'multiple':
        print(sum(total_age)/num_fold)
        preds = get_class(sum(total_gender)/num_fold, sum(total_age)/num_fold, sum(total_mask)/num_fold)

    info['ans'] = preds
    info.to_csv('submission.csv', index=False)
    print('inference is done!')

if __name__ == '__main__':
    main()
