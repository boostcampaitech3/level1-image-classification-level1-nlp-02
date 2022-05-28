import torch
from models.model import Network
from utils.metrics import cal_metric
from utils.utils import get_device
import numpy as np
import random
from tqdm import tqdm
import os
import wandb

class Trainer(object):
    def __init__(self, config:dict, dataset, model_checkpoint_dir, fold):
        self.config = config
        self.dataset = dataset
        self.device = get_device()
        self.fold = fold
        self.model_checkpoint_dir = model_checkpoint_dir

    def train_and_validate_one_epoch(self, model, epoch, dataloaders, criterion_ce, criterion_bce, optimizer, best_score, output_structure):

        for phase in ['train', 'valid']:
            dataloader = dataloaders[phase]
            running_loss, correct = 0,0
            target_dict, pred_dict = {'target':[], 'age':[], 'gender':[], 'mask':[]}, {'target':[], 'age':[], 'gender':[], 'mask':[]}
            total_size = len(dataloader.dataset)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in dataloader:
                optimizer.zero_grad()
                image = batch['image'].to(self.device)
                gender, age, mask, target = [value.to(self.device) for value in batch['labels'].values()]

                with torch.set_grad_enabled(phase=='train'):
                    if output_structure == 'single':
                        output = model(image)
                        loss = criterion_ce(output, target) 

                        _, preds = torch.max(output, 1)
                        target_dict['target'].append(target)
                        pred_dict['target'].append(preds)

                    elif output_structure == 'multiple':
                        gender_out, age_out, mask_out  = model(image)
                        loss = criterion_bce(gender_out.squeeze(), gender.float()) + 1.3*criterion_ce(age_out, age) + criterion_ce(mask_out, mask) 

                        preds_gender = (gender >= 0.5).float()
                        _, preds_age = torch.max(age_out, 1)
                        _, preds_mask = torch.max(mask_out, 1)
                        target_dict['age'].append(age)
                        pred_dict['age'].append(preds_age)
                        target_dict['gender'].append(gender)
                        pred_dict['gender'].append(preds_gender)                        
                        target_dict['mask'].append(mask)
                        pred_dict['mask'].append(preds_mask)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * image.size(0)

            total_loss = running_loss / len(dataloader.dataset)
            for (key0, value0), (key1, value1) in zip(target_dict.items(), pred_dict.items()):
                if len(value0) == 0:
                    continue
                target_dict[key0] = torch.cat(value0)
                pred_dict[key1] = torch.cat(value1)
    
            if output_structure == 'single':
                f1score, total_acc = cal_metric(pred_dict['target'], target_dict['target'], total_size, self.config['model']['num_classes'])
                print(f'{self.fold}|{phase}-[EPOCH:{epoch}] |F1: {f1score:.3f} | ACC: {total_acc:.3f} | Loss: {total_loss:.5f}|')

            elif output_structure == 'multiple':
                f1score_age, total_acc_age = cal_metric(pred_dict['age'], target_dict['age'], total_size, 3)
                f1score_gender, total_acc_gender = cal_metric(pred_dict['gender'], target_dict['gender'], total_size, 2)
                f1score_mask, total_acc_mask = cal_metric(pred_dict['mask'], target_dict['mask'], total_size, 6)
                f1score = np.mean([f1score_age, f1score_gender, f1score_mask])
                total_acc = np.mean([total_acc_age, total_acc_gender, total_acc_mask])
                print(f'{self.fold}|{phase}-[EPOCH:{epoch}] |F1: {f1score:.3f} | f1score_age:{f1score_age:.3f} | f1score_gender:{f1score_gender:.3f} | f1score_mask:{f1score_mask:.3f} | ACC: {total_acc:.3f} | Loss: {total_loss:.5f}|')

            if phase == 'valid' and f1score > best_score:
                best_score = f1score
                print(f'{best_score:.3f} model saved')
                self._checkpoint(model, epoch, best_score)

            if not self.config['experiment']['debugging']:
                wandb.log({f"f1score_{phase}":f1score, f"loss_{phase}":total_loss }, step=epoch)

        return best_score
        
    def train(self):
        if not self.config['experiment']['debugging']:
            wandb.init(project='image_classification', reinit=True, config=self.config)
            wandb.run.name = self.model_checkpoint_dir.split('/')[-1] + '_' +self.config['model']['output_structure']
            wandb.run.save()

        set_randomseed(self.config['random_seed'])
        model = Network(self.config).to(self.device)

        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_bce = torch.nn.BCELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config['LR']))
        output_structure = self.config['model']['output_structure'] 
        
        train_dataloader, valid_dataloader = self.dataset.make_dataloader()
        dataloaders = {'train':train_dataloader, 'valid': valid_dataloader}
    
        print('*****'*40)
        print(self.fold, '-fold training start!')
        print('*****'*40)

        best_score = 0
        for epoch in tqdm(range(self.config['num_epochs'])):
            best_score = self.train_and_validate_one_epoch(model, epoch, dataloaders, 
                                                            criterion_ce, criterion_bce,
                                                            optimizer, best_score, output_structure)

    def _checkpoint(self, model, epoch, f1score):
        state = {
            'model' : model.state_dict(),
            'epoch' : epoch,
            'f1_score' : f1score
        }
        torch.save(state, os.path.join(self.model_checkpoint_dir, f"{self.config['model']['model_name']}_{self.fold}.pt"))

def set_randomseed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
        

            
            
