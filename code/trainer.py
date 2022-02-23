import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import random

import numpy as np
from tqdm import tqdm
import wandb


from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
    
    def train(self, config, train_loader, valid_loader):
        wandb.init(project=config.wandb_project, entity='sujeongim')
        wandb.config = {
            'dropout':0.5,
            'epochs': config.n_epochs,
            'batch_size': config.batch_size,
        }
        
        # set random seed
        torch.manual_seed(config.random_seed)       # About PyTorch
        torch.backends.cudnn.deterministic = True   # About CuDNN -> Maybe cause slow training ?
        torch.backends.cudnn.benchmark = False      # About CuDNN
        np.random.seed(config.random_seed)          # About Numpy
        random.seed(config.random_seed)             # About transform -> Also should be in augmentation.py ?
        torch.cuda.manual_seed(config.random_seed)  # About GPU

        lowest_loss = np.inf
        best_model = None

        for epoch in range(config.n_epochs):
            train_loss_total = 0

            # train
            self.model.train()
            pbar = tqdm(train_loader)
            for X, y in pbar:
                X = X.to(config.device)
                y = y.to(config.device)

                y_pred = self.model.forward(X)
                loss_out = self.crit(y_pred, y)

                self.optimizer.zero_grad()
                loss_out.backward()
                self.optimizer.step()

                train_loss_total += float(loss_out)
                pbar.set_postfix({'epoch': f'{epoch}/{config.n_epochs}', 'loss': f'{train_loss_total/config.batch_size:.5f}'})

            train_loss_avg = train_loss_total / len(train_loader)

            # validation
            self.model.eval()
            pbar = tqdm(valid_loader)
            with torch.no_grad():
                valid_loss = 0.
                valid_precision = 0.
                valid_recall = 0.
                valid_f1_score = 0.
                
                for X, y in pbar:
                    X = X.to(config.device)
                    y = y.to(config.device)

                    model_pred = self.model.forward(X)
                    loss_out = self.crit(model_pred, y)
                    _, y_pred = torch.max(model_pred.data, 1)

                    y, y_pred = y.cpu().detach().numpy(), y_pred.cpu().detach().numpy()

                    precision = precision_score(y, y_pred, average='macro', zero_division=1)
                    recall = recall_score(y, y_pred, average='macro', zero_division=1)
                    f1 = f1_score(y, y_pred, average='macro', zero_division=1) # 데이터 불균형 따로 고려하지 않음

                    valid_loss += loss_out
                    valid_precision += precision
                    valid_recall += recall
                    valid_f1_score += f1
                
                valid_loss /= len(valid_loader)
                valid_precision /= len(valid_loader)
                valid_recall /= len(valid_loader)
                valid_f1_score /= len(valid_loader)


            # update model
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss

                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'epoch': epoch_index,
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': valid_loss},
                #     f'checkpoint/epoch{epoch_index}_{valid_loss:.5f}.pt')
                
            print(f'Epoch-{epoch} Train Loss: {train_loss_avg:.5f}  Valid Loss: {valid_loss:.5f}  Best Valid Loss: {lowest_loss:.5f}  Precision: {valid_precision:.5f}  Recall: {valid_recall:.5f}  f1_score: {valid_f1_score:.5f}')

            wandb.log({
                'train_loss': train_loss_avg,
                'valid_loss': valid_loss,
                'precision': valid_precision,
                'recall': valid_recall,
                'f1_score': valid_f1_score})

        return self.model





