import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from tqdm import tqdm
import wandb

from model import GenderClassifier
from dataloader import get_gender_loaders
from utils import get_metrics


def define_argparser():
    p = argparse.ArgumentParser()

    # p.add_argument('--model_fn', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--train_valid_ratio', type=float, default=0.8)
    p.add_argument('--random_seed', type=int, default=42)

    p.add_argument('--wandb_project', type=str, required=True)

    config = p.parse_args()

    return config


def main(config):
    wandb.init(project=config.wandb_project, entity='dooholee')
    wandb.config = {
        'epochs': config.n_epochs,
        'batch_size': config.batch_size,
        'random_seed': config.random_seed,
    }

    # set random seed
    torch.manual_seed(config.random_seed)      # About PyTorch
    torch.backends.cudnn.deterministic = True  # About CuDNN -> Maybe cause slow training ?
    torch.backends.cudnn.benchmark = False     # About CuDNN
    np.random.seed(config.random_seed)         # About Numpy
    random.seed(config.random_seed)            # About transform -> Also should be in augmentation.py ?
    torch.cuda.manualseed(config.random_seed)  # About GPU

    best_test_loss = 9999.

    device = torch.device(0)

    model = GenderClassifier().to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    train_loader, valid_loader = get_gender_loaders(config)


    # start
    for epoch_index in range(config.n_epochs):
        running_loss = 0.

        # train
        model.train()
        pbar = tqdm(train_loader)
        for _, (X, y) in enumerate(pbar):
            X = X.to(device)
            X = X.float()

            y = y.to(device)
            y = y.float()

            y_hat = model(X)
            y_hat = torch.squeeze(y_hat)

            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss)
            pbar.set_postfix({'epoch': f'{epoch_index}/{config.n_epochs}', 'loss': f'{running_loss/config.batch_size:.5f}'})
    
        train_loss = running_loss / len(train_loader)

        
        # valid
        model.eval()
        with torch.no_grad():
            valid_loss = 0.
            valid_precision = 0.
            valid_recall = 0.
            valid_f1_score = 0.
            for _, (X, y) in enumerate(valid_loader):
                X, y = X.to(device), y.to(device)
                X = X.float()
                y = y.float()

                y_hat = model(X)
                y_hat = torch.squeeze(y_hat)

                loss = loss_fn(y_hat, y)

                valid_loss += float(loss)

                precision, recall, f1_score = get_metrics(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
                valid_precision += precision
                valid_recall += recall
                valid_f1_score += f1_score

            valid_loss /= len(valid_loader)
            valid_precision /= len(valid_loader)
            valid_recall /= len(valid_loader)
            valid_f1_score /= len(valid_loader)
        

        # Save
        if valid_loss <= best_test_loss:
            best_test_loss = valid_loss

        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'epoch': epoch_index,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': valid_loss},
        #     f'checkpoint/epoch{epoch_index}_{valid_loss:.5f}.pt')

        print(f'Epoch-{epoch_index} Train Loss: {train_loss:.5f}  Valid Loss: {valid_loss:.5f}  Best Valid Loss: {best_test_loss:.5f}  Precision: {valid_precision:.5f}  Recall: {valid_recall:.5f}  f1_score: {valid_f1_score:.5f}')

        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'precision': valid_precision,
            'recall': valid_recall,
            'f1_score': f1_score})


if __name__ == '__main__':
    config = define_argparser()
    main(config)