import argparse
from copy import deepcopy
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import wandb

from model import GenderClassifier, MaskClassifier
from dataloader import get_gender_loaders, get_mask_loaders
from utils import get_metrics


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--train_valid_ratio', type=float, default=0.8)
    p.add_argument('--random_seed', type=int, default=42)

    p.add_argument('--wandb_project', type=str, required=True)
    p.add_argument('--wandb_entity', type=str, required=True)
    p.add_argument('--classes', type=int, required=True)

    config = p.parse_args()

    return config


def main(config):
    # hide user warnings in CLI
    warnings.filterwarnings(action='ignore')

    # wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    # wandb.config = {
    #     'epochs': config.n_epochs,
    #     'batch_size': config.batch_size,
    #     'random_seed': config.random_seed,
    # }

    # set random seed
    torch.manual_seed(config.random_seed)       # About PyTorch
    torch.backends.cudnn.deterministic = True   # About CuDNN -> Maybe cause slow training ?
    torch.backends.cudnn.benchmark = False      # About CuDNN
    np.random.seed(config.random_seed)          # About Numpy
    random.seed(config.random_seed)             # About transform -> Also should be in augmentation.py ?
    torch.cuda.manual_seed(config.random_seed)  # About GPU

    best_test_loss = 9999.

    device = torch.device(0)

    # model = GenderClassifier().to(device)
    model = MaskClassifier().to(device)
    # loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss 의 label 은 long 타입이어야 함 ([0.2, 0.5, 0.9], [2])
    optimizer = optim.Adam(model.parameters())
    # train_loader, valid_loader = get_gender_loaders(config)
    train_loader, valid_loader = get_mask_loaders(config)


    # start
    for epoch_index in range(config.n_epochs):
        running_loss = 0.

        # train
        model.train()
        pbar = tqdm(train_loader)
        for _, (X, y) in enumerate(pbar):
            X = X.to(device).float()
            y = y.to(device).float() if config.classes < 3 else y.to(device)

            y_hat = model(X)
            y_hat = torch.squeeze(y_hat)

            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss)
            pbar.set_postfix({
                'epoch': f'{epoch_index+1}/{config.n_epochs}',
                'loss': f'{running_loss/config.batch_size:.5f}'})
    
        train_loss = running_loss / len(train_loader)

        
        # valid
        model.eval()
        with torch.no_grad():
            if config.classes > 2:
                # this for sklearn.metrics.classification_report
                y_all = []
                y_hat_all = []

            valid_loss = 0.
            valid_precision = 0.
            valid_recall = 0.
            valid_f1_score = 0.
            for _, (X, y) in enumerate(valid_loader):
                X = X.to(device).float()
                y = y.to(device).float() if config.classes < 3 else y.to(device)

                y_hat = model(X)
                y_hat = torch.squeeze(y_hat)

                loss = loss_fn(y_hat, y)

                valid_loss += float(loss)

                # average (for is_binary=False)
                # - micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
                # - macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                # - weighted: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                # - samples: Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score
                precision, recall, f1_score = get_metrics(
                    y.cpu().detach().numpy(),
                    y_hat.cpu().detach().numpy(),
                    is_binary=True if config.classes < 3 else False,
                    average='macro')

                valid_precision += precision
                valid_recall += recall
                valid_f1_score += f1_score

                if config.classes > 2:
                    # for look multi class metrics
                    y_all.extend(y.cpu().detach().numpy())
                    y_hat_all.extend(y_hat.cpu().detach().numpy().argmax(axis=1))


            valid_loss /= len(valid_loader)
            valid_precision /= len(valid_loader)
            valid_recall /= len(valid_loader)
            valid_f1_score /= len(valid_loader)
        

        # Check best valid loss
        if valid_loss <= best_test_loss:
            best_test_loss = valid_loss
            best_loss_model = deepcopy(model.state_dict())
            best_loss_optimizer = deepcopy(optimizer.state_dict())
            best_loss_epoch = epoch_index
            best_loss_precision = valid_precision
            best_loss_recall = valid_recall
            best_loss_f1_score = valid_f1_score


        # # for checkpoint?
        # type in here...


        print(f'Epoch-{epoch_index+1} Train Loss: {train_loss:.5f}  Valid Loss: {valid_loss:.5f}  Best Valid Loss: {best_test_loss:.5f}  Precision: {valid_precision:.5f}  Recall: {valid_recall:.5f}  f1_score: {valid_f1_score:.5f}')
        
        # wandb.log({
        #     'train_loss': train_loss,
        #     'valid_loss': valid_loss,
        #     'precision': valid_precision,
        #     'recall': valid_recall,
        #     'f1_score': valid_f1_score})
    

    if config.classes > 2:
        print(classification_report(
            y_all,
            y_hat_all,
            target_names=['none', 'incorrect', 'correct']))


    # wandb artifacts
    # torch.save({
    #     'model_state_dict': best_loss_model,
    #     'epoch': best_loss_epoch,
    #     'optimizer_state_dict': best_loss_optimizer,
    #     'loss': best_test_loss,
    #     'precision': best_loss_precision,
    #     'recall': best_loss_recall,
    #     'f1_score': best_loss_f1_score},
    #     f'checkpoint/{config.model_fn}_epoch{best_loss_epoch}_{best_test_loss:.5f}.pt')
    
    # run = wandb.init(project=config.wandb_project)
    # artifact = wandb.Artifact(config.model_fn, type='model')
    # artifact.add_file(f'checkpoint/{config.model_fn}_epoch{best_loss_epoch}_{best_test_loss:.5f}.pt')
    # run.log_artifact(artifact)
    # run.finish()


if __name__ == '__main__':
    config = define_argparser()
    main(config)