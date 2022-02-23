import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_loader
import augmentation
from trainer import Trainer

from models.cnn_basic import CNN_Basic


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--wandb_project', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model', type=str, default='cnn_basic')

    config = p.parse_args()

    return config

def get_model(config):
    if config.model == 'cnn_basic':
        model = CNN_Basic(name='cnn', xdim=[3, 256, 256], ksize=3, cdims=[16, 32, 64], hdims=[576, 128], ydim=18, USE_BATCHNORM=False)
    else:
        raise NotImplementedError('You need to specify model name.')
    return model


def prediction(image_id : str):

    image = Image.open(test_image_path + image_id)
    preprocessed_image = augmentation.basic_transforms(image)
    input_image = preprocessed_image.unsqueeze(0)
    
    with torch.no_grad():
        model_pred = trained_model(input_image)
        _, y_pred = torch.max(model_pred.data, 1)

    return y_pred


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    config.device = device
    
    train_loader, valid_loader = get_loader(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))

    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(model, optimizer, crit)
    trained_model = trainer.train(config, train_loader, valid_loader)

    #prediction
    test_dataset = pd.read_csv('../input/data/eval/info.csv')
    test_image_path = '../input/data/eval/images/'
    test_dataset['ans'] = test_dataset['ImageID'].apply(prediction)

    test_datset.to_csv('../output/ans.csv')
    


if __name__ == '__main__':
    config = define_argparser()
    main(config)
