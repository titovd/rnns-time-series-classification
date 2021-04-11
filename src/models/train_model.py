import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from rnns_models import *

import catalyst
from catalyst import dl


MODELS_DICT = {
    "simple-rnn" : SimpleRNN,
    "simple-gru" : SimpleGRU,
    "simple-lstm" : SimpleLSTM,
    "lstm-attention" : AttentionModel,
    "gru-fcn" : GRUFCNModel
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-rnn')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--path', type=str)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--enable-cuda", action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    X_tr = np.load(os.path.join(args.path, 'train_wisdm_data.npy'))
    y_tr = np.load(os.path.join(args.path, 'train_wisdm_label.npy'))

    X_val = np.load(os.path.join(args.path, 'val_wisdm_data.npy'))
    y_val = np.load(os.path.join(args.path, 'val_wisdm_label.npy'))

    X_test = np.load(os.path.join(args.path, 'test_wisdm_data.npy'))
    y_test = np.load(os.path.join(args.path, 'test_wisdm_label.npy'))

    train_loader = DataLoader(list(zip(X_tr, np.argmax(y_tr, axis=1))),
                              batch_size=args.batch_size, 
                              shuffle=True)
    val_loader = DataLoader(list(zip(X_val, np.argmax(y_val, axis=1))),
                              batch_size=args.batch_size, 
                              shuffle=True)
    loaders = {
        "train" : train_loader,
        "valid" : val_loader,
    }

    model = MODELS_DICT[args.model](3, 6, args.hidden_size, 0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False)

    runner = dl.SupervisedRunner(
        input_key="features",
        output_key="logits",
        target_key="targets",
        loss_key="loss"
    )

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=args.num_epochs,
        valid_loader="valid",
        valid_metric="accuracy",
        minimize_valid_metric=False,
        verbose=False,
        callbacks = [
            dl.ConfusionMatrixCallback(
                 input_key="logits", target_key="targets", num_classes=6, 
                 class_names=['Jogging',  'LyingDown',  'Sitting',  'Stairs',  'Standing',  'Walking'],
                 normalized=True
            ),
            dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=6),
        ],
        loggers = [catalyst.loggers.csv.CSVLogger()]
    )
if __name__ == '__main__':
    main()


