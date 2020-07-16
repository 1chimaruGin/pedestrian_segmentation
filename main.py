import torch
import argparse
from utils.train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optimizer', type=str, default=None, help='Optimizer')
    parser.add_argument('-n', '--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-sch', '--lr_scheduler', type=bool, default=True, help='LR_Scheduler')
    args = parser.parse_args()

    train(args.optimizer, args.num_classes, args.epochs, args.lr_scheduler, device)