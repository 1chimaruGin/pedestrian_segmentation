import torch
from engine import train_one_epoch, evaluate
from dataset import get_dataset
from model import get_model_instance_segmentation
import torch.optim as optim
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--optimizer', type=str, default=None, help='Optimizer')
    parser.add_argument('-n', '--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-sch', '--lr_scheduler', type=bool, default=True, help='LR_Scheduler')
    args = parser.parse_args()

    def train(optimizer, num_classes, num_epochs, scheduler):
        load = get_dataset()
        model = get_model_instance_segmentation(num_classes)
        model = model.to(device)

        if optimizer == 'Adam':
            exp_optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            exp_optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9,weight_decay=0.0005)

        if scheduler:
            lr_scheduler = optim.lr_scheduler.StepLR(exp_optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            train_one_epoch(model, exp_optimizer, load['train'], device, epoch, print_freq=10)
            lr_scheduler.step()
            evaluate(model, load['val'], device=device)

        torch.save(model.state_dict(), 'best_model')
    
        print('Finished')

    train(args.optimizer, args.num_classes, args.epochs, args.lr_scheduler)