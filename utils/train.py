import torch
from utils.engine import train_one_epoch, evaluate
from utils.dataset import get_dataset
from utils.model import get_model_instance_segmentation
import torch.optim as optim

def train(optimizer, num_classes, num_epochs, scheduler, device):
    load = get_dataset()
    model = get_model_instance_segmentation(num_classes)
    model = model.to(device)

    if optimizer == 'Adam':
        exp_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        exp_optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    if scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(exp_optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, exp_optimizer, load['train'], device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, load['val'], device=device)

    torch.save(model.state_dict(), 'best_model')
    
    print('Finished')