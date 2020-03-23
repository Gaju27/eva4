import torch
import torch as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def get_arguments():
    SEED = 1
    # CUDA?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For reproducibility
    torch.manual_seed(SEED)

    if device:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=64, num_workers=2, pin_memory=True) if device else dict(shuffle=True, batch_size=4)

    return dataloader_args

def get_optimizer(net):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    return optimizer
    
