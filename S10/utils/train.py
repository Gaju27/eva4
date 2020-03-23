import torch
import torchvision
# import torchvision.transforms as transforms
from .data_loader import *
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

#SEED = 1
#
## CUDA?
## cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## print("CUDA Available?", device)
#
## For reproducibility
#torch.manual_seed(SEED)
#
#if device:
#    torch.cuda.manual_seed(SEED)
#
## dataloader arguments - something you'll fetch these from cmdprmt
#dataloader_args = dict(shuffle=True, batch_size=64, num_workers=2, pin_memory=True) if device else dict(shuffle=True, batch_size=4)
#
#transform = transforms.Compose(
#    [transforms.RandomResizedCrop(32),
#        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#
#trainloader = torch.utils.data.DataLoader(trainset,** dataloader_args)
## transform()

def train(net, device, optimizer,criterion, epoch,trainloader):
  #trainloader = dataloader_cifar10(root='./data',, split='train', batch_size=args.batch_size)
  net.train()
  pbar = tqdm(trainloader)
  correct = 0
  processed = 0
  for batch_idx, (input, labels) in enumerate(pbar):
    # get samples
    input, labels = input.to(device), labels.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    outputs = net(input)

    # Calculate loss
#     loss = F.nll_loss(outputs, target)
    loss = criterion(outputs, labels)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    _, pred = torch.max(outputs, 1)  # get the index of the max log-probability
    correct += pred.eq(labels.view_as(pred)).sum().item()
    processed += len(input)

    pbar.set_description(desc= f'Epoch={epoch} Loss={loss.item()} Batch_id={batch_idx} Training Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
    
