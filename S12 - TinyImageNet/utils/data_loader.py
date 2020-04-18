import os
import pdb
import sys

import torch.utils.data
from torchvision import datasets, transforms
from utils.has_cuda import *

device = has_cuda()

root_path = "C:/Users/gajanana_ganjigatti/Documents/Gaju_data/Quest/eva4/S12/tiny-imagenet-200/tiny-imagenet-200"
dataloader_args = dict(batch_size=512, num_workers=2, pin_memory=True) if device else dict( batch_size=4)
imagenet_traindir = os.path.join(root_path, 'train')
imagenet_valdir = os.path.join(root_path, 'val')
imagenet_testdir = os.path.join(root_path, 'test')
imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def tinyImgNet_dataloader(split='train'):

    if split == 'train':
        data_transform = datasets.ImageFolder(
            imagenet_traindir,
                transform=transforms.Compose([transforms.Pad(padding=1, padding_mode="edge"),
                transforms.RandomHorizontalFlip(),  
                transforms.RandomRotation(20),
                transforms.RandomCrop(size=(64, 64), padding=4),
#                 transforms.RandomErasing(scale=(0.16, 0.16), ratio=(1, 1)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)]))

        shuffle_flag = True
    elif split == 'val':
        data_transform = datasets.ImageFolder(
            imagenet_valdir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)]))

        shuffle_flag = False
    else:
        data_transform = datasets.ImageFolder(
            imagenet_testdir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)]))
        shuffle_flag = False

    loader = torch.utils.data.DataLoader(data_transform, shuffle=shuffle_flag, **dataloader_args)

    return loader
