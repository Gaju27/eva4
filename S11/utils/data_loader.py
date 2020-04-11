import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as a
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, RandomBrightness, ShiftScaleRotate, \
    JpegCompression, HueSaturationValue, Cutout
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader
import cv2 as cv2
import numpy as np
from PIL import Image



def dataloader_cifar10(split='train'):


    SEED = 1
    # CUDA?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For reproducibility
    torch.manual_seed(SEED)

    if device:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if device else dict(
        shuffle=True, batch_size=4)

    if split == 'train':
        def stron_aug(p=.5):
            return Compose([a.augmentations.transforms.RandomCrop(32,32), 
                            a.augmentations.transforms.Cutout(num_holes=1, max_h_size=8, max_w_size=8)], p=p)

        def augment(aug, image):
            return aug(image=image)['image']

        class alb_transform(object):
            def __call__(self, img):
                aug = stron_aug(p=.9)
                return Image.fromarray(augment(aug, np.array(img)))

        data_transform = transforms.Compose([alb_transform(), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        train_flag = True
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_flag = False

    dataset = datasets.CIFAR10(root='./data', train=train_flag, download=True, transform=data_transform)
    loader = DataLoader(dataset, **dataloader_args)

    return loader
