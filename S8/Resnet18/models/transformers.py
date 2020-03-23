# import torch
# import torchvision
# import torchvision.transforms as transforms
#
# def transformers(load):
#     SEED = 1
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("CUDA Available?", device)
#     # For reproducibility
#     torch.manual_seed(SEED)
#
#     if device:
#         torch.cuda.manual_seed(SEED)
#
#         dataloader_args = dict(shuffle=True, batch_size=4, num_workers=2, pin_memory=True) if device else dict(
#             shuffle=True, batch_size=4)
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#         trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         if load == 'test':
#             testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
#             return testloader
#         elif load == 'train':
#             trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
#             return trainloader
