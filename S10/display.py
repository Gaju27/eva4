import torch
from torchsummary import summary

def has_cuda():
    """Check for Cuda"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device
