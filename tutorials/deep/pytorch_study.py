import torch
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T

train_set = MNIST(
    '.MNIST',
    transform=T.ToTensor(),
    train=True,
    download=True
)
print(len(train_set))
print(train_set.data.shape)
print(train_set.targets.shape)
print(train_set.data[99])
print(train_set.data[99].shape)
print(train_set.targets[99])
print(train_set.data.min().item(), train_set.data.max().item())