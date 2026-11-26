# 1. Поток данных, 2. как объявить модель 3. цикл обучения
from pyexpat import model
from random import shuffle
from tkinter import HIDDEN
from tracemalloc import start

from sympy import im
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T

train_set = MNIST('.MNIST', transform=T.ToTensor(), train=True,download=True)
test_set = MNIST('.MNIST', transform=T.ToTensor(), train = False,download=True)

images,labels = next(iter(train_set))
"""
plt.imshow(images[0], cmap = 'gray')
plt.title(labels)
plt.show()"""

# DataLoader -  Собирает отдельные примеры из датасета в мини-батчи (batch_size), объединяя их в тензоры, чтобы сеть могла обрабатывать данные параллельно.
train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(test_set,batch_size=4096,shuffle=True)

# Объявления модели
from torch import nn
class Firstnet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_features,hidden_features//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features//2,out_features)    
        )

    def forward(self, x):
        h = torch.flatten(x, start_dim=1) #Распоковка тензора в вектор. start_dem=1 пропуск первого элемента(там инфа о батче)
        return self.model(h)

model = Firstnet(784,128,10)
print(model(images[None,:,:,:]))

IMG_SIZE = 28
NUM_CLASSES = 10
HIDDEN_SIZE = 64
NUM_EPOCHS = 3
LR = 0.01
model = Firstnet(IMG_SIZE**2, HIDDEN_SIZE, NUM_CLASSES)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0)
criterion = nn.CrossEntropyLoss()

from tqdm.notebook import tqdm

for epoch in range(NUM_EPOCHS):

    model.train()#модель учиться(в написанной модели вкл dropout)
    for images, labels in tqdm(train_loader, desc=f'Ушла эпоха{epoch}'):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step() #обновление градиентного спуска

    model.eval()#скор данных(выкл dropout)-считается матожидание
    val_loss, val_correct, val_n = 0.0,0,0
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            val_loss += criterion(logits,labels).item() * len(images)

            val_correct += (logits.argmax(dim=-1)==labels).sum().item()
            val_n += len(images)

    print(f'Epoch {epoch}: val loss {val_loss / val_n:.4f}, val_accuracy {val_correct / val_n:.4f}')
        
