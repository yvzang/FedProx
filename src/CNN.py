import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    '''def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.AvgPool = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))               # (3,32,32) -> (16,32,32)
        x = self.MaxPool(F.relu(self.conv2(x))) # (16,32,32) -> (32,16,16)
        x = F.relu(self.conv3(x))               # (32,16,16) -> (64,16,16)
        x = self.MaxPool(F.relu(self.conv4(x))) # (64,16,16) -> (128,8,8)
        x = self.MaxPool(F.relu(self.conv5(x))) # (128,8,8) -> (256,4,4)
        x = self.AvgPool(x)                          # (256,1,1)
        x = x.view(-1, 256)                     # (256)
        x = self.fc3(self.fc2(self.fc1(x)))     # (32)
        x = self.fc4(x)                         # (10)
        return x'''
    def __init__(self) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, tens):
        tens = self.sequential(tens)
        return tens
    