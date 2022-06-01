from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 16)
        self.layer4 = nn.Linear(16, 6)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.conv1 = nn.Conv1d(9, 16, 1)
        self.conv2 = nn.Conv1d(16, 64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 1)
        x = F.relu(self.layer1(x))
        output = F.log_softmax(x, dim=1)
        return output