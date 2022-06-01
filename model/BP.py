import torch.nn as nn
import torch.nn.functional as F
import torch


class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.layer1 = nn.Linear(9, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 16)
        self.layer4 = nn.Linear(16, 6)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        
    def forward(self, x): 
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x) 
        output = F.log_softmax(x, dim=1)
        return output