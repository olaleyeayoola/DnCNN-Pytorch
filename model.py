import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1,
            bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        for _ in range(18):
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
        out = self.conv3(out)
        return out
