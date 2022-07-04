"""
时间:2022/3/31
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25), stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(40, momentum=0.1, affine=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop1 = nn.Dropout(0.5)
        self.fc = nn.Linear(40 * 1 * 21, 2, bias=False)
        # self.fc = nn.Linear(40 * 1 * 19, 2, bias=False)
        # self.fc = nn.Linear(40 * 1 * 14, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x.mul(x)
        x = self.pool1(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop1(x)

        # x = x.view(-1, 40 * 1 * 19)
        # x = x.view(-1, 40 * 1 * 14)
        x = x.view(-1, 40 * 1 * 21)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def shallow_net():

    return ShallowNet()
