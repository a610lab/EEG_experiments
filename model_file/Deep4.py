import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep4Net(nn.Module):
    def __init__(self):
        super(Deep4Net, self).__init__()
        self.elu = nn.ELU()
        self.drop = nn.Dropout(0.5)

        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=1)


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 3), stride=1,
                               padding=(0, 1), bias=False)
        self.conv11 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(3, 1), stride=1,
                                padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(25, momentum=0.1, affine=True, eps=1e-5)

        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False)

        self.bn2 = nn.BatchNorm2d(50, momentum=0.1, affine=True, eps=1e-5)

        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(100)

        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(200)

        # self.fc = nn.Linear(200*1*9, 2, bias=False)
        # self.fc = nn.Linear(200*1*12, 2, bias=False)
        self.fc = nn.Linear(200*1*11, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.max1(x)

        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.max1(x)

        x = self.drop(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.max1(x)

        x = self.drop(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.max1(x)

        # x = x.view(-1, 200 * 1 * 9)
        # x = x.view(-1, 200 * 1 * 12)
        x = x.view(-1, 200 * 1 * 11)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def deep4():

    return Deep4Net()
