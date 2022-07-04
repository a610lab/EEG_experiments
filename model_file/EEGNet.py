import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.elu = nn.ELU()

        self.avg1 = nn.AvgPool2d(kernel_size=(1, 4), stride=4)
        self.avg2 = nn.AvgPool2d(kernel_size=(1, 8), stride=8)


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.dw_conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 1), stride=1,
                                  padding=0, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.drop_out = nn.Dropout(0.25)

        self.dw_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), stride=1,
                                  padding=0, bias=False)
        self.pw_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # self.fc = nn.Linear(16 * 1 * 12, 2, bias=False)
        # self.fc = nn.Linear(16*1*9, 2, bias=False)
        self.fc = nn.Linear(16*1*11, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.__pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.dw_conv1(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg1(x)
        x = self.drop_out(x)

        x = self.__pad(x)
        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg2(x)
        x = self.drop_out(x)

        if x.size()[-1] == 12:
            x = x.view(-1, 16 * 1 * 12)
        elif x.size()[-1] == 9:
            x = x.view(-1, 16 * 1 * 9)
        elif x.size()[-1] == 11:
            x = x.view(-1, 16 * 1 * 11)
        x = self.fc(x)
        x = self.softmax(x)

        return x




    def __pad(self, x):
        p1d = (31, 32)
        p2d = (7, 8)

        if x.size()[-1] == 400 or x.size()[-1] == 300 or x.size()[-1] == 376:
            x_padded = F.pad(x, p1d, "constant", 0)
            return x_padded
        elif x.size()[-1] == 100 or x.size()[-1] == 75 or x.size()[-1] == 94:
            x_padded = F.pad(x, p2d, "constant", 0)
            return x_padded


def eegnet():

    return EEGNet()
