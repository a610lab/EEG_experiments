"""
时间:2021/7/21
"""
import torch
import torch.nn as nn

class BasicBlock_Rearrange(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock_Rearrange, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=(1, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = self.relu(x)

        return x

class Bottleneck_Rearrange(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(Bottleneck_Rearrange, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=(1, stride), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=self.expansion * out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.relu(x)

        return x

class ResNet_Rearrange(nn.Module):

    def __init__(self, block, blocks_num, num_classes=None, include_top=True):
        super(ResNet_Rearrange, self).__init__()

        if num_classes is None:
            raise Exception('num_classes cannot be None')

        self.include_top = include_top
        self.in_channel = 64
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.layer1 = self._make_layer(block, blocks_num[0], channel=64)
        self.layer2 = self._make_layer(block, blocks_num[1], channel=128, stride=2)
        self.layer3 = self._make_layer(block, blocks_num[2], channel=256, stride=2)
        self.layer4 = self._make_layer(block, blocks_num[3], channel=512, stride=2)

        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, block_num, channel, stride=1):
        down_sample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion,
                          kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride, down_sample))

        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)

        return x

def resnet34_rearrange(num_classes=1000, include_top=True):
    return ResNet_Rearrange(BasicBlock_Rearrange, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50_rearrange(num_classes=1000, include_top=True):
    return ResNet_Rearrange(Bottleneck_Rearrange, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101_rearrange(num_classes=1000, include_top=True):
    return ResNet_Rearrange(Bottleneck_Rearrange, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)