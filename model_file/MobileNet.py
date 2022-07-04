import torch
import torch.nn as nn

def make_divisible(channel, divisor=8, min_channel=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_channel is None:
        min_channel = divisor
    new_channel = max(min_channel, int(channel + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channel < 0.9 * channel:
        new_channel += divisor
    return new_channel

class ConvBNReLU_Rearrange(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU_Rearrange, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual_Mobile_Rearrange(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual_Mobile_Rearrange, self).__init__()
        hidden_channel = in_channel * expand_ratio

        self.use_shortcut = True if sum(stride) == 2 and in_channel == out_channel else False

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU_Rearrange(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
			ConvBNReLU_Rearrange(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2_Rearrange(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2_Rearrange, self).__init__()
        block = InvertedResidual_Mobile_Rearrange
        input_channel = make_divisible(32 * alpha, round_nearest)
        last_channel = make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        features = []
        features.append(ConvBNReLU_Rearrange(1, input_channel, stride=1))
		
        for t, c, n, s in inverted_residual_setting:
            out_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, out_channel, stride=(1, stride), expand_ratio=t))
                input_channel = out_channel
        
        features.append(ConvBNReLU_Rearrange(input_channel, last_channel, 1))
        
        self.features = nn.Sequential(*features)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def mobilenetV2_rearrange():
    return MobileNetV2_Rearrange(num_classes=2)