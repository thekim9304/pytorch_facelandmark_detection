import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV1(nn.Module):
    def __init__(self, in_size, num_landmark):
        super(MobileNetV1, self).__init__()

        self.in_size = in_size
        self.num_landmark = num_landmark

        if in_size[0] == 40:
            self.last_size = 2
        elif in_size[0] == 100:
            self.last_size = 4
        elif in_size[0] == 200:
            self.last_size = 7
        elif in_size[0] == 300:
            self.last_size = 10

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(self.in_size[-1], 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
        )
        self.conv1 = conv_dw(512, 1024, 2)
        self.conv2 = conv_dw(1024, 1024, 1)

        self.fc = nn.Linear(1024, num_landmark)

    def forward(self, x):
        x = self.model(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, self.last_size)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def main():
    size = 200

    in_tensor = torch.randn(1, 1, size, size).cuda()

    model = MobileNetV1((size, size, 1), 42).cuda()

    out_tensor = model(in_tensor)

    print(out_tensor.shape)

if __name__ == '__main__':
    main()
