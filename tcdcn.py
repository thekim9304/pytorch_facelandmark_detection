import time

import torch
import torch.nn as nn

class TCDCN(nn.Module):
    def __init__(self, in_size, num_landmark):
        super(TCDCN, self).__init__()

        self.in_size = in_size
        self.num_landmark = num_landmark

        if in_size[0] == 40:
            self.last_size = 2
        elif in_size[0] == 100:
            self.last_size = 9
        elif in_size[0] == 200:
            self.last_size = 22
        elif in_size[0] == 300:
            self.last_size = 34

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_size[-1], out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(64*self.last_size*self.last_size, num_landmark)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(-1, 64*self.last_size*self.last_size)
        x = self.fc(x)

        return x

def main():
    size = 100

    in_tensor = torch.randn(1, 1, size, size).cuda()

    model = TCDCN((size, size, 1), 42).cuda().eval()

    for _ in range(100):
        prev_time = time.time()
        out_tensor = model(in_tensor)
        print(time.time() - prev_time)

    print(out_tensor.shape)


if __name__ == '__main__':
    main()
