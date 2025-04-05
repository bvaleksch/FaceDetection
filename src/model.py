import torch
import torch.nn as nn

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MyModel(nn.Module):
    def __init__(self, image_size=(3, 128, 128)):
        super(MyModel, self).__init__()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.image_size = image_size
        x = torch.rand([1] + list(image_size))

        self.x1 = nn.Conv2d(self.image_size[0], 32, (3, 3))
        x = self.x1(x)
        self.b1 = nn.BatchNorm2d(32)
        x  = self.b1(x)
        self.m1 = nn.AvgPool2d((2, 2))
        x = self.m1(x)

        self.x2 = nn.Conv2d(32, 64, (3, 3))
        x = self.x2(x)
        self.b2 = nn.BatchNorm2d(64)
        x = self.b2(x)
        self.m2 = nn.AvgPool2d((2, 2))
        x = self.m2(x)

        self.x3 = nn.Conv2d(64, 128, (3, 3))
        x = self.x3(x)
        self.b3 = nn.BatchNorm2d(128)
        x = self.b3(x)
        self.m3 = nn.AvgPool2d((2, 2))
        x = self.m3(x)

        self.x4 = nn.Conv2d(128, 64, (3, 3))
        x = self.x4(x)
        self.b4 = nn.BatchNorm2d(64)
        x = self.b4(x)
        self.m4 = nn.AvgPool2d((2, 2))
        x = self.m4(x)

        self.f = nn.Flatten()
        x = self.f(x)
        self.l1 = nn.Linear(x.size()[1], 326)
        x = self.l1(x)
        self.l2 = nn.Dropout(0.2)
        x = self.l2(x)
        self.l3 = nn.Linear(326, 143)
        x = self.l3(x)
        self.l4 = nn.Dropout(0.2)
        x = self.l4(x)
        self.l5 = nn.Linear(143, 4)
        x = self.l5(x)

    def forward(self, x):
        x = self.gelu(self.x1(x))
        x = self.gelu(self.b1(x))
        x = self.gelu(self.m1(x))

        x = self.gelu(self.x2(x))
        x = self.gelu(self.b2(x))
        x = self.gelu(self.m2(x))

        x = self.gelu(self.x3(x))
        x = self.gelu(self.b3(x))
        x = self.gelu(self.m3(x))

        x = self.gelu(self.x4(x))
        x = self.gelu(self.b4(x))
        x = self.gelu(self.m4(x))

        x = self.f(x)

        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)

        x = self.l3(x)
        x = self.gelu(x)
        x = self.l4(x)

        x = self.l5(x)
        x = self.sigmoid(x)

        return x