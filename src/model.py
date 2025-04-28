import torch
import torch.nn as nn


class MyConv2d(nn.Module):
    def __init__(self, inp, out, kernel_size, stride=1, padding="same"):
        super(MyConv2d, self).__init__()
        self.function = nn.GELU()
        self.x1 = nn.Conv2d(inp, out, kernel_size, stride, padding)
        self.x2 = nn.Conv2d(out, out, kernel_size, stride, padding)

    def forward(self, x):
        x = self.function(self.x1(x))
        x = self.function(self.x2(x))

        return x

class FirstModel(nn.Module):
    def __init__(self, image_size=(3, 128, 128)):
        super(FirstModel, self).__init__()
        self.gelu = nn.GELU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.image_size = image_size
        kernel_size = (3, 3)
        x = torch.rand([1] + list(image_size))

        self.x1 = MyConv2d(self.image_size[0], 32, kernel_size)
        x = self.x1(x)
        self.m1 = nn.MaxPool2d((2, 2))
        x = self.m1(x)
        self.b1 = nn.BatchNorm2d(32)
        x  = self.b1(x)

        self.x2 = MyConv2d(32, 64, kernel_size)
        x = self.x2(x)
        self.m2 = nn.MaxPool2d((2, 2))
        x = self.m2(x)
        self.b2 = nn.BatchNorm2d(64)
        x = self.b2(x)

        self.x3 = MyConv2d(64, 128, kernel_size)
        x = self.x3(x)
        self.m3 = nn.MaxPool2d((2, 2))
        x = self.m3(x)
        self.b3 = nn.BatchNorm2d(128)
        x = self.b3(x)

        self.x4 = MyConv2d(128, 256, kernel_size)
        x = self.x4(x)
        self.m4 = nn.MaxPool2d((2, 2))
        x = self.m4(x)
        self.b4 = nn.BatchNorm2d(256)
        x = self.b4(x)

        self.x5 = MyConv2d(256, 256, kernel_size)
        x = self.x5(x)
        self.m5 = nn.MaxPool2d((2, 2))
        x = self.m5(x)
        self.b5 = nn.BatchNorm2d(256)
        x = self.b5(x)

        x = self.flatten(x)

        self.l1 = nn.Linear(x.size()[1], 512)
        self.l2 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.gelu(self.x1(x))
        x = self.m1(x)
        x = self.b1(x)

        x = self.gelu(self.x2(x))
        x = self.m2(x)
        x = self.b2(x)

        x = self.gelu(self.x3(x))
        x = self.m3(x)
        x = self.b3(x)

        x = self.gelu(self.x4(x))
        x = self.m4(x)
        x = self.b4(x)

        x = self.gelu(self.x5(x))
        x = self.m5(x)
        x = self.b5(x)

        x = self.flatten(x)

        x = self.dropout(self.gelu(self.l1(x)))
        x = self.dropout(self.gelu(self.l2(x)))
        x = self.dropout(self.gelu(self.l3(x)))
        x = self.dropout(self.gelu(self.l4(x)))
        x = self.sigmoid(self.l5(x))

        return x

