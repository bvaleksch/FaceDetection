import torch
import torch.nn as nn


class MyConv2d(nn.Module):
    def __init__(self, inp, out, kernel_size, stride=1, padding=0):
        super(MyConv2d, self).__init__()
        self.function = nn.GELU()
        self.x1 = nn.Conv2d(inp, out, kernel_size, stride, padding)
        self.x2 = nn.Conv2d(out, out, kernel_size, stride, padding)
        self.x3 = nn.Conv2d(out, out, kernel_size, stride, padding)

    def forward(self, x):
        x = self.function(self.x1(x))
        x = self.function(self.x2(x))
        x = self.function(self.x3(x))

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
        x = self.gelu(self.m1(x))
        x = self.gelu(self.b1(x))

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

class MyModel2(nn.Module):
    def __init__(self, image_size=(3, 128, 128)):
        super(MyModel2, self).__init__()
        self.gelu = nn.GELU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.image_size = image_size
        kernel_size = (3, 3)
        x = torch.rand([1] + list(image_size))

        self.x1 = MyConv2d(self.image_size[0], 32, kernel_size)
        x = self.x1(x)
        self.m1 = nn.AvgPool2d((2, 2))
        x = self.m1(x)
        self.b1 = nn.BatchNorm2d(32)
        x  = self.b1(x)

        self.x2 = MyConv2d(32, 64, kernel_size)
        x = self.x2(x)
        self.m2 = nn.AvgPool2d((2, 2))
        x = self.m2(x)
        self.b2 = nn.BatchNorm2d(64)
        x = self.b2(x)

        self.x3 = MyConv2d(64, 128, kernel_size)
        x = self.x3(x)
        self.m3 = nn.AvgPool2d((2, 2))
        x = self.m3(x)
        self.b3 = nn.BatchNorm2d(128)
        x = self.b3(x)

        self.x4 = MyConv2d(128, 256, kernel_size)
        x = self.x4(x)
        self.m4 = nn.AvgPool2d((2, 2))
        x = self.m4(x)
        self.b4 = nn.BatchNorm2d(256)
        x = self.b4(x)

        x = self.flatten(x)

        self.l1 = nn.Linear(x.size()[1], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 4)
        
    def forward(self, x):
        x = self.gelu(self.x1(x))
        x = self.gelu(self.m1(x))
        x = self.gelu(self.b1(x))

        x = self.gelu(self.x2(x))
        x = self.gelu(self.m2(x))
        x = self.gelu(self.b2(x))

        x = self.gelu(self.x3(x))
        x = self.gelu(self.m3(x))
        x = self.gelu(self.b3(x))

        x = self.gelu(self.x4(x))
        x = self.gelu(self.m4(x))
        x = self.gelu(self.b4(x))

        x = self.flatten(x)

        x = self.dropout(self.gelu(self.l1(x)))
        x = self.dropout(self.gelu(self.l2(x)))
        x = self.sigmoid(self.l3(x))

        return x

class MyModel3(nn.Module):
    def __init__(self, image_size=(3, 128, 128)):
        super(MyModel3, self).__init__()
        self.gelu = nn.GELU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
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

        x = self.flatten(x)

        self.l1 = nn.Linear(x.size()[1], 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 4)
        
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

        x = self.flatten(x)

        x = self.dropout(self.gelu(self.l1(x)))
        x = self.dropout(self.gelu(self.l2(x)))
        x = self.dropout(self.gelu(self.l3(x)))
        x = self.sigmoid(self.l4(x))

        return x
