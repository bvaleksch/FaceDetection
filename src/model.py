import torch.nn as nn

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

def maxpool_output_shape(h_w, kernel_size=2):
    """
    Utility function for computing output of max pooling
    takes a tuple of (h, w) and returns a tuple of (h, w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    # Calculate output height and width
    h = h_w[0] // kernel_size[0]
    w = h_w[1] // kernel_size[1]
    
    return h, w


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
        self.image_size = image_size
        self.out = image_size[1:]
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.x1 = SeparableConv2D(self.image_size[0], 32, (3, 3))
        self.out = conv_output_shape(conv_output_shape(self.out, (3, 3)), 1)
        self.b1 = nn.BatchNorm2d(32)
        self.m1 = nn.MaxPool2d((2, 2))
        self.out = maxpool_output_shape(self.out, (2, 2))

        self.x2 = SeparableConv2D(32, 64, (3, 3))
        self.out = conv_output_shape(conv_output_shape(self.out, (3, 3)), 1)
        self.b2 = nn.BatchNorm2d(64)
        self.m2 = nn.MaxPool2d((2, 2))
        self.out = maxpool_output_shape(self.out, (2, 2))

        self.x3 = SeparableConv2D(64, 128, (3, 3))
        self.out = conv_output_shape(conv_output_shape(self.out, (3, 3)), 1)
        self.b3 = nn.BatchNorm2d(128)
        self.m3 = nn.MaxPool2d((2, 2))
        self.out = maxpool_output_shape(self.out, (2, 2))

        self.f = nn.Flatten()
        self.l = nn.Linear(self.out[0]*self.out[1]*128, 4)

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

        x = self.f(x)
        x = self.l(x)
        x = self.sigmoid(x)

        return x
