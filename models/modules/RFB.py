import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.gelu(x)


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.gelu(x)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=(3, 3), dilation=(3, 3))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 5), padding=(1, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 3), padding=(2, 1)),
            BasicConv2d(out_channel, out_channel, 3, padding=(5, 5), dilation=(5, 5))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 7), padding=(2, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 5), padding=(3, 2)),
            BasicConv2d(out_channel, out_channel, 3, padding=(7, 7), dilation=(7, 7))
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class RFB3D_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB3D_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicConv3d(out_channel, out_channel, 3, padding=(1, 3, 3), dilation=(1, 3, 3))
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 5, 3), padding=(1, 2, 1)),
            BasicConv3d(out_channel, out_channel, 3, padding=(1, 5, 5), dilation=(1, 5, 5))
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(5, 5, 7), padding=(2, 2, 3)),
            BasicConv3d(out_channel, out_channel, kernel_size=(5, 7, 5), padding=(2, 3, 2)),
            BasicConv3d(out_channel, out_channel, 3, padding=(1, 7, 7), dilation=(1, 7, 7))
        )
        self.conv_cat = BasicConv3d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv3d(in_channel, out_channel, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



if __name__ == '__main__':
    import time, os

    start = time.time()
    a = torch.randn((8, 6, 12, 56, 56))
    te = RFB3D_modified(6, 12)
    print(te(a).shape)
    print(time.time() - start)
