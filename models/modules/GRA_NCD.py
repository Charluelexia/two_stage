import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.RFB import BasicConv2d, RFB_modified


# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel, out_channel=1):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group * out_channel, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, out_channel, 3, padding=1)

    def forward(self, x, y):
        # reverse guided block
        y_ = -1 * (torch.sigmoid(y)) + 1

        if self.group == 1:
            x_cat = torch.cat((x, y_), 1)
        else:
            assert self.group % 2 == 0, f"{self.group} is invalid."
            list_xc = torch.chunk(x, self.group, dim=1)
            list_cat_xy = []
            for xc in list_xc:
                list_cat_xy.append(xc)
                list_cat_xy.append(y_)
            x_cat = torch.cat(list_cat_xy, 1)

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, channel // 4)
        self.strong_gra = GRA(channel, channel // 16)

    def forward(self, x, y):
        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y


class CrossUpdateLayer(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=3, act_func_y=None):
        super(CrossUpdateLayer, self).__init__()
        self.conv_update = nn.Conv2d(in_channel + mid_channel + out_channel, in_channel, 3, padding=1)
        self.conv_basic = BasicConv2d(out_channel, mid_channel, 3, padding=1)
        self.conv_res = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1),
                                      nn.Identity() if act_func_y is None else act_func_y())
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.relu(x + self.conv_update(torch.cat([x, self.conv_basic(y), 1 - torch.sigmoid(y)], dim=1)))
        y = y + self.conv_res(x)
        return x, y


class CrossUpdateBlock(nn.Module):
    def __init__(self, in_channel, out_channel=1, depth=3):
        super(CrossUpdateBlock, self).__init__()
        self.layers = nn.ModuleList()
        list_act_func=[nn.Mish]*(depth-2)
        list_act_func.insert(0,nn.ELU)
        list_act_func.insert(depth,nn.ReLU)
        for i in range(depth):
            self.layers.append(
                CrossUpdateLayer(in_channel, out_channel, mid_channel=16,
                                 act_func_y=list_act_func[i]))

    def forward(self, x, y):
        for sub in self.layers:
            x, y = sub(x, y)

        return y


class NeighborConnectionDecoder(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, in_channels):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels[2], in_channels[1], 3, padding=1)
        self.conv_upsample2 = BasicConv2d(in_channels[1], in_channels[0], 3, padding=1)
        self.conv_upsample3 = BasicConv2d(in_channels[1], in_channels[0], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(in_channels[2], in_channels[1], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * in_channels[1], 2 * in_channels[0], 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * in_channels[1], 2 * in_channels[1], 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * in_channels[0], 3 * in_channels[0], 3, padding=1)
        self.conv4 = BasicConv2d(3 * in_channels[0], 3 * in_channels[0], 3, padding=1)
        self.conv5 = nn.Conv2d(3 * in_channels[0], 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


if __name__ == '__main__':
    import time

    dummy_x1 = torch.rand(1, 768, 28, 28)
    dummy_x2 = torch.rand(1, 384, 14, 14)
    dummy_x3 = torch.rand(1, 192, 7, 7)
    t = time.time()
    model = NeighborConnectionDecoder([768, 384, 192])
    x = model(dummy_x3, dummy_x2, dummy_x1)
    print(time.time() - t)
    print(x.shape)
