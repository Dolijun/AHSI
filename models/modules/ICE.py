import torch
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, GELU, GroupNorm
import torch.nn.functional as F


class ICELayer(Module):
    def __init__(self, inter_channel: int = 64, nclasses: int = 21):
        super(ICELayer, self).__init__()
        self.inter_channel = inter_channel
        self.nclasses = nclasses
        self.ice45 = ICEBlock()
        self.ice23 = ICEBlock()
        self.seg_head = Conv2d(self.inter_channel * 2, self.nclasses, kernel_size=1, stride=1)
        self.edge_head = Conv2d(self.inter_channel * 2, 1, kernel_size=1, stride=1)
        self.FuseGFF2 = FuseGFF(in_channels=self.inter_channel, out_channels=self.inter_channel)
        self.FuseGFF3 = FuseGFF(in_channels=self.inter_channel, out_channels=self.inter_channel)
        self.FuseGFF4 = FuseGFF(in_channels=self.inter_channel, out_channels=self.inter_channel)
        self.FuseGFF5 = FuseGFF(in_channels=self.inter_channel, out_channels=self.inter_channel)

    def forward(self, side2, side3, side4, side5):
        side2gff, side3gff = self.ice23(side2, side3)
        side4gff, side5gff = self.ice45(side4, side5)
        side5gff = self.FuseGFF5(side5gff)
        side4gff = self.FuseGFF4(side4gff)
        side3gff = self.FuseGFF3(side3gff)
        side2gff = self.FuseGFF2(side2gff)
        side2gff_t = F.interpolate(side2gff, scale_factor=0.5, mode='bilinear', align_corners=True)
        seg = self.seg_head(torch.cat((side5gff, side4gff), dim=1))
        edge = self.edge_head(torch.cat((side3gff, side2gff_t), dim=1))

        return side2gff, side3gff, side4gff, side5gff, seg, edge


class ICEBlock(Module):
    def __init__(self):
        super(ICEBlock, self).__init__()

    def forward(self, low, high):
        size_low = low.size()[-2:]
        size_high = high.size()[-2:]

        g_low = torch.sigmoid(low)
        g_high = torch.sigmoid(high)

        gs_low = F.interpolate(g_low * low, size=size_high, mode='bilinear', align_corners=True)  # 64,1/4
        gs_high = F.interpolate(g_high * high, size=size_low, mode='bilinear', align_corners=True)  # 64,1/2

        low_gff = (1 + g_low) * low + (1 - g_low) * gs_high  # 64,1/2
        high_gff = (1 + g_high) * high + (1 - g_high) * gs_low  # 64,1/4

        return low_gff, high_gff


class FuseGFF(Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(FuseGFF, self).__init__()
        self.FG = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            GELU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GroupNorm(1, out_channels),
            GELU()
        )

    def forward(self, input):
        output = self.FG(input)
        return output
