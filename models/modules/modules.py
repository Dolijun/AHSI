# coding:utf-8
import torch.nn as nn
import torch
import numpy as np

BatchNorm = nn.BatchNorm2d


# 裁剪
class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, indices)
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x


# 底部特征抽取
class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, out_channels=1, kernel_size=kernel_sz, stride=stride,
                                                padding=upconv_pad, bias=False)
            ##doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


# 顶部特征抽取
class Res5OutputCrop(nn.Module):

    def __init__(self, in_channels=2048, kernel_sz=16, stride=8, nclasses=20, upconv_pad=0, do_crops=True):
        super(Res5OutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(in_channels, nclasses, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsampled = nn.ConvTranspose2d(nclasses, out_channels=nclasses, kernel_size=kernel_sz, stride=stride,
                                            padding=upconv_pad, bias=False, groups=nclasses)
        if self._do_crops is True:
            self.crops = Crop(2, offset=kernel_sz // 4)
        else:
            self.crops = MyIdentity(None, None)

    def forward(self, res, reference):
        res = self.conv(res)
        res = self.upsampled(res)
        res = self.crops(res, reference)
        return res


# bilinear kernel
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class UpsampleCrop(nn.Module):

    def __init__(self, num_input, num_output, kernel_sz=4, stride=2, upconv_pad=0, do_crops=True):
        super(UpsampleCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_input, num_output, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsampled = nn.ConvTranspose2d(num_output, out_channels=num_output, kernel_size=kernel_sz, stride=stride,
                                            padding=upconv_pad, bias=False, groups=num_output)
        if self._do_crops is True:
            self.crops = Crop(2, offset=kernel_sz // 4)
        else:
            self.crops = MyIdentity(None, None)

    def forward(self, res, reference):
        res = self.conv(res)
        res = self.upsampled(res)
        res = self.crops(res, reference)
        return res


# CRM
class RefineResidual(nn.Module):
    def __init__(self, num_input, num_output, rate1=1, rate2=1):
        super(RefineResidual, self).__init__()
        self.conv1x1 = nn.Conv2d(num_input, num_output, kernel_size=1, bias=False)
        self.CRM = nn.Sequential(
            nn.Conv2d(num_output, num_output, kernel_size=3, stride=1, padding=rate1, dilation=rate1, bias=False),
            # BatchNorm(num_output),
            nn.GroupNorm(1, num_output),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_output, num_output, kernel_size=3, stride=1, padding=rate2, dilation=rate2, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        residual = x
        x = self.CRM(x)
        return self.relu(x + residual)


# LAM
class LocationAware(nn.Module):
    def __init__(self, channels):
        super(LocationAware, self).__init__()
        self.LAM = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, low_feature, high_feature):
        cat_feature = torch.cat((low_feature, high_feature), 1)
        weight = self.LAM(cat_feature)
        low_feature = weight * low_feature
        return low_feature + high_feature


'''
# LAM
class LocationAware(nn.Module):
    def __init__(self, channels):
        super(LocationAware, self).__init__()
        self.LAM = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
            nn.Sigmoid()
        )

    def forward(self, low_feature, high_feature):
        cat_feature = torch.cat((low_feature, high_feature), 1)
        weight = self.LAM(cat_feature)
        low_feature = weight * low_feature
        return low_feature + high_feature
'''

'''
# LAM
class LocationAware(nn.Module):
    def __init__(self, channels):
        super(LocationAware, self).__init__()
        self.LAM = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, low_feature, high_feature):
        weight = self.LAM(high_feature)
        low_feature = weight * low_feature
        return low_feature + high_feature
'''
