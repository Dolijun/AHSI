# coding:utf-8
'''
自底向上
'''
import torch.nn as nn
import torch.nn.functional

from .modules import Res5OutputCrop, get_upsample_filter, LocationAware, RefineResidual

BatchNorm = nn.BatchNorm2d


class CRM_LAM(nn.Module):
    def __init__(self, nclasses=20, feat_chans=None):
        super(CRM_LAM, self).__init__()
        self.nclasses = nclasses

        Res5Output_fn = Res5OutputCrop

        self.downsample1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=False)
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.downsample3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.downsample4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)

        if nclasses == 19:
            print('Assuming Cityscapes CRM_LAM')
            self.score_cls_side5 = Res5Output_fn(in_channels=512, kernel_sz=16, stride=8, nclasses=nclasses,
                                                 upconv_pad=4, do_crops=False)
        else:
            print('Assuming Classical SBD CRM_LAM')
            self.score_cls_side5 = Res5Output_fn(in_channels=512, kernel_sz=16, stride=8, nclasses=nclasses)

        self.CRM1a = RefineResidual(num_input=feat_chans[0], num_output=32, rate1=1, rate2=1)
        self.CRM2a = RefineResidual(num_input=feat_chans[0], num_output=64, rate1=1, rate2=1)
        self.CRM3a = RefineResidual(num_input=feat_chans[1], num_output=128, rate1=1, rate2=1)
        self.CRM4a = RefineResidual(num_input=feat_chans[2], num_output=256, rate1=1, rate2=1)
        self.CRM5a = RefineResidual(num_input=feat_chans[3], num_output=512, rate1=1, rate2=1)

        self.CRM2b = RefineResidual(num_input=64, num_output=64, rate1=1, rate2=1)
        self.CRM3b = RefineResidual(num_input=128, num_output=128, rate1=1, rate2=1)
        self.CRM4b = RefineResidual(num_input=256, num_output=256, rate1=1, rate2=1)
        self.CRM5b = RefineResidual(num_input=512, num_output=512, rate1=1, rate2=1)

        self.LAM2 = LocationAware(channels=64)
        self.LAM3 = LocationAware(channels=128)
        self.LAM4 = LocationAware(channels=256)
        self.LAM5 = LocationAware(channels=512)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, res1, res2, res3, res4, res5):
        if res1 is None:
            res1 = torch.nn.functional.interpolate(res2, scale_factor=2, mode='bilinear')
        side1 = self.CRM1a(res1)
        side2 = self.CRM2a(res2)
        side3 = self.CRM3a(res3)
        side4 = self.CRM4a(res4)
        side5 = self.CRM5a(res5)

        fused2 = self.LAM2(self.downsample1(side1), side2)
        x = self.CRM2b(fused2)
        fused3 = self.LAM3(self.downsample2(x), side3)
        x = self.CRM3b(fused3)
        fused4 = self.LAM4(self.downsample3(x), side4)
        x = self.CRM4b(fused4)
        side5 = torch.nn.functional.interpolate(side5, scale_factor=2, mode='bilinear')
        fused5 = self.LAM5(self.downsample4(x), side5)
        x = self.CRM5b(fused5)
        score_5 = self.score_cls_side5(x, input)

        return score_5
