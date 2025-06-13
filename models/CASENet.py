# coding:utf-8
import torch
import torch.nn as nn
import math

from .modules import SideOutputCrop, Res5OutputCrop, get_upsample_filter

BatchNorm = nn.BatchNorm2d


class CASENet(nn.Module):
    def __init__(self, nclasses=20, feat_chans=None):
        super(CASENet, self).__init__()
        self.nclasses = nclasses
        self.feat_chans = feat_chans
        ####let's make pointers to keep compatibility for now
        SideOutput_fn = SideOutputCrop
        Res5Output_fn = Res5OutputCrop

        # The original casenet implementation has padding when upsampling cityscapes that are not used for SBD.
        # Leaving like this for now such that it is clear and it matches the original implementation (for fair comparison)

        if nclasses == 19:
            print('Assuming Cityscapes CASENET')
            self.score_edge_side1 = SideOutput_fn(feat_chans[0])
            self.score_edge_side2 = SideOutput_fn(feat_chans[0], kernel_sz=4, stride=2, upconv_pad=1, do_crops=False)
            self.score_edge_side3 = SideOutput_fn(feat_chans[1], kernel_sz=8, stride=4, upconv_pad=2, do_crops=False)
            self.score_cls_side5 = Res5Output_fn(in_channels=feat_chans[3], kernel_sz=16, stride=8, nclasses=nclasses, upconv_pad=4,
                                                 do_crops=False)
        else:
            print('Assuming Classical SBD CASENET')
            self.score_edge_side1 = SideOutput_fn(feat_chans[0])
            self.score_edge_side2 = SideOutput_fn(feat_chans[0], kernel_sz=4, stride=2)
            self.score_edge_side3 = SideOutput_fn(feat_chans[1], kernel_sz=8, stride=4)
            self.score_cls_side5 = Res5Output_fn(in_channels=feat_chans[3], kernel_sz=16, stride=8, nclasses=nclasses)

        self.ce_fusion = nn.Conv2d(4 * nclasses, nclasses, groups=nclasses, kernel_size=1, stride=1, padding=0,
                                   bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # manually initializing the new layers.
        self.score_edge_side1.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side1.conv.bias.data.zero_()
        # -
        self.score_edge_side2.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side2.conv.bias.data.zero_()
        # -
        self.score_edge_side3.conv.weight.data.normal_(0, 0.01)
        self.score_edge_side3.conv.bias.data.zero_()
        # -
        self.ce_fusion.weight.data.fill_(0.25)
        self.ce_fusion.bias.data.zero_()

    def _sliced_concat(self, res1, res2, res3, res5):
        out_dim = self.nclasses * 4
        out_tensor = torch.FloatTensor(res1.size(0), out_dim, res1.size(2), res1.size(3)).cuda()
        class_num = 0
        for i in range(0, out_dim, 4):
            out_tensor[:, i, :, :] = res5[:, class_num, :, :]
            out_tensor[:, i + 1, :, :] = res1[:, 0, :, :]  # it needs this trick for multibatch
            out_tensor[:, i + 2, :, :] = res2[:, 0, :, :]
            out_tensor[:, i + 3, :, :] = res3[:, 0, :, :]

            class_num = class_num + 1

        return out_tensor

    def forward(self, input, res1, res2, res3, res4, res5):
        if res1 is None:
            res1 = torch.nn.functional.interpolate(res2, scale_factor=2, mode='bilinear')
            res5 = torch.nn.functional.interpolate(res5, scale_factor=2, mode='bilinear')
        side_1 = self.score_edge_side1(res1, input)
        side_2 = self.score_edge_side2(res2, input)
        side_3 = self.score_edge_side3(res3, input)
        side_5 = self.score_cls_side5(res5, input)

        # combine outputs and classify
        sliced_cat = self._sliced_concat(side_1, side_2, side_3, side_5)
        acts = self.ce_fusion(sliced_cat)

        return side_5, acts
