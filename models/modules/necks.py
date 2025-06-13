import torch
from torch import nn
import torch.nn.functional as F


class LinearNeck(nn.Module):
    def __init__(self, feat_chans, out_chans=256, num_feats=4):
        super(LinearNeck, self).__init__()
        if isinstance(feat_chans, int):
            feat_chans = [feat_chans] * num_feats
        else:
            assert isinstance(feat_chans, list) or isinstance(feat_chans, tuple), "feat_chans format error"

        if isinstance(out_chans, int):
            out_chans = [out_chans] * num_feats
        else:
            assert isinstance(out_chans, list) or isinstance(out_chans, tuple), "out_chans format error"

        self.neck_layers = nn.ModuleList()
        self.neck_layers.extend([
            nn.Conv2d(feat_chans[0], out_chans[0], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(feat_chans[1], out_chans[1], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(feat_chans[2], out_chans[2], kernel_size=1, stride=1, bias=False),
            nn.Conv2d(feat_chans[3], out_chans[3], kernel_size=1, stride=1, bias=False),
        ])

    def forward(self, feats):
        feat_out = []
        for feat, neck_layer in zip(feats, self.neck_layers):
            feat_out.append(neck_layer(feat))
        return list(feat_out)


class UpSampleNeck(nn.Module):
    def __init__(self, feat_chans, out_chans=256, factors=2, num_feats=4):
        super(UpSampleNeck, self).__init__()
        if isinstance(feat_chans, int):
            feat_chans = [feat_chans] * num_feats
        else:
            assert isinstance(feat_chans, list) or isinstance(feat_chans, tuple), "feat_chans format error"

        if isinstance(out_chans, int):
            out_chans = [out_chans] * num_feats
        else:
            assert isinstance(out_chans, list) or isinstance(out_chans, tuple), "out_chans format error"

        if isinstance(factors, int):
            factors = [factors] * num_feats
        else:
            assert isinstance(factors, list) or isinstance(factors, tuple), "out_chans format error"

        self.upsample_factors = factors

        self.neck_layers = nn.ModuleList()
        self.neck_layers.extend([
            nn.Conv2d(feat_chans[0], out_chans[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(feat_chans[1], out_chans[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(feat_chans[2], out_chans[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(feat_chans[3], out_chans[3], kernel_size=3, stride=1, padding=1, bias=False),
        ])

    def forward(self, feats):
        feat_out = []
        for feat, neck_layer, factor in zip(feats, self.neck_layers, self.upsample_factors):
            feat = F.interpolate(feat, scale_factor=factor, mode='bilinear', align_corners=True)
            feat_out.append(neck_layer(feat))
        return list(feat_out)
