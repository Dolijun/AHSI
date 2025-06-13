import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *


class CrossFusionLayer(nn.Module):
    def __init__(self, in_chan: int = 256, feats_num: int = 4):
        super(CrossFusionLayer, self).__init__()
        self.inter_channel = in_chan
        self.feats_num = feats_num
        self.squeezes = nn.ModuleList()
        for i in range(feats_num):
            self.squeezes.append(nn.Sequential(
                nn.Linear(self.inter_channel * feats_num, self.inter_channel * 4, bias=True),
                nn.GELU(),
                nn.Linear(self.inter_channel * 4, self.inter_channel, bias=True),
                nn.Dropout(0.1)
            ))

    def forward(self, *feats):
        assert len(feats) == self.feats_num
        feats_out = []
        sizes = [feat.size()[-2:] for feat in feats]
        for idx, size in enumerate(sizes):
            feat = torch.cat([F.interpolate(feat, size=size, mode='bilinear', align_corners=True) for feat in feats], dim=1)
            feat = rearrange(feat, 'n c h w -> n h w c')
            feat = self.squeezes[idx](feat)
            feat = rearrange(feat, 'n h w c -> n c h w')
            feats_out.append(feat)

        return feats_out

class CrossFusion(nn.Module):
    def __init__(self, inter_channel, feats_num, factor=4):
        super(CrossFusion, self).__init__()

        self.squeeze = nn.Sequential(
            nn.Linear(inter_channel * feats_num, inter_channel * factor, bias=False),
            nn.LayerNorm(inter_channel * factor, eps=1e-6),
            nn.GELU(),
            nn.Linear(inter_channel * factor, inter_channel, bias=True),
            nn.Dropout(0.1)
        )


    def forward(self, x, *feats):
        feats_out = []
        skip = x
        size_ = x.size()[-2:]
        feats_out.append(x)
        for feat in feats:
            feats_out.append(F.interpolate(feat, size=size_, mode='bilinear', align_corners=True))
        feat = torch.cat(feats_out, dim=1)
        feat = rearrange(feat, 'n c h w -> n h w c')
        feat = self.squeeze(feat)
        feat = rearrange(feat, 'n h w c -> n c h w')
        feat = feat + x

        return feat



if __name__ == "__main__":
    input1 = torch.ones((1, 1, 3,3))
    input2 = torch.ones((1, 1, 6,6))
    input3 = torch.ones((1, 1, 9,9))
    fuse = CrossFusionLayer(1, 3)
    outs = fuse(input1, input2, input3)
    for out in outs:
        print(out.size())

