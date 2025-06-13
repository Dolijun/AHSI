import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseGFF(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(FuseGFF, self).__init__()
        self.FG = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.FG(input)
        return output


class GroundTrans(nn.Module):
    def __init__(self, in_channels=64, return_map=False):
        super(GroundTrans, self).__init__()
        self.return_map = return_map
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, bias=True)
        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=True),
            nn.GroupNorm(1, self.in_channels)
        )
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, bias=True)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, bias=True)

    def forward(self, x_high, x_low):
        # x_high n,c,h/2,w/2
        # x_low n,c,h,w

        batch_size = x_low.size(0)
        g_x = self.g(x_high).view(batch_size, self.inter_channels, -1)  # n,c/2,hw/4
        g_x = g_x.permute(0, 2, 1)  # n,hw/4,c/2

        theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1)  # n,c/2,hw
        theta_x = theta_x.permute(0, 2, 1)  # n,hw,c/2
        phi_x = self.phi(x_high).view(batch_size, self.inter_channels, -1)  # n,c/2,hw/4
        f = torch.matmul(theta_x, phi_x)  # n,hw,hw/4

        N = f.size(-1)  # hw/4
        f_div_c = f / N  # n,hw,hw/4

        y = torch.matmul(f_div_c, g_x)  # n,hw,c/2
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, x_low.size(2), x_low.size(3))  # n,c/2,h,w

        z = self.W_z(y)  # n,c,h,w
        # z = z + x_low  # n,c,h,w

        if self.return_map:
            return [z, f.cpu().detach().numpy()]

        return z


class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=21):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num

    def forward(self, feats, probs):
        probs = F.interpolate(probs, size=feats.shape[-2:], mode='bilinear', align_corners=True)  # b,21,h/2,w/2

        b, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(b, c, -1)
        feats = feats.view(b, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # b*hw/4*64
        probs = F.softmax(probs, dim=2)  # b*21*hw/4
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # b*64*21*1
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(ObjectAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        # x 64*h/2*w/2
        # proxy 64*21*1
        b, h, w = x.size(0), x.size(2), x.size(3)
        query = self.f_pixel(x).view(b, self.key_channels, -1)  # b*32*hw/4
        query = query.permute(0, 2, 1)  # b*hw/4*32
        key = self.f_object(proxy).view(b, self.key_channels, -1)  # b*32*21
        value = self.f_down(proxy).view(b, self.key_channels, -1)
        value = value.permute(0, 2, 1)  # b*21*32

        sim_map = torch.matmul(query, key)  # b*hw/4*21
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)  # b*hw/4*32
        context = context.permute(0, 2, 1).contiguous()  # b*32*hw/4
        context = context.view(b, self.key_channels, *x.size()[2:])  # b*32*h/2*w/2
        context = self.f_up(context)  # b*64*h/2*w/2

        return context


class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels)

        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        # feats 64*h/2*w/2
        # proxy_feats 64*21*1
        context = self.object_context_block(feats, proxy_feats)  # b*64*h/2*w/2

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))  # b*64*h/2*w/2

        return output


class Ocr(nn.Module):
    def __init__(self, in_channels=64, num_class=21):
        super(Ocr, self).__init__()
        # self.conv3x3_ocr = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.GroupNorm(1, in_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.ocr_gather_head = SpatialGather_Module(cls_num=num_class)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=in_channels,
                                                 key_channels=int(in_channels / 2),
                                                 out_channels=in_channels,
                                                 dropout=0.05
                                                 )

    def forward(self, feats, out_aux):
        # feats 64*h/2*w/2
        # out_aux 21*h*w
        # feats = self.conv3x3_ocr(feats)  # 64*h/2*w/2
        context = self.ocr_gather_head(feats, out_aux)  # 64*21*1
        feats = self.ocr_distri_head(feats, context)  # 512*h/2*w/2

        return feats


class Top_Down(nn.Module):
    def __init__(self, nclasses=20, feat_chans=None, inter_channel=64):

        super(Top_Down, self).__init__()
        if feat_chans is None:
            feat_chans = [256, 512, 1024, 2048]
        self.nclasses = nclasses
        self.inter_channels = inter_channel
        self.down5 = nn.Conv2d(feat_chans[-1], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down4 = nn.Conv2d(feat_chans[-2], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down3 = nn.Conv2d(feat_chans[-3], self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.down2 = nn.Conv2d(feat_chans[-4], self.inter_channels, kernel_size=1, stride=1, bias=False)

        self.FuseGFF2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF3 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF4 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF5 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)

        self.FuseGFF2_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF3_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF4_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)
        self.FuseGFF5_2 = FuseGFF(in_channels=self.inter_channels, out_channels=self.inter_channels)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_chans[-1], self.inter_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(1, self.inter_channels),
            nn.ReLU(inplace=True)
        )

        self.tg5 = GroundTrans(in_channels=self.inter_channels)
        # self.tg4 = GroundTrans()
        self.t54 = GroundTrans(in_channels=self.inter_channels)
        self.tg3 = GroundTrans(in_channels=self.inter_channels)
        self.t53 = GroundTrans(in_channels=self.inter_channels)
        self.t43 = GroundTrans(in_channels=self.inter_channels)
        self.tg2 = GroundTrans(in_channels=self.inter_channels)
        # self.t52 = GroundTrans()
        # self.t42 = GroundTrans()
        # self.t32 = GroundTrans()
        # return_map = True

        self.c5 = nn.Conv2d(self.inter_channels * 2, self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.c4 = nn.Conv2d(self.inter_channels * 2, self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.c3 = nn.Conv2d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, bias=False)
        self.c2 = nn.Conv2d(self.inter_channels * 2, self.inter_channels, kernel_size=1, stride=1, bias=False)

        self.head = nn.Conv2d(self.inter_channels, self.nclasses, kernel_size=1, stride=1)
        if nclasses == 20:
            self.seg_head = nn.Conv2d(self.inter_channels * 2, self.nclasses + 1, kernel_size=1, stride=1)
            self.ocr = Ocr(in_channels=self.inter_channels, num_class=self.nclasses + 1)
        else:
            self.seg_head = nn.Conv2d(self.inter_channels * 2, self.nclasses, kernel_size=1, stride=1)
            self.ocr = Ocr(in_channels=self.inter_channels, num_class=self.nclasses)
        self.edge_head = nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, res1, res2, res3, res4, res5):
        n, c, h, w = input.shape
        side2 = self.down2(res2)  # 64,1/2
        side3 = self.down3(res3)  # 64,1/4
        side4 = self.down4(res4)  # 64,1/8
        side5 = self.down5(res5)  # 64,1/8

        g2 = torch.sigmoid(side2)
        g3 = torch.sigmoid(side3)
        g4 = torch.sigmoid(side4)
        g5 = torch.sigmoid(side5)

        gs2 = F.interpolate(g2 * side2, size=side3.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/4
        gs3 = F.interpolate(g3 * side3, size=side2.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/2
        gs4 = F.interpolate(g4 * side4, size=side5.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/8
        gs5 = F.interpolate(g5 * side5, size=side4.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/8

        side5gff = (1 + g5) * side5 + (1 - g5) * gs4  # 64,1/8
        side4gff = (1 + g4) * side4 + (1 - g4) * gs5  # 64,1/8
        side3gff = (1 + g3) * side3 + (1 - g3) * gs2  # 64,1/4
        side2gff = (1 + g2) * side2 + (1 - g2) * gs3  # 64,1/2

        side5gff = self.FuseGFF5(side5gff)  # 64,1/8
        side4gff = self.FuseGFF4(side4gff)  # 64,1/8
        side3gff = self.FuseGFF3(side3gff)  # 64,1/4
        side2gff = self.FuseGFF2(side2gff)  # 64,1/2

        side5gff = F.interpolate(side5gff, size=side4gff.shape[-2:], mode='bilinear', align_corners=True)
        seg = torch.cat([side5gff, side4gff], dim=1)
        seg = self.seg_head(seg)
        # seg = F.interpolate(seg, scale_factor=4, mode='bilinear', align_corners=True)  # 21,1/2
        # seg = F.interpolate(seg, scale_factor=8, mode='bilinear', align_corners=True)  # 21,1
        seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=True)
        side2gff_t = F.interpolate(side2gff, scale_factor=0.5, mode='bilinear', align_corners=True)
        edge = torch.cat([side3gff, side2gff_t], dim=1)
        edge = self.edge_head(edge)
        # edge = F.interpolate(edge, scale_factor=2, mode='bilinear', align_corners=True)  # 1,1/2
        # edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=True)  # 1,1
        edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=True)
        side5gff_2 = self.FuseGFF5_2(side5gff)  # 64,1/8
        side4gff_2 = self.FuseGFF4_2(side4gff)  # 64,1/8
        side3gff_2 = self.FuseGFF3_2(side3gff)  # 64,1/4
        side2gff_2 = self.FuseGFF2_2(side2gff)  # 64,1/2

        global_context = self.global_context(res5)  # 64,1*1
        global_context = F.interpolate(global_context, size=side5gff_2.size()[2:], mode='bilinear',
                                       align_corners=True)  # 64,1/8

        # hm = self.t53(side5gff_2, side3gff_2)[1]
        # return {'t53': hm}

        s5 = torch.cat((self.tg5(global_context, side5gff_2),
                        side5gff_2), 1)
        s5 = self.c5(s5)  # 64,1/8

        # s4 = torch.cat((self.tg4(global_context, side4gff_2), self.t54(side5gff_2, side4gff_2),
        #                 side4gff_2), 1)
        s4 = torch.cat((self.t54(side5gff_2, side4gff_2), side4gff_2), 1)
        s4 = self.c4(s4)  # 64,1/8

        s3 = torch.cat(
            (self.tg3(global_context, side3gff_2), self.t53(side5gff_2, side3gff_2), self.t43(side4gff_2, side3gff_2),
             side3gff_2), 1)
        s3 = self.c3(s3)  # 64,1/4

        # s2 = torch.cat(
        #     (self.tg2(global_context, side2gff_2), self.t52(side5gff_2, side2gff_2), self.t42(side4gff_2, side2gff_2),
        #      self.t32(side3gff_2, side2gff_2), side2gff_2), 1)
        s2 = torch.cat(
            (self.tg2(global_context, side2gff_2), side2gff_2), 1)
        s2 = self.c2(s2)  # 64,1/2

        sum45 = s4 + F.interpolate(s5, size=s4.shape[-2:], mode='bilinear', align_corners=True)  # 64,1/8
        sum34 = s3 + F.interpolate(sum45, scale_factor=2, mode='bilinear', align_corners=True)  # 64,1/4
        sum23 = s2 + F.interpolate(sum34, scale_factor=2, mode='bilinear', align_corners=True)  # 64,1/2

        sum_23 = self.ocr(sum23, seg)  # 64*h/2*w/2
        # 3, 64
        # 3, 64
        final_feature = self.head(sum_23)  # 20,1/2
        # sedge = F.interpolate(final_feature, scale_factor=2, mode='bilinear', align_corners=True)  # 20,1
        sedge = F.interpolate(final_feature, size=(h, w), mode='bilinear', align_corners=True)

        return sedge, seg, edge
