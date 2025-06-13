# coding:utf-8
'''
Resnet101
'''
import torch.nn as nn
import math

BatchNorm = nn.BatchNorm2d


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, block_no, stride=1, downsample=None):  # add dilation factor
        super(Bottleneck, self).__init__()
        if block_no < 5:
            dilation = 2
            padding = 2
        else:
            dilation = 4
            padding = 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # define ceil mode
        self.layer1 = self._make_layer(64, 3, 2)  # res2
        self.layer2 = self._make_layer(128, 4, 3, stride=2)  # res3
        self.layer3 = self._make_layer(256, 23, 4, stride=2)  # res4
        self.layer4 = self._make_layer(512, 3, 5, stride=1)  # res5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, block_no, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * 4),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, block_no, stride, downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, block_no))

        return nn.Sequential(*layers)

    def forward(self, x):
        # res1
        res1 = self.conv1(x)
        res1 = self.bn1(res1)
        res1 = self.relu(res1)
        # res2
        res2 = self.maxpool(res1)
        res2 = self.layer1(res2)
        # res3
        res3 = self.layer2(res2)
        # res4
        res4 = self.layer3(res3)
        # res5
        res5 = self.layer4(res4)

        return res1, res2, res3, res4, res5

if __name__ == '__main__':
    import torch
    resnet = ResNet101()
    input = torch.randn((1, 3, 512, 512))

    outputs = resnet(input)

    for idx, res in enumerate(outputs):
        print(f"res{idx + 1}: {res.shape}")

