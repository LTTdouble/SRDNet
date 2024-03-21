import torch
import torch.nn as nn
from architecture import common
import torch.nn as nn
import torch.nn.functional as F
from option import opt

class IGM(nn.Module):
   
    def __init__(self, in_channels, out_channels, gropus=1):
        super(IGM, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        features1 = 32
        distill_rate = 0.5
        self.distilled_channels = int(features * distill_rate)
        self.remaining_channels = int(features - self.distilled_channels)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features1, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features1, kernel_size=3, padding=1, groups=1,
                      bias=False), nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features1, kernel_size=3, padding=1, groups=1,
                      bias=False))

        self.conv1_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1,
                      bias=False), nn.ReLU(inplace=True))
        self.conv3_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1,
                      bias=False), nn.ReLU(inplace=True))
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        # self.conv7_1 = nn.Sequential(
           #  nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
            #           bias=False))

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input):
        dit1, remain1 = torch.split(input, (self.distilled_channels, self.remaining_channels), dim=1)
        out1_1 = self.conv1_1(dit1)
        out1_1_t = self.ReLU(out1_1)
        out2_1 = self.conv1_1(out1_1_t)
        out3_1 = self.conv3_1(out2_1)

        out1_2 = self.conv1_1(remain1)
        out1_2_t = self.ReLU(out1_2)
        out2_2 = self.conv2_1(out1_2_t)
        out3_2 = self.conv3_1(out2_2)

        out3_t = torch.cat([out3_1, out3_2], dim=1)
        out3 = self.ReLU(out3_t)

        out1_1t = self.conv1_1_1(input)
        out1_2t1 = self.conv2_1_1(out1_1t)
        out1_3t1 = self.conv3_1_1(out1_2t1)

        out1_3t1 = out3 + out1_3t1 + input

        out4_1 = self.conv4_1(out1_3t1)
        out5_1 = self.conv5_1(out4_1)
        out6_1 = self.conv6_1(out5_1)
        # out7_1 = self.conv7_1(out6_1)

        out6_1 = out6_1 + input + out1_3t1

        return out6_1


def make_model(opt, parent=False):
    return SRDNet(opt)


class SRDNet(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(SRDNet, self).__init__()

        n_feats = opt.n_feats
        kernel_size = 3
        scale = opt.upscale_factor
        act = nn.ReLU(True)

        m_head = [conv(opt.n_colors, n_feats, kernel_size)]
        self.nearest_g = nn.Upsample(scale_factor=scale, mode='bicubic')

        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=opt.res_scale
            ) for _ in range(1)
        ]

        m_body1_1 = [(conv(n_feats, opt.n_colors, kernel_size))]

        m_body2 = [IGM(n_feats, n_feats)]

        m_body3 = [(common.HSL(inplanes=n_feats, planes=n_feats))]

        m_body4 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=opt.res_scale
            ) for _ in range(3)
        ]
        m_body4.append(conv(n_feats, n_feats, kernel_size))
        end = [nn.Conv2d(opt.n_colors, n_feats, kernel_size=3, stride=1, padding=1)]

        m_tail = []
        if scale == 3:
            m_tail.append(
                nn.ConvTranspose2d(n_feats, n_feats, kernel_size=(2 + scale, 2 + scale), stride=(scale, scale),
                                   padding=(1, 1)))
            self.nearest_l = nn.Upsample(scale_factor=scale, mode='bicubic')
        else:
            m_tail.append(
                nn.ConvTranspose2d(n_feats, n_feats, kernel_size=(2 + 2, 2 + 2), stride=(2, 2),
                                   padding=(1, 1)))
            self.nearest_l = nn.Upsample(scale_factor=2, mode='bicubic')
        m_tail.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=(1, 1)))

        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.body1_1 = nn.Sequential(*m_body1_1)
        self.body2 = nn.Sequential(*m_body2)
        self.body3 = nn.Sequential(*m_body3)
        self.body4 = nn.Sequential(*m_body4)
        self.tail = nn.Sequential(*m_tail)
        self.end = nn.Sequential(*end)

        end_1 = []
        end_1.append(nn.Conv3d( n_feats,1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)))
        self.end_1 = nn.Sequential(*end_1)

        head_3D = []
        head_3D.append(nn.Conv3d(1, n_feats, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)))
        self.head_3D = nn.Sequential(*head_3D)

        basic_3D = [
            common.TreeDBlock(
                cin=1 * n_feats, cout=n_feats * 1, use_relu=True, fea_num=1
            ) for _ in range(3)
        ]
        m_tail_3D = []

        self.gamma = nn.Parameter(torch.ones(3))
        self.basic_3D = nn.Sequential(*basic_3D)
        self.reduceD = nn.Conv3d(n_feats * 3, n_feats, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        if scale == 3:
            m_tail_3D.append(
                nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                   padding=(1, 1, 1)))
        else:
            m_tail_3D.append(
                nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + 2, 2 + 2), stride=(1, 2, 2),
                                   padding=(1, 1, 1)))
        m_tail_3D.append(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        m_tail_3D.append(nn.Conv3d(n_feats, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        self.tail_3D = nn.Sequential(*m_tail_3D)

        m_tail_g = []
        if scale == 3:
            m_tail_g.append(
                nn.ConvTranspose2d(n_feats, n_feats, kernel_size=(2 + 1, 2 + 1), stride=(1, 1), padding=(1, 1)))

        else:
            m_tail_g.append(
                nn.ConvTranspose2d(n_feats, n_feats, kernel_size=(2 + scale // 2, 2 + scale // 2),
                                   stride=(scale // 2, scale // 2),
                                   padding=(1, 1)))
        m_tail_g.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        m_tail_g.append(nn.Conv2d(n_feats, opt.n_colors, kernel_size=3, stride=1, padding=1))
        self.tail_g = nn.Sequential(*m_tail_g)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        CSKC = self.nearest_g(x)
        x1 = self.head(x)
        res1 = self.body1(x1)
        res2 = self.body2(res1)
        res1_1 = self.body1_1(res1 + res2) + x
        res1_3D = res1_1.unsqueeze(1)
        res1_3D = self.head_3D(res1_3D)
        H = []

        for i in range(3):
            res1_3D = self.basic_3D[i](res1_3D, opt.nEpochs)
            res1_2D = self.end_1(res1_3D)
            res1_2D = res1_2D.squeeze(1)

            H.append(res1_3D * self.gamma[i])

        res1_3D = torch.cat(H, 1)
        res1_3D = self.reduceD(res1_3D)
        res3 = self.body3(res2 + self.head(res1_2D))
        res4 = self.tail(res3)

        res4_3D = self.tail_3D(res1_3D)
        res4_3D = res4_3D.squeeze(1)
        res4_3D = self.end(res4_3D)

        x4 = res4 + res4_3D + self.nearest_l(res1)
        x4 = self.body4(x4)
        x4 = self.tail_g(x4)
        x4 = x4 + CSKC

        return x4
