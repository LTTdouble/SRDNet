import math

import torch
import torch.nn as nn
from option import opt
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return wn(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias))


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False ,act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0), use_relu=True):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TreeDBlock(nn.Module):
    def __init__(self, cin, cout, use_relu, fea_num):
        super(TreeDBlock, self).__init__()

        self.spatiallayer = nn.Conv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                        bias=False)
        self.spectralayer = nn.Conv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                                        bias=False)
        self.Conv_mixdence = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                        use_relu=use_relu)
        self.relu = nn.ReLU(inplace=True)

        self.fea_num = fea_num

        self.component_num = cout
        self.feat_weight = nn.Parameter(torch.rand(fea_num * 64), requires_grad=True)  # [N,*,c,u,v,h,w]
        self.component_weight1 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.component_weight2 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.temperature_1 = 0.2
        self.temperature_2 = 0.2

    def forward(self, x, epoch):

        t1 = 1.0
        t2 = 1.0
        if epoch <= 30:  # T  1 ==> 0.1
            self.temperature_1 = t1 * (1 - epoch / 35)
            self.temperature_2 = t2 * (1 - epoch / 35)
        else:
            self.temperature_1 = 0.05
            self.temperature_2 = 0.05

        if (self.fea_num > 1):
            x = torch.cat(x, dim=1)  # [fea_num B C H W]

            [B, L, C, H, W] = x.shape

            feat_weight = self.feat_weight.clamp(0.02, 0.98)
            feat_weight = feat_weight[None, :, None, None, None]
            # p shape[fea_num 1 1 1 1]
            # noise r1 r2
            noise_feat_r1 = torch.rand((B, self.fea_num * 64))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
            noise_feat_r2 = torch.rand((B, self.fea_num * 64))[:, :, None, None, None].cuda()
            noise_feat_logits = torch.log(torch.log(noise_feat_r1) / torch.log(noise_feat_r2))
            feat_weight_soft = torch.sigmoid(
                (torch.log(feat_weight / (1 - feat_weight)) + noise_feat_logits) / self.temperature_1)
            feat_logits = feat_weight_soft

            x = x * feat_logits
        # else:
        #     x = torch.cat(x, 1)

        # # SELECT NETWOKR
        component_weight1 = self.component_weight1.clamp(0.02, 0.98)
        component_weight1 = component_weight1[None, :, None, None, None]
        component_weight2 = self.component_weight2.clamp(0.02, 0.98)
        component_weight2 = component_weight2[None, :, None, None, None]

        [B, L, C, H, W] = x.shape

        # s2
        noise_component_r1 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r2 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits1 = torch.log(torch.log(noise_component_r1) / torch.log(noise_component_r2))
        component_weight_gumbel1 = torch.sigmoid(
            (torch.log(component_weight1 / (1 - component_weight1)) + noise_component_logits1) / self.temperature_2)
        logits2 = component_weight_gumbel1

        # s3
        noise_component_r3 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r4 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits2 = torch.log(torch.log(noise_component_r3) / torch.log(noise_component_r4))
        component_weight_gumbel2 = torch.sigmoid(
            (torch.log(component_weight2 / (1 - component_weight2)) + noise_component_logits2) / self.temperature_2)
        logits3 = component_weight_gumbel2

        output = self.relu(self.Conv_mixdence(x))

        output = self.spectralayer(output) * logits2 + output
        output = self.spatiallayer(output) * logits3 + output+x

        return output

def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class HSL(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(HSL, self).__init__()
        # spectral attention
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.conv_x = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                         bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax_left = nn.Softmax(dim=2)

        # spatial attention (after fusion conv)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn = nn.Conv2d(inplanes * 2, inplanes, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        # self.reset_parameters()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.ReLU = nn.ReLU(inplace=False)

        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True

        self.gamma_rnn = nn.Parameter(torch.ones(2))

    def forward(self,x):
        """
        Args:
                x (Tensor):Features with shape (b, c, h, w).

        Returns:
            Tensor: Features after HSL with the shape (b, c, h, w).
        # """
        #  spectral attention
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = (self.conv_q_right(x))
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = (self.conv_up(context))

        mask_ch = self.ReLU(self.sigmoid(context))
        x1 = self.ReLU(self.conv_x(x))
        out = x1 * mask_ch+x

        # fusion
        feat_fusion = self.conv_x(out)

        # spatial attention
        attn1 = self.ReLU(self.conv_x(out))
        attn_max = self.max_pool1(attn1)
        attn_avg = self.avg_pool1(attn1)
        attn2 = torch.cat([self.gamma_rnn[0] * attn_max, self.gamma_rnn[1] * attn_avg], 1)
        attn2 = self.spatial_attn(attn2)

        # pyramid levels
        attn_level = self.ReLU(self.conv_x(attn2))
        attn_max = self.max_pool1(attn_level)
        attn_avg = self.avg_pool1(attn_level)
        attn_level =self.spatial_attn(torch.cat([self.gamma_rnn[0] *attn_max, self.gamma_rnn[1] *attn_avg], 1))

        attn_level = self.ReLU(self.conv(attn_level))
        attn_level = self.upsample(attn_level)

        batch, channel, height, width = attn2.size()
        if opt.upscale_factor==3:
            attn3 =self.conv(attn2[:,:,:,:width])+ attn_level[:,:,:height,:width]
        else:
            attn3 = self.conv(attn2)+ attn_level
        attn3 = self.conv(attn3)
        attn3 = self.upsample(attn3)
        attn4 = attn3+attn1

        attn_add = self.conv(attn4)
        attn_out = torch.sigmoid(attn_add)
        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat2 =  feat_fusion * attn_out*2+attn_add+out
        return feat2

