import torch
import torch.nn as nn
# from basicsr.models.archs import recons_video81 as recons_video
# from basicsr.models.archs import flow_pwc82 as flow_pwc
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.ops import DeformConv2d


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        reduction = 1
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        reduction = 1
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class RepConv(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv, self).__init__()
        self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size // 2, groups=n_feat)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)

    def forward(self, x):
        res_1 = self.conv_1(x)
        # return res_1
        res_2 = self.conv_2(x)
        return res_1 + res_2 + x


class RepConv2(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv2, self).__init__()
        # self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, bias=bias, padding=1, groups=n_feat)

    def forward(self, x):
        # res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_2 + x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class CAB1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB1, self).__init__()
        modules_body = []
        scale_factor = 1
        n_scale_feat = int(scale_factor * n_feat)
        self.norm = LayerNorm2d(n_feat)
        modules_body.append(conv(n_feat, n_scale_feat * 2, 1, bias=bias))
        modules_body.append(RepConv2(n_scale_feat * 2, kernel_size, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2 * n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))
        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = self.body(self.norm(x))
        # res = self.CA(res)
        res = x + res * self.beta
        return res


class CAB2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, add_channel=0):
        super(CAB2, self).__init__()
        modules_body = []
        scale_factor = 1
        self.n_feat = n_feat
        self.add_channel = add_channel
        n_scale_feat = int(scale_factor * n_feat)
        self.conv1 = nn.Conv2d(self.add_channel, self.add_channel, 3, bias=bias, padding=1, groups=self.add_channel)
        self.norm = LayerNorm2d(self.add_channel + n_feat)
        modules_body.append(conv(n_feat + self.add_channel, n_scale_feat * 2, 1, bias=bias))
        modules_body.append(RepConv2(n_scale_feat * 2, kernel_size, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_scale_feat, kernel_size, bias))
        modules_body.append(conv(n_scale_feat, 2 * n_scale_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer2(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_scale_feat, n_feat, 1, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x_input):
        shortcut, hw = x_input[:, 0:self.n_feat], x_input[:, self.n_feat:]
        hw = self.conv1(hw)
        res = self.body(self.norm(torch.cat((shortcut, hw), dim=1)))
        # res = self.CA(res)
        res = shortcut + res * self.beta
        return res


class PixelShufflePack(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        # self.init_weights()

    def init_weights(self):
        default_init_weights(self, 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x



class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = self.down(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class KGS_block(nn.Module):
    def __init__(self, n_features, kernel_size, reduction, bias=False, scale_unetfeats=48):
        super(KGS_block, self).__init__()
        n_feat = n_features
        scale_unetfeats = int(n_feat / 2)
        act = nn.PReLU()
        number = n_feat // 2 // 8
        self.number = number
        self.encoder_level1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8 * self.number),
                               CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_1 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8 * self.number),
                                 CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_2 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8 * self.number),
                                 CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1_3 = [CAB2(n_feat, 5, reduction, bias=bias, act=act, add_channel=8 * self.number),
                                 CAB1(n_feat, 5, reduction, bias=bias, act=act)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level1_1 = nn.Sequential(*self.encoder_level1_1)
        self.encoder_level1_2 = nn.Sequential(*self.encoder_level1_2)
        self.encoder_level1_3 = nn.Sequential(*self.encoder_level1_3)

        kernel_size = 3
        padding = kernel_size // 2
        self.encoder_ker1 = CAB1(n_feat, 3, reduction, bias=bias, act=act)

        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, stride=1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(n_feat, 8 * 2, 1, stride=1, padding=0, bias=True)
        )

    def spatial_shift2(self, x, offset):
        """
        Args:
            x: (B, 8*C, H, W) feature map
            offset: (B, 8*2, H, W) offsets in pixels (dx, dy) for each channel
        Returns:
            warped feature map (B, 8*C, H, W)
        """
        B, C, H, W = x.shape
        assert C % 8 == 0, "Channel count must be divisible by 8"
        C_per_shift = C // 8

        # base grid 생성: torch.linspace로 x와 같은 dtype/device로 생성
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H - 1, H, dtype=x.dtype, device=x.device),
            torch.linspace(0, W - 1, W, dtype=x.dtype, device=x.device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 1, H, W)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 1, H, W)
        base_grid = torch.cat((grid_x, grid_y), dim=1)  # (B, 2, H, W)

        # reshape offset: (B, 8*2, H, W) -> (B, 8, 2, H, W)
        offset = offset.view(B, 8, 2, H, W)
        vgrid = base_grid.view(B, 1, 2, H, W) + offset

        vgrid[:, :, 0, :, :] = 2.0 * vgrid[:, :, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, :, 1, :, :] = 2.0 * vgrid[:, :, 1, :, :].clone() / max(H - 1, 1) - 1.0

        # split x into 8 chunks
        x_split = x.view(B, 8, C_per_shift, H, W)

        # apply grid_sample for each shifted group
        out = []
        for i in range(8):
            warped = F.grid_sample(
                x_split[:, i],  # (B, C_per_shift, H, W)
                vgrid[:, i].permute(0, 2, 3, 1),  # (B, H, W, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            out.append(warped)

        # concatenate along channel dimension
        out = torch.cat(out, dim=1)  # (B, 8*C_per_shift, H, W)
        return out


    def channel_shift(self, x, offset, div=2, step=0):
        B, C, H, W = x.shape

        y = x[:, 0:8 * self.number, ...]


        if step == 0: # shift 0
            hw = x
            offset = offset
        elif step == 1: # shift +1
            hw = x[:, -8 * self.number:, ...]
            hw = torch.cat((hw[1:], hw[-1:]), dim=0)
            y = torch.cat((y, hw), dim=1)
            offset = torch.cat((offset[1:], offset[-1:]), dim=0)
        elif step == 2: # shift -1
            hw = x[:, -8 * self.number:, ...]
            hw = torch.cat((hw[0:1], hw[0:-1]), dim=0)
            y = torch.cat((y, hw), dim=1)
            offset = torch.cat((offset[0:1], offset[0:-1]), dim=0)

        hw = self.spatial_shift2(hw, offset)

        return torch.cat((y, hw), dim=1)

    def forward(self, x, kernel):

        x_ker = self.encoder_ker1(kernel)
        offset1 = self.offset_conv1(x_ker)

        x = self.channel_shift(x, offset1, step=1)
        x = self.encoder_level1(x)
        x = self.channel_shift(x, offset1, step=2)
        x = self.encoder_level1_1(x)
        x = self.channel_shift(x, offset1, step=1)
        x = self.encoder_level1_2(x)
        x = self.channel_shift(x, offset1, step=2)
        x = self.encoder_level1_3(x)

        return x, x_ker


class KernelDeformableBlock(nn.Module):
    def __init__(self, channel_blur, channel_kernel, deform_groups):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2
        out_channels = deform_groups * 3 * (kernel_size ** 2)

        #CAB1(n_feat, 5, reduction, bias=bias, act=act)
        self.naf_block_blur = CAB1(channel_blur, 3, 4, True, None)
        self.naf_block_kernel = CAB1(channel_kernel, 3, 4, True, None)

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channel_kernel, channel_kernel, 1, stride=1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(channel_kernel, out_channels, 1, stride=1, padding=0, bias=True)
        )

        self.deform = DeformConv2d(channel_blur, channel_blur, kernel_size, padding=2, groups=deform_groups, dilation=2)
        self.fusion = nn.Conv2d(in_channels=channel_blur * 2, out_channels=channel_blur, kernel_size=3, stride=1, padding=1)


    def offset_gen(self, x):
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return offset, mask

    def forward(self, feat_blur, feat_kernel):
        '''
        Input
            feat_blur: (B, 256, H/8, W/8)
            feat_kernel: (B, 256, H/8, W/8)

        Output
            feat_blur: (B, 256, H/8, W/8)
        '''

        feat_blur = self.naf_block_blur(feat_blur)
        feat_kernel = self.naf_block_kernel(feat_kernel)

        offset, mask = self.offset_gen(self.offset_conv(feat_kernel))
        feat = self.deform(feat_blur, offset, mask)

        out = torch.cat((feat_blur, feat), dim=1)
        out = self.fusion(out)

        return out, feat_kernel


class Encoder2(nn.Module):
    def __init__(self, n_features, kernel_size=3, reduction=4, bias=False, scale_unetfeats=48):
        super(Encoder2, self).__init__()
        n_feat = n_features
        scale_unetfeats = 0
        act = nn.PReLU()
        n_feat0 = 24
        # n_feats = 48
        self.act = act

        self.encoder_level0_def = KernelDeformableBlock(n_feat0, n_feat0, 2)
        self.encoder_level1_def = KernelDeformableBlock(n_feat, n_feat, 4)
        self.encoder_level2_def = KernelDeformableBlock(n_feat + scale_unetfeats, n_feat + scale_unetfeats, 8)

        self.concat_ker = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.down01_ker = nn.Sequential(nn.Conv2d(n_feat0, n_feat, 2, 2, 0, bias=False), nn.PReLU())
        self.down12_ker = DownSample(n_feat, scale_unetfeats)
        self.skip_attn1_ker = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.up21_ker = SkipUpSample(n_feat, scale_unetfeats)

        self.encoder_level2 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)
        self.encoder_level2_1 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)
        self.encoder_level2_2 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)

        self.concat = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.down01 = nn.Sequential(nn.Conv2d(n_feat0, n_feat, 2, 2, 0, bias=False), nn.PReLU())

        self.down12 = DownSample(n_feat, scale_unetfeats)

        self.decoder_level1 = KGS_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_1 = KGS_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_2 = KGS_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_3 = KGS_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_4 = KGS_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_5 = KGS_block(n_feat, kernel_size, reduction, bias)

        self.decoder_level2 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)
        self.decoder_level2_1 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)
        self.decoder_level2_2 = KGS_block(n_feat + scale_unetfeats, kernel_size, reduction, bias)


        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.upsample0 = PixelShufflePack(n_feat, n_feat0, 2, upsample_kernel=3)
        self.skip_conv = CAB(n_feat0, kernel_size, reduction, bias=bias,
                             act=act)  # conv(n_feat, n_feat, kernel_size, bias=bias)
        self.out_conv = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.conv_hr0 = conv(n_feat0, n_feat0, kernel_size, bias=bias)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        div = 4
        self.slice_c = n_feat // div


    def forward(self, x, ker, reverse=False):
        # shortcut = x
        # x = self.concat(x)
        x = self.concat(x)
        ker = self.concat_ker(ker)

        shortcut = x

        x, ker = self.encoder_level0_def(x, ker)
        x = self.down01(x)
        ker = self.down01_ker(ker)

        shortcut_enc1 = x
        shortcut_enc1_ker = ker

        enc1, enc1_ker = self.encoder_level1_def(x, ker)

        enc1_down = self.down12(enc1)
        enc1_ker_down = self.down12_ker(enc1_ker)

        enc2, enc2_ker = self.encoder_level2_def(enc1_down, enc1_ker_down)

        enc2, enc2_ker = self.encoder_level2(enc2, enc2_ker)
        enc22, enc2_ker = self.encoder_level2_1(enc2, enc2_ker)
        enc22, enc2_ker = self.encoder_level2_2(enc22, enc2_ker)
        dec2, dec2_ker = self.decoder_level2(enc22, enc2_ker)
        dec22, dec2_ker = self.decoder_level2_1(dec2, dec2_ker)
        dec22, dec2_ker = self.decoder_level2_2(dec22, dec2_ker)

        x = self.up21(dec22, self.skip_attn1(shortcut_enc1))
        x_ker = self.up21_ker(dec2_ker, self.skip_attn1_ker(shortcut_enc1_ker))

        dec1, dec1_ker = self.decoder_level1(x, x_ker)
        dec11, dec1_ker = self.decoder_level1_1(dec1, dec1_ker)
        dec11, dec1_ker = self.decoder_level1_2(dec11, dec1_ker)
        dec11, dec1_ker = self.decoder_level1_3(dec11, dec1_ker)
        dec11, dec1_ker = self.decoder_level1_4(dec11, dec1_ker)
        dec11, dec1_ker = self.decoder_level1_5(dec11, dec1_ker)

        dec11_out = self.conv_hr0(self.act(self.upsample0(dec11))) + self.skip_conv(shortcut)
        dec11_out = self.out_conv(dec11_out)
        return dec11_out  # [enc11, enc22, enc33], [dec11, dec22, dec33]



class TFR_UNet(nn.Module):
    def __init__(self, n_feat0, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(TFR_UNet, self).__init__()
        scale_unetfeats = 4
        self.encoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(1)]
        self.encoder_level3 = [CAB(n_feat0 + 2 * scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(1)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat0, scale_unetfeats)
        self.down23 = DownSample(n_feat0 + scale_unetfeats, scale_unetfeats)

        self.decoder_level1 = [CAB(n_feat0, kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(1)]
        self.decoder_level3 = [CAB(n_feat0 + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(1)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat0, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat0 + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat0, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat0 + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        shortcut = x
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1


class GyroDVD(nn.Module):
    def __init__(self, n_feats2=128, future_frames=1, past_frames=1):
        super(GyroDVD, self).__init__()
        self.n_feats = 48
        self.n_feats2 = n_feats2
        self.num_ff = future_frames
        self.num_fb = past_frames
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.n_feats0 = 24
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats0, 3, 1, 1))

        self.feat_extract_rot = nn.Sequential(nn.Conv2d(16, self.n_feats0, 3, 1, 1))

        self.feat_extract_tran = nn.Sequential(nn.Conv2d(16, self.n_feats0, 3, 1, 1))


        self.conv_last = conv(self.n_feats0, 3, 5, bias=False)

        self.lrelu = nn.PReLU()
        self.stage1 = Encoder2(self.n_feats2, scale_unetfeats=0)
        self.orb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False,
                             scale_unetfeats=0)
        self.orb1_rot = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False,
                             scale_unetfeats=0)
        self.orb1_tran = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False,
                             scale_unetfeats=0)
        self.conv_trans = conv(self.n_feats0, self.n_feats0, 3, bias=True)
        self.conv_trans_rot = conv(self.n_feats0, self.n_feats0, 3, bias=True)
        self.conv_trans_tran = conv(self.n_feats0, self.n_feats0, 3, bias=True)

        self.rorb1 = TFR_UNet(self.n_feats0, self.n_feats, kernel_size=3, reduction=4, act=nn.PReLU(), bias=False,
                              scale_unetfeats=0)
        self.rconcat = nn.Conv2d(self.n_feats0 * 3, self.n_feats0, 3, 1, 1, bias=True)

        div = 4
        self.slice_c = self.n_feats // div

    def stage0(self, x0):
        shortcut = x0
        x0 = self.orb1(x0)
        res0 = x0 + shortcut
        return res0, self.conv_trans(res0)

    def stage0_rot(self, x0):
        shortcut = x0
        x0 = self.orb1_rot(x0)
        res0 = x0 + shortcut
        return res0, self.conv_trans_rot(res0)

    def stage0_tran(self, x0):
        shortcut = x0
        x0 = self.orb1_tran(x0)
        res0 = x0 + shortcut
        return res0, self.conv_trans_tran(res0)


    def stage2(self, x0, sam1_feats, decoder_out0):
        x = self.rconcat(torch.cat((x0, sam1_feats, decoder_out0), dim=1))
        shortcut = x
        x = self.rorb1(x)
        x = x + shortcut
        x = self.conv_last(x)
        return x

    def forward(self, x, k_rot=None, k_tran=None):
        batch_size, frames, channels, height, width = x.shape
        x = x[0]
        k_rot = k_rot[0]
        k_tran = k_tran[0]

        shortcut = x
        x0 = self.feat_extract(x)
        x0_rot = self.feat_extract_rot(k_rot)
        x0_tran = self.feat_extract_tran(k_tran)

        sam_features0, sam_features = self.stage0(x0)
        _, x0_rot = self.stage0_rot(x0_rot)
        _, x0_tran = self.stage0_tran(x0_tran)

        x0_ker = x0_rot + x0_tran

        decoder_outs = self.stage1(sam_features, x0_ker)
        output_features = self.stage2(x0[self.num_fb:frames-self.num_ff], sam_features0[self.num_fb:frames-self.num_ff], decoder_outs[self.num_fb:frames-self.num_ff])
        out = output_features + shortcut[self.num_fb:frames-self.num_ff]

        return out.unsqueeze(0)


@ARCH_REGISTRY.register()
class GyroDVD_128(GyroDVD):
    def __init__(self, n_feats2=128, future_frames=1, past_frames=1):
        super().__init__(
            n_feats2=n_feats2,
            future_frames=future_frames,
            past_frames=past_frames,
        )

@ARCH_REGISTRY.register()
class GyroDVD_96(GyroDVD):
    def __init__(self, n_feats2=96, future_frames=1, past_frames=1):
        super().__init__(
            n_feats2=n_feats2,
            future_frames=future_frames,
            past_frames=past_frames,
        )

@ARCH_REGISTRY.register()
class GyroDVD_64(GyroDVD):
    def __init__(self, n_feats2=64, future_frames=1, past_frames=1):
        super().__init__(
            n_feats2=n_feats2,
            future_frames=future_frames,
            past_frames=past_frames,
        )

@ARCH_REGISTRY.register()
class GyroDVD_48(GyroDVD):
    def __init__(self, n_feats2=48, future_frames=1, past_frames=1):
        super().__init__(
            n_feats2=n_feats2,
            future_frames=future_frames,
            past_frames=past_frames,
        )