import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
from model.depth.modules import PAB, EncoderB, DecoderB, Output


class SegStereo(nn.Module):
    def __init__(self, cfg):
        super(SegStereo, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        # self.disp_head = ParallelDisp()

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, x_r=None, need_fp=False):
        h, w = x.shape[-2:]
        out = []

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]        # c1: 1/4   c4: 1/16
        disp = None

        # if x_r is not None:
        #     feats_r = self.backbone.base_forward(x_r)
        #     disp = self.disp_head(feats, feats_r)

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            if disp is not None:
                return out, out_fp, disp
            else:
                return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class ParallelDisp(nn.Module):
    def __init__(self):
        super(ParallelDisp, self).__init__()
        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################
        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #
        ## channels  #  16 #  32   #  256  #  512  #  1024/2048 #
        ###############################################################

        # Feature Extraction
        self.feature_decode = _Decoder([2048, 1024, 512, 256, 128])

        # Cascaded Parallax-Attention Module
        # self.cas_pam = CascadedPAM([128, 96, 64])
        self.cas_pam = CascadedPAM([512, 256, 128])

        # Output Module
        self.output = Output()

        # Disparity Refinement
        # self.refine = Refinement([64, 96, 128, 160, 160, 128, 96, 64, 32, 16])
        self.refine = Refinement([256, 256, 256, 512, 512, 256, 256, 256, 128, 64])
        ##                       # 0 # 1  # 2  # 3   # 4   # 5  # 6  # 7  # 8  # 9

    def forward(self, fea_left, fea_right, max_disp=0):
        # b, _, h, w = x_left.shape
        fea_left = [fea[:2] for fea in fea_left]

        # Feature Extraction
        (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine = self.feature_decode(fea_left)
        (fea_right_s1, fea_right_s2, fea_right_s3), _       = self.feature_decode(fea_right)

        # Cascaded Parallax-Attention Module
        cost_s1, cost_s2, cost_s3 = self.cas_pam([fea_left_s1, fea_left_s2, fea_left_s3],
                                                 [fea_right_s1, fea_right_s2, fea_right_s3])

        # Output Module
        if self.training:
            disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(cost_s1, max_disp // 16)
            disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(cost_s2, max_disp // 8)
            disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output(cost_s3, max_disp // 4)
        else:
            disp_s3 = self.output(cost_s3, max_disp // 4)

        # Disparity Refinement
        disp = self.refine(fea_refine, disp_s3)

        if self.training:
            return disp, \
                   [att_s1, att_s2, att_s3], \
                   [att_cycle_s1, att_cycle_s2, att_cycle_s3], \
                   [valid_mask_s1, valid_mask_s2, valid_mask_s3]
        else:
            return disp


# Hourglass Module for Feature Extraction
class _Decoder(nn.Module):
    def __init__(self, channels):
        super(_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.D0 = EncoderB(1, channels[0], channels[0], downsample=False)
        self.D1 = EncoderB(1, channels[0], channels[1], downsample=False)
        self.D2 = DecoderB(1, channels[1] + channels[1], channels[2])                  # scale: 1/16
        self.D3 = DecoderB(1, channels[2] + channels[2], channels[3])                  # scale: 1/8
        self.D4 = DecoderB(1, channels[3] + channels[3], channels[4])                  # scale: 1/4

    def forward(self, x):
        fea_E1, fea_E2, fea_E3, fea_E4 = x
    #   #  256 # 512  # 1024  #  2048

        fea_D0 = self.D0(fea_E4)                                                       # scale: 1/32  channel: 2048
        fea_D1 = self.D1(fea_D0)                                                       # scale: 1/32  channel: 1024
        fea_D2 = self.D2(torch.cat((fea_D1, fea_E3), 1))                               # scale: 1/16  channel: 512
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E2), 1))                # scale: 1/8   channel: 256
        fea_D4 = self.D4(torch.cat((self.upsample(fea_D3), fea_E1), 1))                # scale: 1/4   channel: 128

        return (fea_D2, fea_D3, fea_D4), fea_E1


# Cascaded Parallax-Attention Module
class CascadedPAM(nn.Module):
    def __init__(self, channels):
        super(CascadedPAM, self).__init__()
        self.stage1 = PAM_stage(channels[0])
        self.stage2 = PAM_stage(channels[1])
        self.stage3 = PAM_stage(channels[2])

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(channels[0] + channels[1], channels[1], 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[2], channels[2], 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True)
        )


    def forward(self, fea_left, fea_right):
        '''
        :param fea_left:    feature list [fea_left_s1, fea_left_s2, fea_left_s3]
        :param fea_right:   feature list [fea_right_s1, fea_right_s2, fea_right_s3]
        '''
        fea_left_s1, fea_left_s2, fea_left_s3 = fea_left
        fea_right_s1, fea_right_s2, fea_right_s3 = fea_right

        b, _, h_s1, w_s1 = fea_left_s1.shape
        b, _, h_s2, w_s2 = fea_left_s2.shape

        # stage 1: 1/16
        cost_s0 = [
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device),
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device)
        ]

        fea_left, fea_right, cost_s1 = self.stage1(fea_left_s1, fea_right_s1, cost_s0)

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b2(torch.cat((fea_left, fea_left_s2), 1))
        fea_right = self.b2(torch.cat((fea_right, fea_right_s2), 1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s1[1].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stage2(fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b3(torch.cat((fea_left, fea_left_s3), 1))
        fea_right = self.b3(torch.cat((fea_right, fea_right_s3), 1))

        cost_s2_up = [
            F.interpolate(cost_s2[0].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s2[1].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s3 = self.stage3(fea_left, fea_right, cost_s2_up)

        return [cost_s1, cost_s2, cost_s3]


class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


# Disparity Refinement Module
class Refinement(nn.Module):
    def __init__(self, channels):
        super(Refinement, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, channels[0] + 1, channels[0], downsample=False)   # scale: 1/4
        self.E1 = EncoderB(1, channels[0],     channels[1], downsample=True)    # scale: 1/8
        self.E2 = EncoderB(1, channels[1],     channels[2], downsample=True)    # scale: 1/16
        self.E3 = EncoderB(1, channels[2],     channels[3], downsample=True)    # scale: 1/32

        self.D0 = EncoderB(1, channels[4],     channels[4], downsample=False)   # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[5], channels[5])           # scale: 1/16
        self.D2 = DecoderB(1, channels[5] + channels[6], channels[6])           # scale: 1/8
        self.D3 = DecoderB(1, channels[6] + channels[7], channels[7])           # scale: 1/4
        self.D4 = DecoderB(1, channels[7],               channels[8])           # scale: 1/2
        self.D5 = DecoderB(1, channels[8],               channels[9])           # scale: 1

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )

    def forward(self, fea, disp):
        # scale the input disparity
        disp = disp / (2 ** 5)

        fea_E0 = self.E0(torch.cat((disp, fea), 1))                         # scale: 1/4  256
        fea_E1 = self.E1(fea_E0)                                            # scale: 1/8  512
        fea_E2 = self.E2(fea_E1)                                            # scale: 1/16 512
        fea_E3 = self.E3(fea_E2)                                            # scale: 1/32 768

        fea_D0 = self.D0(fea_E3)                                            # scale: 1/32 768
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E2), 1))     # scale: 1/16 768+512
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E1), 1))     # scale: 1/8  512+512
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E0), 1))     # scale: 1/4  256+256
        fea_D4 = self.D4(self.upsample(fea_D3))                             # scale: 1/2  256
        fea_D5 = self.D5(self.upsample(fea_D4))                             # scale: 1

        # regression
        confidence = self.confidence(fea_D5)
        disp_res = self.disp(fea_D5)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp, scale_factor=4, mode='bilinear') * (1-confidence) + disp_res * confidence

        # scale the output disparity
        # note that, the size of output disparity is 4 times larger than the input disparity
        return disp * 2 ** 7