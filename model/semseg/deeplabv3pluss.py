from torch.cuda.amp import autocast

import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class DLSideout(nn.Module):
    def __init__(self, cfg, with_cp):
        super(DLSideout, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'],
                                                             with_cp=with_cp)
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        self.with_cp = with_cp
        enc_channels = [256, 512, 1024, 2048]

        self.head = ASPPModule(enc_channels[3], cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(enc_channels[0], 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(enc_channels[3] // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.sideout1 = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.sideout2 = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.sideout3 = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.sideout4 = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.sideout5 = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

        self.sideout = [self.sideout1, self.sideout2, self.sideout3, self.sideout4, self.sideout5]

    def forward(self, x, num_ulb=0):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if self.training and num_ulb > 0:
            outs = self._decode_ms(torch.cat((c1, nn.Dropout2d(0.5)(c1[-num_ulb:]))),
                                   torch.cat((c4, nn.Dropout2d(0.5)(c4[-num_ulb:]))))

            out_list = []
            out_fp_list = []
            for it in outs:
                it = F.interpolate(it, size=(h, w), mode="bilinear", align_corners=True)
                out = it[:-num_ulb]
                out_fp = it[-num_ulb:]
                out_list.append(out)
                out_fp_list.append(out_fp)
            return out_list, out_fp_list
        elif self.training and num_ulb == 0:
            outs = self._decode_ms(c1, c4)
            out_list = []
            for it in outs:
                it = F.interpolate(it, size=(h, w), mode="bilinear", align_corners=True)
                out_list.append(it)
            return out_list
        else:
            out = self._decode(c1, c4)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

            return out

    def _decode_ms(self, c1, c4):

        def _inner_decode(c1, c4):
            c4, features = self.head(c4)
            c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

            c1 = self.reduce(c1)

            feature = torch.cat([c1, c4], dim=1)
            feature = self.fuse(feature)
            out = self.classifier(feature)

            sideout = [classfier(feat) for feat, classfier in zip(features, self.sideout)]
            sideout.append(out)

            return sideout

        if self.with_cp:
            return cp.checkpoint(_inner_decode, c1, c4)
        else:
            return _inner_decode(c1, c4)

    def _decode(self, c1, c4):
        c4, _ = self.head(c4)
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
        return self.project(y), [feat0, feat1, feat2, feat3, feat4]
