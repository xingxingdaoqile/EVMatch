import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg, with_cp):
        super(DeepLabV3Plus, self).__init__()

        #resnet.__dict__[cfg['backbone']]: 通过访问 resnet 模块的 __dict__ 属性来获取对应于 cfg['backbone'] 指定名称的 ResNet 架构。例如，如果 cfg['backbone'] 是 'resnet50'，则此行将返回 resnet50 函数
        #断言检查: 确保 cfg['backbone'] 等于 'xception'。这是一种安全检查，如果条件不满足，程序会抛出 AssertionError，提示用户配置错误。这表明该代码只允许使用 xception 作为主干网络的选择之一
        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'],
                                                             with_cp=with_cp)#是一个用于控制卷积层膨胀率的参数
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        #低层特征用于细粒度信息，高层特征用于全局上下文信息
        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        #低层特征 提供细节信息，通过 1x1 卷积将通道数减少到 48。
        #高层特征 提供抽象信息，通过降维使其通道数减少为 high_channels // 8。
        #高层和低层特征在通道维度上拼接后，经过两个 3x3 卷积进行融合，输出一个统一的特征图，既包含了细节，又包含了抽象信息，用于最终的分类
        #降低通道数，以减少计算复杂度并保留局部的精细信息
        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        #fuse模块通过两个3x3卷积层将高层和低层特征进行融合，融合后输出256通道的特征
        #对高、低层次特征进行卷积操作，并输出统一的特征图，为最终分类做准备
        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False):#need_fp: 是否需要返回额外的特征图
        h, w = x.shape[-2:]

        #feats 是一个包含多个不同层次特征图的列表或元组
        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        #对 c1 进行拼接操作。nn.Dropout2d(0.5)(c1) 会在特征图 c1 上随机丢弃50%的特征，以增加网络的泛化能力，并将结果和原始的 c1 进行通道上的拼接
        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2) #这一行将 outs 沿着通道维度切分成两个部分：out: 最终的输出结果，out_fp: 额外的特征图（feature map）

            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out #返回融合后的特征图

    #这是一个解码函数，用于进一步处理 c1 和 c4 两个特征图，并生成最终的输出
    def _decode(self, c1, c4):
        c4 = self.head(c4) #这一行通过网络的 head 部分处理特征图 c4，通常用于对高层次特征进行进一步提炼
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1) #对 c1 进行降维处理，通常是通过 self.reduce 函数来减少通道数

        feature = torch.cat([c1, c4], dim=1) #将处理后的 c1 和 c4 在通道维度上进行拼接，形成一个新的特征图
        feature = self.fuse(feature) #对拼接后的特征图进行进一步融合操作，通常是通过卷积层或其他操作来整合低层次和高层次的特征

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block

#它利用全局平均池化将特征压缩为单一的空间位置，再通过插值将其恢复到原始尺寸，并通过卷积和归一化增强特征表达
#nn.AdaptiveAvgPool2d(1)：自适应的 2D 平均池化，将输入特征图的空间维度（height × width）压缩为 1×1。这个操作可以获得全局上下文信息
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False), #1×1 卷积，用于调整通道数，同时保留空间维度（这里因为池化后是 1×1，空间维度就是 1）
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x) #将输入 x 通过 self.gap 进行全局池化和卷积处理。经过 nn.AdaptiveAvgPool2d(1) 之后，输出的空间维度将变为 1×1，通道数为 out_channels，即 pool 是一个全局上下文信息特征
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)#使用 F.interpolate 函数对池化后的特征图进行 双线性插值，将其恢复到输入特征图的原始空间尺寸 (h, w)。这一步是为了将全局特征与原始分辨率保持一致
        #mode="bilinear": 指定插值方法为双线性插值，align_corners=True: 保证插值时像素的对齐，使得边界像素与原图像的角像素对齐

class ASPPModule(nn.Module):#ASPP模块能够聚合来自不同尺度的上下文特征，提升分割的精度
    #atrous_rates: 三个膨胀卷积的膨胀率列表，用于控制卷积核的感受野大小
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
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)#1 表示在通道维度上进行拼接，最终得到一个新的特征图 y，它的通道数是 feat0, feat1, feat2, feat3, feat4 的通道数之和
        return self.project(y)
