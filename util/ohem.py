import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# see https://github.com/charlesCXK/TorchSemiSeg/blob/main/furnace/seg_opr/loss_opr.py
class ProbOhemCrossEntropy2d(nn.Module):
    #reduction：用于指定如何汇总损失（mean 表示取平均，sum 表示取总和）
    #thresh：难例的概率阈值。OHEM 会选择低于此阈值的“难分类”样本来计算损失
    #min_kept：保留的最小难例数
    #down_ratio：通常是用于缩小分辨率的比例（在这里未使用）
    #use_weight：是否使用类别权重，用于处理类别不平衡问题
    def __init__(self, ignore_index, reduction='mean', thresh=0.7, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_index)

    #pred：模型的预测输出，通常是形状为 [batch_size, num_classes, height, width] 的张量。
    #target：真实的标签，形状为 [batch_size, height, width]
    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1) #将 target（真实标签）展开成一维向量，方便后续处理
        valid_mask = target.ne(self.ignore_index) #返回一个布尔掩码，标记哪些像素的标签不等于 ignore_index，目的：确定有效像素，忽略掉标记为 ignore_index 的无效像素
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1) #对预测结果 pred 进行 softmax 操作，计算每个像素属于不同类别的概率分布
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        #num_valid 是有效像素的数量。如果有效像素少于 min_kept，则跳过难例挖掘；否则，进行难例挖掘。
        #难例挖掘通过对每个有效像素的预测概率进行排序，然后根据 min_kept 和 thresh 选择难例
        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1) #使用掩码 valid_mask 将无效像素的概率设置为 1，因为它们不参与损失计算
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)] #获取每个像素对应类别的预测概率，用于后续进行难例选择
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh: #如果排序后的最小难例概率大于设定的阈值 self.thresh，则更新阈值为该难例的概率，以确保选择足够难的样本
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold) #生成一个掩码 kept_mask，表示保留哪些难例（预测概率小于或等于阈值的样本）
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index) #将无效像素标记为 ignore_index，确保它们不参与损失计算
        target = target.view(b, h, w)

        return self.criterion(pred, target) #最后计算并返回损失，使用 self.criterion（交叉熵损失函数）在经过 OHEM 筛选后的样本上计算损失
