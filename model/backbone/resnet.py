import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


__all__ = ['ResNet', 'resnet50', 'resnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)#为了保证卷积后输出大小不受影响（保持宽高），这里的 padding 是由 dilation 决定的。如果 dilation=1，则相当于普通的 3x3 卷积；如果 dilation=2，卷积核会有间隔，则需要更大的 padding 以保证输出尺寸。bias=False：卷积层中不使用偏置项，因为一般在使用 Batch Normalization 时，偏置项可以被归一化去除，因此可以省略。


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, with_cp=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)#原地操作，直接在输入张量上修改数据，节省内存但要小心使用
        self.downsample = downsample
        self.stride = stride
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):

            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
        # if x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)#当启用检查点机制且输入需要计算梯度时，使用检查点来节省内存
        else:
            # print(self.with_cp)
            out = _inner_forward(x) #在其他情况下（推理阶段或未启用检查点）直接进行标准的前向传播
        # out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    #layers：传入一个包含四个元素的列表，定义每个残差阶段中子块的数量（例如，[3, 4, 6, 3] 对应 ResNet-50）
    #zero_init_residual：如果为 True，则零初始化残差块中最后的 BatchNorm 层，以改进模型的收敛。
    #groups 和 width_per_group：决定卷积层分组以及组宽，用于调整卷积的宽度
    #这是一个包含 3 个布尔值的列表，对应 layer2, layer3, 和 layer4。它决定了是否在这些层中用膨胀卷积替代下采样操作（stride=2）。如果对应的布尔值为 True，则在该层中使用膨胀卷积代替 stride=2 的卷积，以避免下采样和分辨率损失
    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, with_cp=False):#允许用膨胀卷积替换某些下采样操作，从而保留更多的空间信息，适用于图像分割任务
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_cp=with_cp)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], with_cp=with_cp)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], with_cp=with_cp)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], with_cp=with_cp)

        #权重初始化：使用 Kaiming 初始化卷积层的权重，设置标准化层的权重为 1，偏置为 0。如果 zero_init_residual 为 True，则对 Bottleneck 结构的最后一层 BN 做零初始化
        #这里的 self.modules() 是 PyTorch 模块中的方法，返回模型中所有子模块（例如，卷积层、归一化层等）。该循环遍历模型中的每一个子模块
        #mode='fan_out' 表示初始化考虑到输出神经元的数量
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        self.with_cp = with_cp
    #构建由多个 block 组成的层，包含多个残差单元。downsample 用于调整输入维度和步幅
    #blocks：该层包含的残差块的数量
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, with_cp=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, with_cp=with_cp))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)#最后，将所有残差块（layers 列表）用 nn.Sequential 包装成一个层，返回这个层。这样，整个残差层就可以作为网络中的一部分，依次执行所有的残差块

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4 #c1 到 c4 代表了从低级到高级特征的逐层提取

#用来构建具体的 ResNet 模型，比如 ResNet-50 和 ResNet-101。如果指定 pretrained=True，则加载预训练权重
#arch: 字符串，表示网络的架构（如 'resnet50'、'resnet101' 等）
#**kwargs: 其他额外的关键字参数，允许灵活传递额外的参数
def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # 设置预训练权重路径: 根据传入的 arch 参数，生成预训练权重文件的路径。%s 将被 arch 字符串替换，形成完整路径，如 "pretrained/resnet50.pth"
    # 加载预训练权重: 使用 torch.load 函数从指定路径加载预训练权重。返回的 state_dict 是一个包含模型参数（权重和偏置）的字典
    #加载权重到模型: 将加载的 state_dict 中的权重加载到模型中。strict=False 表示如果 state_dict 中包含某些参数而模型中没有，或者模型中有而 state_dict 中没有，也不会抛出错误。这对于模型结构的轻微变更是有用的
    if pretrained:
        pretrained_path = "/kaggle/input/resnet101/pytorch/default/1/%s.pth" % arch
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    return model

#resnet50 和 resnet101：分别用于构建 ResNet-50 和 ResNet-101 模型，传入了残差块数（如 [3, 4, 6, 3] 对应 ResNet-50）
def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)
