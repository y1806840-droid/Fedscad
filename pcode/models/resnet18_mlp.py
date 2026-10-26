# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['resnet18_mlp']

def _decide_num_classes(dataset):
    if dataset == 'cifar10' or dataset == 'svhn':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif 'imagenet' in dataset:
        if 'tiny_imagenet' in dataset:
            return 200
        return 1000
    elif dataset == 'mnist' or dataset == 'fmnist':
        return 10
    elif dataset == 'femnist':
        return 62
    else:
        raise NotImplementedError(f'this dataset ({dataset}) is not supported yet.')

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def _num_groups(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_conv_bias=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride, bias=w_conv_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, bias=w_conv_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=w_conv_bias), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18CifarEncoder(nn.Module):

    def __init__(self, w_conv_bias=True, scaling=1.0):
        super(ResNet18CifarEncoder, self).__init__()
        base = max(1, int(64 * scaling))
        self.in_planes = base
        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=w_conv_bias)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(base, 2, stride=1, w_conv_bias=w_conv_bias)
        self.layer2 = self._make_layer(base * 2, 2, stride=2, w_conv_bias=w_conv_bias)
        self.layer3 = self._make_layer(base * 4, 2, stride=2, w_conv_bias=w_conv_bias)
        self.layer4 = self._make_layer(base * 8, 2, stride=2, w_conv_bias=w_conv_bias)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, blocks, stride, w_conv_bias=True):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride=stride, w_conv_bias=w_conv_bias))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1, w_conv_bias=w_conv_bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)
        return feat

class ResNet18CifarMLP(nn.Module):
    """Small-resolution ResNet-18 with separate task and projection heads."""

    def __init__(self, dataset, use_norm=False, feat_dim=256, w_conv_bias=True, w_fc_bias=False, scaling=1.0):
        super(ResNet18CifarMLP, self).__init__()
        self.num_classes = _decide_num_classes(dataset)
        self.use_norm = use_norm
        self.encoder = ResNet18CifarEncoder(w_conv_bias=w_conv_bias, scaling=scaling)
        dim_in = max(1, int(64 * scaling)) * 8
        self.head = nn.Sequential(nn.Linear(dim_in, 512, bias=w_fc_bias), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim, bias=w_fc_bias))
        if use_norm:
            self.fc = NormedLinear(dim_in, self.num_classes)
        else:
            self.fc = nn.Linear(dim_in, self.num_classes, bias=w_fc_bias)
        self.weight_keys = [['head.0.weight', 'head.0.bias'], ['head.3.weight', 'head.3.bias'], ['fc.weight', 'fc.bias']]

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)
        return (feat_mlp, logits, feat)

def resnet18_mlp(conf):
    dataset = conf.data
    use_norm = getattr(conf, 'use_norm', False)
    feat_dim = getattr(conf, 'feat_dim', 512)
    w_conv_bias = getattr(conf, 'w_conv_bias', True)
    w_fc_bias = getattr(conf, 'w_fc_bias', True)
    scaling = float(getattr(conf, 'resnet_scaling', 1.0))
    if 'cifar' in dataset or 'svhn' in dataset or 'imagenet' in dataset:
        return ResNet18CifarMLP(dataset, use_norm=use_norm, feat_dim=feat_dim, w_conv_bias=w_conv_bias, w_fc_bias=w_fc_bias, scaling=scaling)
    else:
        raise NotImplementedError('resnet18_mlp 当前仅支持 CIFAR/SVHN 尺度输入')
