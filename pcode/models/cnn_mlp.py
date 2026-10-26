# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['cnn_mlp']

def _decide_num_classes(dataset):
    if dataset == 'cifar10' or dataset == 'svhn':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif 'imagenet' in dataset:
        return 1000
    elif 'mnist' == dataset:
        return 10
    elif 'fmnist' == dataset:
        return 10
    elif 'femnist' == dataset:
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

class CNNMLPCifar(nn.Module):
    """Two-convolution backbone with a 512-D task feature and auxiliary MLP."""

    def __init__(self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True, use_norm=True, feat_dim=128):
        super(CNNMLPCifar, self).__init__()
        self.num_classes = _decide_num_classes(dataset)
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(3, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.feature_dim = 64 * 5 * 5
        self.mlp_head = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Linear(256, feat_dim))
        self.fc1 = nn.Linear(self.feature_dim, 512)
        if use_norm:
            self.classifier = NormedLinear(512, self.num_classes)
        else:
            self.classifier = nn.Linear(512, self.num_classes, bias=w_fc_bias)
        self.weight_keys = [['mlp_head.0.weight', 'mlp_head.0.bias'], ['mlp_head.3.weight', 'mlp_head.3.bias'], ['fc1.weight', 'fc1.bias'], ['classifier.weight', 'classifier.bias'], ['conv2.weight', 'conv2.bias'], ['conv1.weight', 'conv1.bias']]
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x, gmv_dict=None, alpha=0.1, y=None):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))
        activation2 = self.conv2(x)
        x = self.pool(F.relu(activation2))
        x = x.view(-1, self.feature_dim)
        h = F.relu(self.fc1(x))
        enhanced_h = h
        feat_mlp = F.normalize(self.mlp_head(enhanced_h), dim=1)
        logits = self.classifier(enhanced_h)
        if self.save_activations:
            self.activations = [activation1, activation2]
        return (feat_mlp, logits, h)

def cnn_mlp(conf):
    return CNNMLPCifar(conf.data, w_conv_bias=getattr(conf, 'w_conv_bias', True), w_fc_bias=getattr(conf, 'w_fc_bias', False), use_norm=getattr(conf, 'use_norm', True), feat_dim=getattr(conf, 'feat_dim', 128))
