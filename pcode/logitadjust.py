# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_num_list = cls_num_list + 1
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        m_list = torch.clamp(m_list, min=-10, max=10)
        self.register_buffer('m_list', m_list.view(1, -1))
        self.weight = weight

    def forward(self, x, target):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f'警告：logits 包含 nan/inf')
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        if x.shape[1] != self.m_list.shape[1]:
            raise ValueError(f'LogitAdjust shape mismatch: Logits={x.shape}, m_list={self.m_list.shape}. Did you set correct num_classes ({self.m_list.shape[1]}) for dataset?')
        x_m = x + self.m_list
        if torch.isnan(x_m).any() or torch.isinf(x_m).any():
            print(f'警告：adjusted logits 包含 nan/inf')
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return F.cross_entropy(x_m, target, weight=self.weight)
