# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

"""Fixed shared class anchors and normalized cosine scores."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _etf_matrix(num_classes, feat_dim, dtype, device):
    C = int(num_classes)
    A = int(feat_dim)
    if A >= C - 1:
        I = torch.eye(C, dtype=torch.float64, device=device)
        J = torch.ones(C, C, dtype=torch.float64, device=device) / C
        M = I - J
        Q, R = torch.linalg.qr(M)
        B = Q[:, :C - 1]
        Y = M @ B
        s = float(torch.sqrt(torch.tensor(C / (C - 1), dtype=torch.float64)))
        Y = s * Y
        if A > C - 1:
            pad = torch.zeros(C, A - (C - 1), dtype=torch.float64, device=device)
            Y = torch.cat([Y, pad], dim=1)
        Y = Y.to(dtype)
        Y = F.normalize(Y, dim=1)
        return Y
    else:
        return F.normalize(torch.randn(C, A, dtype=dtype, device=device), dim=1)

def _random_unit_matrix(num_classes, feat_dim, dtype, device, seed=0):
    C = int(num_classes)
    A = int(feat_dim)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    raw = torch.randn(C, A, dtype=torch.float64, generator=generator)
    return F.normalize(raw, dim=1).to(device=device, dtype=dtype)

def _paired_class_matrix(num_classes, feat_dim, dtype, device):
    """Create a degenerate codebook in which adjacent classes share anchors.

    The group-level directions form a simplex ETF whenever the feature
    dimension permits it, but classes ``(0, 1)``, ``(2, 3)``, and so on are
    assigned exactly the same direction.  This preserves a fixed global
    reference while deliberately removing within-pair class separability.
    """
    C = int(num_classes)
    A = int(feat_dim)
    if C <= 0 or A <= 0:
        raise ValueError('num_classes and feat_dim must be positive.')
    num_groups = (C + 1) // 2
    group_anchors = _etf_matrix(num_groups, A, dtype=dtype, device=device)
    return group_anchors.repeat_interleave(2, dim=0)[:C]

class _AnchorState:

    def __init__(self, anchors):
        self.Ave = anchors
        self.Amount = torch.zeros(anchors.size(0), dtype=anchors.dtype)

class SharedAnchorScores(nn.Module):
    """Temperature-scaled cosine scores against fixed shared class anchors."""

    def __init__(self, feature_dim, temperature=0.2, num_classes=10, dtype_str='float32', anchor_init='etf', anchor_seed=0, **kwargs):
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError('The anchor temperature must be positive.')
        dtype = getattr(torch, dtype_str)
        mode = str(anchor_init).lower()
        if mode == 'etf':
            a = _etf_matrix(num_classes, feature_dim, dtype, torch.device('cpu'))
        elif mode in ('random', 'rand'):
            a = _random_unit_matrix(num_classes, feature_dim, dtype, torch.device('cpu'), anchor_seed)
        elif mode in ('paired', 'paired_class'):
            a = _paired_class_matrix(num_classes, feature_dim, dtype, torch.device('cpu'))
        else:
            raise ValueError('Supported anchor initializations: etf, random, paired_class.')
        self.estimator = _AnchorState(a.clone())
        self.estimator_old = _AnchorState(a.clone())

    def set_client_specific_anchors(self, client_id):
        return None

    def round_reset(self):
        pass

    def forward(self, features, labels=None, **kwargs):
        anchors = F.normalize(self.estimator_old.Ave.to(features.device).detach(), dim=1, eps=1e-06)
        features = F.normalize(features, dim=1, eps=1e-06)
        return features @ anchors.T / self.temperature
