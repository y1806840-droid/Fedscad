# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch

def define_optimizer(conf, model, optimizer_name, lr=None):
    base_lr = conf.lr if lr is None else lr
    mlp_lr_ratio = getattr(conf, 'mlp_lr_ratio', 1.0)
    fc_lr_ratio = getattr(conf, 'fc_lr_ratio', 1.0)
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        param_lr = base_lr
        if 'mlp_head' in key or 'head' in key:
            param_lr = base_lr * mlp_lr_ratio
        elif 'classifier' in key or 'fc' in key:
            param_lr = base_lr * fc_lr_ratio
        params.append({'params': [value], 'name': key, 'weight_decay': 0.0 if 'bn' in key or key.endswith('scaling_train') else conf.weight_decay, 'param_size': value.size(), 'nelement': value.nelement(), 'lr': param_lr})
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=conf.momentum_factor, nesterov=conf.use_nesterov)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, betas=(conf.adam_beta_1, conf.adam_beta_2), eps=conf.adam_eps)
    else:
        raise NotImplementedError
    return optimizer
