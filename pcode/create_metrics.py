# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import math
import torch

class Metrics(object):
    """"""

    def __init__(self, model, task='classification'):
        self.num_classes = getattr(model, 'num_classes', None)
        self.task = task
        self.metric_names = None
        self._metrics_fn_name = self._infer().__name__

    @property
    def metrics_fn(self):
        return getattr(self, self._metrics_fn_name)

    def evaluate(self, loss, output, target, **kwargs):
        return self.metrics_fn(loss, output, target, **kwargs)

    def _infer(self):
        if self.task == 'classification':
            print('model.classes', self.num_classes)
            self.topks = (1, 5) if self.num_classes is not None and self.num_classes >= 5 else (1,)
            self.metric_names = ['top{}'.format(topk) for topk in self.topks]
            return self._accuracy
        elif self.task == 'language_modeling':
            self.metric_names = ['ppl']
            return self._ppl
        elif self.task == 'transformer_nmt':
            self.metric_names = ['ppl', 'top1']
            return self._transformer_nmt
        else:
            raise NotImplementedError
        assert self.metric_names is not None

    def _accuracy(self, loss, output, target):
        """Computes the precision@k for the specified values of k"""
        res = []
        if not self.topks or any((k <= 0 for k in self.topks)):
            print(f'Warning: Invalid self.topks {self.topks}, returning [0]')
            return [0] * max(len(self.topks), 1)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f'Warning: output contains nan or inf: {output}')
            return [0] * len(self.topks)
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f'Warning: target contains nan or inf: {target}')
            return [0] * len(self.topks)
        if (target < 0).any() or (target >= output.size(1)).any():
            print(f'Warning: target out of bounds [0, {output.size(1) - 1}]: {target}')
            return [0] * len(self.topks)
        maxk = max(self.topks)
        batch_size = target.size(0)
        if batch_size == 0:
            print(f'Warning: batch_size is 0, returning [0]')
            return [0] * len(self.topks)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for topk in self.topks:
            correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size).item()
            if not torch.isfinite(torch.tensor(acc)):
                print(f'Warning: accuracy for top-{topk} is {acc}, setting to 0')
                acc = 0.0
            res.append(acc)
        return res

    def _ppl(self, loss, output, target):
        return [math.exp(loss)]

    def _transformer_nmt(self, loss, output, target, **kwargs):
        pred = output.max(1)[1]
        n_correct = pred.eq(target)
        n_correct = n_correct.masked_select(kwargs['non_pad_mask']).sum().item()
        return [math.exp(loss), n_correct / kwargs['n_samples']]
