# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch

class ModuleState:

    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.keys = set(state_dict.keys())

    def __add__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            new_dict[key] = self.state_dict[key] + other.state_dict[key]
        return ModuleState(new_dict)

    def __iadd__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            self.state_dict[key] += other.state_dict[key]
        return self

    def __sub__(self, other):
        assert other.keys == self.keys
        assert isinstance(other, ModuleState)
        new_dict = {}
        for key in self.keys:
            new_dict[key] = self.state_dict[key] - other.state_dict[key]
        return ModuleState(new_dict)

    def __mul__(self, factor):
        assert isinstance(factor, float) or isinstance(factor, torch.Tensor)
        new_dict = {}
        for key in self.keys:
            data = self.state_dict[key]
            if data.dtype == torch.int64:
                new_dict[key] = self.state_dict[key]
            else:
                value = self.state_dict[key] * factor
                if value.shape != data.shape:
                    value = value.reshape(data.shape)
                new_dict[key] = value
        return ModuleState(new_dict)

    def mul_by_key(self, factor, by_key):
        assert isinstance(factor, float) or isinstance(factor, torch.Tensor)
        new_dict = {}
        for key in self.keys:
            data = self.state_dict[key]
            if data.dtype == torch.int64:
                new_dict[key] = self.state_dict[key]
            elif by_key is not None and by_key == key:
                value = self.state_dict[key] * factor
                if value.shape != data.shape:
                    value = value.reshape(data.shape)
                new_dict[key] = value
            else:
                new_dict[key] = self.state_dict[key]
        return ModuleState(new_dict)

    def __div__(self, factor):
        return self.__mul__(1.0 / factor)

    def copy_to_module(self, module):
        """
        Use this to copy the state to a module object when you need to maintain the
        computation graph that led to this particular state. This does break the model
        for normal optimizers down the line.
        """
        for name, sub_module in module.named_modules():
            params = sub_module._parameters
            for key in params:
                if params[key] is None:
                    continue
                param_name = f'{name}.{key}' if name else key
                if param_name in self.keys:
                    params[key].data.copy_(self.state_dict[param_name])
            buffers = sub_module._buffers
            for key in buffers:
                if buffers[key] is None:
                    continue
                buffer_name = f'{name}.{key}' if name else key
                if buffer_name in self.keys:
                    buffers[key].data.copy_(self.state_dict[buffer_name])

    def copy_part_to_module(self, module):
        """related to lg-fedavg"""
        for name, module in module.named_modules():
            if 'fc2' in name or 'classifier' in name:
                params = module._parameters
                for key in params:
                    param_name = f'{name}.{key}'
                    if param_name in self.keys:
                        params[key] = self.state_dict[param_name]
    __rmul__ = __mul__
    __truediv__ = __div__
