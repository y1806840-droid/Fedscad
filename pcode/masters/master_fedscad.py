# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

"""Server communication and history correction adapted from FedDyn."""
import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pcode.masters.master_fedavg import MasterFedAvg
from pcode.utils.tensor_buffer import TensorBuffer

class MasterFedSCAD(MasterFedAvg):

    def __init__(self, conf):
        super().__init__(conf)
        self.arch_client_ids = {}
        self.arch_param_dim = {}
        for client_id in self.client_ids:
            arch = self.clientid2arch[client_id]
            self.arch_client_ids.setdefault(arch, []).append(client_id)
        for arch, model in self.client_models.items():
            self.arch_param_dim[arch] = int(parameters_to_vector(model.parameters()).numel())
        self.client_history = {client_id: torch.zeros(self.arch_param_dim[self.clientid2arch[client_id]]) for client_id in self.client_ids}

    def _get_hc_weight(self):
        base_alpha = float(getattr(self.conf, 'hc_weight', 0.001))
        return max(base_alpha, 0.0)

    def _model_to_param_vector(self, model):
        return parameters_to_vector(model.parameters()).detach().cpu().clone()

    def _buffer_to_param_vector(self, arch, flat_buffer):
        model = self.client_models[arch]
        param_names = {name for name, _ in model.named_parameters()}
        parts = []
        offset = 0
        for name, tensor in model.state_dict().items():
            numel = tensor.numel()
            if name in param_names:
                parts.append(flat_buffer[offset:offset + numel].reshape(-1))
            offset += numel
        if len(parts) == 0:
            return torch.zeros(self.arch_param_dim[arch], dtype=flat_buffer.dtype)
        return torch.cat(parts).detach().cpu().clone()

    def _send_model_to_selected_clients(self, selected_client_ids):
        self.conf.logger.log('Master send the models to workers.')
        use_hc = bool(getattr(self.conf, 'use_hc', False))
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            if use_hc:
                dist.send(tensor=self.client_history[selected_client_id].cpu(), dst=worker_rank)
            self.conf.logger.log(f'\tMaster send the global model={arch} to process_id={worker_rank}.')
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        flatten_local_models = {}
        self.conf.logger.log('Master waits to receive the local models.')
        dist.barrier()
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            arch = self.clientid2arch[client_id]
            client_tb = TensorBuffer(list(self.client_models[arch].state_dict().values()))
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            try:
                dist.recv(client_tb.buffer, src=world_id)
                flatten_local_models[client_id] = client_tb
            except RuntimeError as e:
                self.conf.logger.log(f'Master failed to receive model from client-{client_id}: {e}')
                continue
            self.conf.logger.log(f'Master received the local model from client-{client_id} (Worker-{world_id}).')
        dist.barrier()
        self.conf.logger.log('Master received all local models.')
        return flatten_local_models

    def _hc_avg_over_archs(self, flatten_local_models):
        alpha = self._get_hc_weight()
        if alpha <= 0:
            self.conf.logger.log(f'HC is disabled at comm_round={self.conf.graph.comm_round}; use uniform model averaging.')
            return super()._avg_over_archs(flatten_local_models)
        archs = set((self.clientid2arch[client_idx] for client_idx in flatten_local_models.keys()))
        archs_fedavg_models = {}
        for arch in archs:
            arch_client_ids = self.arch_client_ids.get(arch, [])
            arch_selected = [cid for cid in flatten_local_models.keys() if self.clientid2arch[cid] == arch]
            if len(arch_selected) == 0:
                continue
            client_weights = {}
            for idx in arch_selected:
                client_weights[idx] = 1.0 / float(len(arch_selected))
            global_param_vec = self._model_to_param_vector(self.client_models[arch])
            avg_param_vec = torch.zeros_like(global_param_vec)
            avg_full_buffer = torch.zeros_like(flatten_local_models[arch_selected[0]].buffer)
            for client_idx in arch_selected:
                weight = client_weights[client_idx]
                local_param_vec = self._buffer_to_param_vector(arch, flatten_local_models[client_idx].buffer)
                avg_param_vec.add_(local_param_vec, alpha=weight)
                avg_full_buffer.add_(flatten_local_models[client_idx].buffer, alpha=weight)
                self.client_history[client_idx].sub_(local_param_vec - global_param_vec, alpha=1.0)
            grad_mean = torch.zeros_like(global_param_vec)
            if len(arch_client_ids) > 0:
                for cid in arch_client_ids:
                    grad_mean.add_(self.client_history[cid], alpha=1.0 / float(len(arch_client_ids)))
            corrected_param_vec = avg_param_vec - grad_mean
            dummy_model = self.client_models[arch]
            model_state_dict = dummy_model.state_dict()
            offset = 0
            for _, tensor in model_state_dict.items():
                numel = tensor.numel()
                tensor.copy_(avg_full_buffer[offset:offset + numel].view(tensor.size()))
                offset += numel
            vector_to_parameters(corrected_param_vec.to(next(dummy_model.parameters()).device), dummy_model.parameters())
            archs_fedavg_models[arch] = dummy_model
            if len(client_weights) > 0:
                ws = list(client_weights.values())
                self.conf.logger.log(f'HC aggregation stats ({arch}): n={len(ws)}, min_w={min(ws):.6f}, max_w={max(ws):.6f}, avg_w={sum(ws) / len(ws):.6f}')
            self.conf.logger.log(f'Master aggregates over {len(arch_selected)} models ({arch}) using history correction.')
        return archs_fedavg_models

    def _avg_over_archs(self, flatten_local_models):
        if bool(getattr(self.conf, 'use_hc', False)):
            return self._hc_avg_over_archs(flatten_local_models)
        return super()._avg_over_archs(flatten_local_models)
