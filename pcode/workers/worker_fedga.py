# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

"""Client training: task CE, shared-anchor CE, detached-score KD and history correction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from torch.nn.utils import parameters_to_vector
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
from pcode.logitadjust import LogitAdjust
from pcode.proco import ProCoLoss
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.workers.worker_base import WorkerBase

class WorkerFedGA(WorkerBase):

    def __init__(self, conf):
        conf.proco_single_view = True
        super().__init__(conf)
        self.num_classes = 10 if self.conf.data == 'cifar10' else 100 if self.conf.data == 'cifar100' else 200 if 'tiny_imagenet' in self.conf.data else 1000
        fedga_global_init = str(getattr(self.conf, 'fedga_global_init', 'etf')).lower()
        use_etf_init = fedga_global_init == 'etf'
        self.proco_loss = ProCoLoss(contrast_dim=self.conf.feat_dim, temperature=self.conf.proco_temp, num_classes=self.num_classes, dtype_str=getattr(self.conf, 'estimator_dtype', 'float32'), bessel_mode=getattr(self.conf, 'bessel_mode', 'approx'), update_every=getattr(self.conf, 'estimator_update_every', 1), freeze_old=True, use_etf=use_etf_init, global_init=fedga_global_init, anchor_seed=getattr(self.conf, 'manual_seed', 0))
        if hasattr(self.conf, 'logger'):
            init_detail = ' (rho=0.9)' if fedga_global_init == 'clustered' else ' (adjacent classes share anchors)' if fedga_global_init in ('paired_class', 'paired') else ''
            self.conf.logger.log(f'[{conf.algo}] auxiliary class-reference init={fedga_global_init}{init_detail}')
        self.scl_criterion = nn.CrossEntropyLoss(reduction='mean')
        self._fedga_seen_rounds = 0
        self._fedga_enable_aux = False
        self._last_loss_breakdown = {}
        self._logged_anchor_clients = set()

    def _get_feddyn_alpha(self):
        base_alpha = float(getattr(self.conf, 'feddyn_alpha', 0.001))
        return max(base_alpha, 0.0)

    def _recv_model_from_master(self):
        ignore_global_mlp = False
        local_mlp_state = {}
        dist.recv(self.model_tb.buffer, src=0)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        use_feddyn = bool(getattr(self.conf, 'use_feddyn', False))
        self.dual_variable = None
        self.regular_params = None
        self.dual_variable_list = None
        self.regular_params_list = None
        if use_feddyn:
            dual_size = int(parameters_to_vector(self.model.parameters()).numel())
            dual_buffer = torch.zeros(dual_size, dtype=torch.float32)
            dist.recv(dual_buffer, src=0)
            self.dual_variable = dual_buffer.detach().clone()
            self.regular_params = parameters_to_vector(self.model.parameters()).detach().clone()

            def vector_to_parameters_list(vec, parameters):
                pointer = 0
                res = []
                for param in parameters:
                    num_param = param.numel()
                    res.append(vec[pointer:pointer + num_param].view_as(param).detach())
                    pointer += num_param
                return res
        device = torch.device('cuda' if self.conf.graph.on_cuda else 'cpu')
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(device))
        if use_feddyn:
            self.dual_variable = self.dual_variable.to(device)
            self.regular_params = self.regular_params.to(device)
            self.dual_variable_list = vector_to_parameters_list(self.dual_variable, self.model.parameters())
            self.regular_params_list = vector_to_parameters_list(self.regular_params, self.model.parameters())
        self.conf.logger.log(f'Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global model ({self.arch}) from Master.')
        dist.barrier()

    def prepare_local_train_loader(self):
        if self.conf.prepare_data == 'combine':
            local_train_data = self.local_datasets[self.conf.graph.client_id]['train']
            self.train_loader = create_dataset.define_local_data_loader(self.conf, self.conf.graph.client_id, data_type='train', data=local_train_data)
            cls_num_list = self._compute_local_class_counts(local_train_data)
        else:
            self.train_loader, _ = create_dataset.define_data_loader(self.conf, dataset=self.dataset['train'], localdata_id=self.conf.graph.client_id - 1, is_train=True, data_partitioner=self.data_partitioner)
            cls_num_list = [0] * self.num_classes
        return cls_num_list

    def _compute_local_class_counts(self, data):
        class_counts = [0] * self.num_classes
        targets = None
        indices = None
        if hasattr(data, 'dataset') and hasattr(data.dataset, 'targets') and hasattr(data, 'indices'):
            targets = data.dataset.targets
            indices = data.indices
        elif hasattr(data, 'data') and hasattr(data.data, 'targets') and hasattr(data, 'indices'):
            targets = data.data.targets
            indices = data.indices
        elif hasattr(data, 'targets'):
            targets = data.targets
        if targets is None:
            return class_counts
        if indices is None:
            indices = range(len(targets))
        for idx in indices:
            label = targets[idx]
            label = int(label) if not isinstance(label, int) else label
            if 0 <= label < self.num_classes:
                class_counts[label] += 1
        return class_counts

    def _train(self):
        self._fedga_enable_aux = self._fedga_seen_rounds > 0
        self._fedga_seen_rounds += 1
        client_anchor_info = None
        if self.proco_loss is not None:
            self.proco_loss.round_reset()
            self.proco_loss.estimator.Ave.fill_(0)
            self.proco_loss.estimator.Amount.fill_(0)
            client_anchor_info = self.proco_loss.set_client_specific_anchors(self.conf.graph.client_id)
        if client_anchor_info is not None and self.conf.graph.client_id not in self._logged_anchor_clients and hasattr(self.conf, 'logger'):
            permutation = client_anchor_info['permutation']
            signature = client_anchor_info['signature']
            if permutation is not None:
                detail = 'permutation=' + str(permutation[:min(10, permutation.numel())].tolist())
            else:
                detail = 'anchor_signature=' + str([round(float(value), 4) for value in signature.tolist()])
            self.conf.logger.log(f"[FedGA] client-{self.conf.graph.client_id} fixed {client_anchor_info['mode']} {detail}")
            self._logged_anchor_clients.add(self.conf.graph.client_id)
        self.model.train()
        cls_num_list = self.prepare_local_train_loader()
        self._cls_num_list = cls_num_list
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()
        device = next(self.model.parameters()).device
        use_logit_adjust = bool(getattr(self.conf, 'use_logit_adjust', False))
        try:
            if use_logit_adjust:
                self.criterion = LogitAdjust(cls_num_list).to(device)
            else:
                self.criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        except Exception:
            self.criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        self.scl_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        if self.proco_loss is not None:
            self.proco_loss = self.proco_loss.to(device)
        self.optimizer = create_optimizer.define_optimizer(self.conf, model=self.model, optimizer_name=self.conf.optimizer)
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        current_epoch = 0
        while True:
            for param in self.model.parameters():
                param.requires_grad = True
            for data_batch_idx, (_input, _target) in enumerate(self.train_loader):
                data_batch = create_dataset.load_data_batch(self.conf, _input, _target, is_training=True)
                self.optimizer.zero_grad()
                try:
                    loss, _ = self._inference(data_batch, update_tracker=True, phase=None)
                except FloatingPointError as e:
                    self.conf.logger.log(f'Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) encountered non-finite values during training: {e}. Early stop it.')
                    self._terminate_comm_round()
                    return
                loss.backward()
                use_grad_clip = getattr(self.conf, 'use_grad_clip', False)
                if use_grad_clip:
                    clip_norm = getattr(self.conf, 'grad_clip_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                is_last_batch = current_epoch == int(self.conf.local_n_epochs) - 1 and data_batch_idx == len(self.train_loader) - 1
                if (self.scheduler.local_index - 1) % self.conf.summary_freq == 0 or is_last_batch:
                    curr_lr = self.optimizer.param_groups[0]['lr']
                    loss_info = getattr(self, '_last_loss_breakdown', {})
                    self.conf.logger.log(f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) phase=joint, loss_total={loss.detach().item():.6f}, loss_ce={loss_info.get('ce_loss', 0.0):.6f}, loss_scl={loss_info.get('scl_loss', 0.0):.6f}, loss_local_kd={loss_info.get('kd_loss', 0.0):.6f}, loss_global_kd={loss_info.get('global_fc_kd_loss', 0.0):.6f}, loss_dyn={loss_info.get('dyn_loss', 0.0):.6f}, lr={curr_lr:.6f}")
                if self.tracker.stat['loss'].avg > 1000.0 or torch.isnan(torch.tensor(self.tracker.stat['loss'].avg)):
                    self._terminate_comm_round()
                    return
                if self._is_finished_one_comm_round():
                    display_training_stat(self.conf, self.scheduler, self.tracker)
                    self._terminate_comm_round()
                    return
            display_training_stat(self.conf, self.scheduler, self.tracker)
            self.tracker.reset()
            current_epoch += 1
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _release_round_training_state(self):
        """Release local-only state; model weights/BN buffers remain for upload.

        The optimizer, scheduler and HC inputs are recreated/received every
        round. Server-side client history and ProCo statistics are untouched.
        """
        for module in self.model.modules():
            if hasattr(module, 'activations'):
                module.activations = None
        for parameter in self.model.parameters():
            parameter.grad = None
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is not None:
            optimizer.state.clear()
        self.optimizer = None
        self.scheduler = None
        self.dual_variable_list = None
        self.regular_params_list = None
        self.dual_variable = None
        self.regular_params = None

    def _compute_auxiliary_logits(self, features, target):
        return self.proco_loss(features, target, update_features=features.detach())

    def _inference(self, data_batch, update_tracker=True, phase=None):
        x = data_batch['input']
        target = data_batch['target']
        if len(x.shape) == 5:
            x = x[:, 0, :, :, :]
        curr_round = int(self.conf.graph.comm_round)
        feat_fc = None
        model_out = self.model(x)
        if isinstance(model_out, (list, tuple)):
            if len(model_out) >= 3:
                feat_mlp, logits, feat_fc = (model_out[0], model_out[1], model_out[2])
            elif len(model_out) >= 2:
                feat_mlp, logits = (model_out[0], model_out[1])
            else:
                logits = model_out[0]
                feat_mlp = None
        else:
            logits = model_out
            feat_mlp = None
        global_fc_kd_weight = float(getattr(self.conf, 'global_fc_kd_weight', 0.0))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('student logits contain nan/inf')
        ce_loss = self.criterion(logits, target)
        dyn_loss = torch.tensor(0.0, device=logits.device)
        contrast_logits = None
        scl_loss = torch.tensor(0.0, device=logits.device)
        if feat_mlp is not None:
            contrast_logits = self._compute_auxiliary_logits(feat_mlp, target)
            if not torch.isfinite(contrast_logits).all():
                raise FloatingPointError('contrast logits contain nan/inf')
            scl_logits = contrast_logits.float()
            # AGCE backpropagates through the projection head into the backbone.
            scl_loss = self.scl_criterion(scl_logits, target.long())
        global_fc_kd_loss = torch.tensor(0.0, device=logits.device)
        if contrast_logits is not None:
            global_temp = float(getattr(self.conf, 'self_distillation_temperature', 4.0))
            global_temp = global_temp if global_temp > 0 else 1.0
            student_logp = F.log_softmax(logits.float() / global_temp, dim=-1)
            # Detach only the KD target; AGCE still trains the shared representation.
            teacher_scl_logits = contrast_logits.detach()
            teacher_logp = F.log_softmax(teacher_scl_logits.float() / global_temp, dim=-1)
            global_fc_kd_loss = F.kl_div(student_logp, teacher_logp, reduction='batchmean', log_target=True) * global_temp ** 2
        use_feddyn = bool(getattr(self.conf, 'use_feddyn', False))
        if use_feddyn and self.dual_variable_list is not None and (self.regular_params_list is not None):
            alpha = self._get_feddyn_alpha()
            if alpha > 0:
                for p, d, r in zip(self.model.parameters(), self.dual_variable_list, self.regular_params_list):
                    # The server history is unscaled; apply its coefficient once.
                    dyn_loss += -alpha * torch.sum(p * d)
        ce_weight = float(getattr(self.conf, 'ce_weight', 1.0))
        scl_weight = float(getattr(self.conf, 'scl_weight', self.conf.proco_alpha))
        kd_warmup = int(getattr(self.conf, 'kd_warmup_rounds', 0))
        scl_warmup = int(getattr(self.conf, 'scl_warmup_rounds', 0))
        if curr_round <= kd_warmup or not self._fedga_enable_aux:
            global_fc_kd_coeff = 0.0
        else:
            global_fc_kd_coeff = global_fc_kd_weight
        scl_coeff = scl_weight if curr_round > scl_warmup else 0.0
        scl_coeff = 0.0 if curr_round <= scl_warmup else scl_coeff
        if not self._fedga_enable_aux:
            loss = ce_weight * ce_loss + global_fc_kd_coeff * global_fc_kd_loss
        else:
            loss = ce_weight * ce_loss + scl_coeff * scl_loss + global_fc_kd_coeff * global_fc_kd_loss
        if use_feddyn:
            loss = loss + dyn_loss
        if not torch.isfinite(loss).item() or abs(loss.detach().item()) > 1000.0:
            raise FloatingPointError(f'loss diverges with value={loss.detach().item()}')
        self._last_loss_breakdown = {'ce_loss': float(ce_loss.detach().item()), 'scl_loss': float(scl_loss.detach().item()), 'kd_loss': 0.0, 'global_fc_kd_loss': float(global_fc_kd_loss.detach().item()), 'dyn_loss': float(dyn_loss.detach().item())}
        performance = self.metrics.evaluate(loss, logits, target)
        if self.tracker is not None and update_tracker:
            self.tracker.update_metrics([loss.item()] + performance, n_samples=data_batch['input'].size(0))
        return (loss, logits)

    def _send_model_to_master(self):
        import torch.distributed as dist
        from pcode.utils.tensor_buffer import TensorBuffer
        dist.barrier()
        try:
            cpu_params = [p.clone().detach().cpu() for p in self.model.state_dict().values()]
            flatten_model = TensorBuffer(cpu_params, use_cuda=False)
            send_buffer = flatten_model.buffer.contiguous()
            dist.send(tensor=send_buffer, dst=0)
        except RuntimeError as e:
            self.conf.logger.log(f'Worker {dist.get_rank()} failed to send model buffer to master: {e}')
            return
        dist.barrier()
        self.conf.logger.log(f'Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sent the local model ({self.arch}) to Master.')
