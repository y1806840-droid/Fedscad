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
from pcode.shared_anchors import SharedAnchorScores
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.workers.worker_base import WorkerBase

class WorkerFedSCAD(WorkerBase):

    def __init__(self, conf):
        super().__init__(conf)
        self.num_classes = 10 if self.conf.data == 'cifar10' else 100 if self.conf.data == 'cifar100' else 200 if 'tiny_imagenet' in self.conf.data else 1000
        anchor_init = str(getattr(self.conf, 'anchor_init', 'etf')).lower()
        self.anchor_scores = SharedAnchorScores(feature_dim=self.conf.feat_dim, temperature=self.conf.anchor_temperature, num_classes=self.num_classes, dtype_str=getattr(self.conf, 'estimator_dtype', 'float32'), anchor_init=anchor_init, anchor_seed=getattr(self.conf, 'manual_seed', 0))
        if hasattr(self.conf, 'logger'):
            init_detail = ' (adjacent classes share anchors)' if anchor_init in ('paired_class', 'paired') else ''
            self.conf.logger.log(f'[{conf.algo}] auxiliary class-reference init={anchor_init}{init_detail}')
        self.agce_criterion = nn.CrossEntropyLoss(reduction='mean')
        self._seen_rounds = 0
        self._enable_aux = False
        self._last_loss_breakdown = {}
        self._logged_anchor_clients = set()

    def _get_hc_weight(self):
        base_alpha = float(getattr(self.conf, 'hc_weight', 0.001))
        return max(base_alpha, 0.0)

    def _recv_model_from_master(self):
        ignore_global_mlp = False
        local_mlp_state = {}
        dist.recv(self.model_tb.buffer, src=0)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        use_hc = bool(getattr(self.conf, 'use_hc', False))
        self.dual_variable = None
        self.regular_params = None
        self.dual_variable_list = None
        self.regular_params_list = None
        if use_hc:
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
        if use_hc:
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
        self._enable_aux = self._seen_rounds > 0
        self._seen_rounds += 1
        client_anchor_info = None
        if self.anchor_scores is not None:
            self.anchor_scores.round_reset()
            self.anchor_scores.estimator.Ave.fill_(0)
            self.anchor_scores.estimator.Amount.fill_(0)
            client_anchor_info = self.anchor_scores.set_client_specific_anchors(self.conf.graph.client_id)
        if client_anchor_info is not None and self.conf.graph.client_id not in self._logged_anchor_clients and hasattr(self.conf, 'logger'):
            permutation = client_anchor_info['permutation']
            signature = client_anchor_info['signature']
            if permutation is not None:
                detail = 'permutation=' + str(permutation[:min(10, permutation.numel())].tolist())
            else:
                detail = 'anchor_signature=' + str([round(float(value), 4) for value in signature.tolist()])
            self.conf.logger.log(f"[FedSCAD] client-{self.conf.graph.client_id} fixed {client_anchor_info['mode']} {detail}")
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
        self.agce_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        if self.anchor_scores is not None:
            self.anchor_scores = self.anchor_scores.to(device)
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
                    self.conf.logger.log(f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) phase=joint, loss_total={loss.detach().item():.6f}, loss_ce={loss_info.get('ce_loss', 0.0):.6f}, loss_agce={loss_info.get('agce_loss', 0.0):.6f}, loss_kd={loss_info.get('score_kd_loss', 0.0):.6f}, loss_hc={loss_info.get('hc_loss', 0.0):.6f}, lr={curr_lr:.6f}")
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
        round. Server-side client history and shared-anchor state are untouched.
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
        return self.anchor_scores(features, target, update_features=features.detach())

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
        kd_weight = float(getattr(self.conf, 'kd_weight', 0.0))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('student logits contain nan/inf')
        ce_loss = self.criterion(logits, target)
        hc_loss = torch.tensor(0.0, device=logits.device)
        anchor_logits = None
        agce_loss = torch.tensor(0.0, device=logits.device)
        if feat_mlp is not None:
            anchor_logits = self._compute_auxiliary_logits(feat_mlp, target)
            if not torch.isfinite(anchor_logits).all():
                raise FloatingPointError('anchor logits contain nan/inf')
            agce_logits = anchor_logits.float()
            # AGCE backpropagates through the projection head into the backbone.
            agce_loss = self.agce_criterion(agce_logits, target.long())
        score_kd_loss = torch.tensor(0.0, device=logits.device)
        if anchor_logits is not None:
            global_temp = float(getattr(self.conf, 'kd_temperature', 4.0))
            global_temp = global_temp if global_temp > 0 else 1.0
            student_logp = F.log_softmax(logits.float() / global_temp, dim=-1)
            # Detach only the KD target; AGCE still trains the shared representation.
            teacher_anchor_logits = anchor_logits.detach()
            teacher_logp = F.log_softmax(teacher_anchor_logits.float() / global_temp, dim=-1)
            score_kd_loss = F.kl_div(student_logp, teacher_logp, reduction='batchmean', log_target=True) * global_temp ** 2
        use_hc = bool(getattr(self.conf, 'use_hc', False))
        if use_hc and self.dual_variable_list is not None and (self.regular_params_list is not None):
            alpha = self._get_hc_weight()
            if alpha > 0:
                for p, d, r in zip(self.model.parameters(), self.dual_variable_list, self.regular_params_list):
                    # The server history is unscaled; apply its coefficient once.
                    hc_loss += -alpha * torch.sum(p * d)
        ce_weight = float(getattr(self.conf, 'ce_weight', 1.0))
        agce_weight = float(self.conf.agce_weight)
        kd_warmup = int(getattr(self.conf, 'kd_warmup_rounds', 0))
        agce_warmup = int(getattr(self.conf, 'agce_warmup_rounds', 0))
        if curr_round <= kd_warmup or not self._enable_aux:
            kd_coeff = 0.0
        else:
            kd_coeff = kd_weight
        agce_coeff = agce_weight if curr_round > agce_warmup else 0.0
        agce_coeff = 0.0 if curr_round <= agce_warmup else agce_coeff
        if not self._enable_aux:
            loss = ce_weight * ce_loss + kd_coeff * score_kd_loss
        else:
            loss = ce_weight * ce_loss + agce_coeff * agce_loss + kd_coeff * score_kd_loss
        if use_hc:
            loss = loss + hc_loss
        if not torch.isfinite(loss).item() or abs(loss.detach().item()) > 1000.0:
            raise FloatingPointError(f'loss diverges with value={loss.detach().item()}')
        self._last_loss_breakdown = {'ce_loss': float(ce_loss.detach().item()), 'agce_loss': float(agce_loss.detach().item()), 'score_kd_loss': float(score_kd_loss.detach().item()), 'hc_loss': float(hc_loss.detach().item())}
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
