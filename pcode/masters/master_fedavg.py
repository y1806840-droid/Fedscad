# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import copy
import functools
import os
import re
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode import create_dataset
from pcode.masters.master_base import MasterBase
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.logging import save_average_test_stat

class MasterFedAvg(MasterBase):

    def __init__(self, conf):
        super().__init__(conf)
        assert self.conf.fl_aggregate is not None
        if self.conf.prepare_data == 'combine' and self.conf.personal_test:
            self.test_loaders = [self.local_test_loaders[1]]
            print('测试数据数量', len(self.test_loaders[0].dataset))
        self.aggregator = create_aggregator.Aggregator(conf, model=self.master_model, criterion=self.criterion, metrics=self.metrics, dataset=self.dataset, test_loaders=self.test_loaders, clientid2arch=self.clientid2arch)
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.logger.log(f'Master initialized the aggregator/coordinator.')
        self.early_stopping_tracker = EarlyStoppingTracker(patience=conf.early_stopping_rounds)
        self.local_models = dict(((client_id, copy.deepcopy(self.master_model)) for client_id in range(1, 1 + conf.n_clients)))
        self.GM_client_coordinators = {}
        for client_id in self.client_ids:
            self.GM_client_coordinators[client_id] = create_coordinator.Coordinator(conf, self.metrics)
        self.GM_curr_personal_perfs = {}
        self.GM_personal_avg_coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    def run(self):
        flatten_local_models = None
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(f'Master starting one round of federated learning: (comm_round={comm_round}).')
            list_of_local_n_epochs = master_utils.get_n_local_epoch(conf=self.conf, n_participated=self.conf.n_participated)
            self.list_of_local_n_epochs = list_of_local_n_epochs
            selected_client_ids = self._random_select_clients()
            self._check_early_stopping()
            self._activate_selected_clients(selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs)
            if not self.conf.is_finished:
                self._send_model_to_selected_clients(selected_client_ids)
            else:
                dist.barrier()
                self.conf.logger.log(f'Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})')
                return
            flatten_local_models = self._receive_models_from_selected_clients(selected_client_ids)
            self.update_local_models(selected_client_ids, flatten_local_models)
            recovered_local_models = self.update_local_models(selected_client_ids, flatten_local_models)
            self._maybe_probe_feature_similarity(selected_client_ids, recovered_local_models)
            self._aggregate_model_and_evaluate(flatten_local_models)
            self._update_GM_client_performance()
            self.conf.logger.log(f'Master finished one round of federated learning.\n')
        dist.barrier()
        self._finishing()

    def _avg_over_archs(self, flatten_local_models):
        archs = set([self.clientid2arch[client_idx] for client_idx in flatten_local_models.keys()])
        archs_fedavg_models = {}
        for arch in archs:
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model
            self.conf.logger.log(f'Master uniformly average over {len(_flatten_local_models)} received models ({arch}).')
            fedavg_model = self.aggregator.aggregate(master_model=self.master_model, client_models=self.client_models, flatten_local_models=_flatten_local_models, aggregate_fn_name='_s1_federated_average')
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        same_arch = len(self.client_models) == 1 and self.conf.arch_info['master'] == self.conf.arch_info['worker'][0]
        fedavg_models = self._avg_over_archs(flatten_local_models)
        if same_arch:
            fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None
        if self.aggregator.aggregate_fn is not None:
            if fedavg_model is not None:
                performance = master_utils.get_avg_perf_on_dataloaders(self.conf, self.coordinator, fedavg_model, self.criterion, self.metrics, self.test_loaders, label=f'fedavg_test_loader')
            else:
                assert 'knowledge_transfer' in self.conf.fl_aggregate['scheme']
                performance = None
                for _arch, _fedavg_model in fedavg_models.items():
                    master_utils.get_avg_perf_on_dataloaders(self.conf, self.coordinator, _fedavg_model, self.criterion, self.metrics, self.test_loaders, label=f'fedavg_test_loader_{_arch}')
            client_models = self.aggregator.aggregate(master_model=self.master_model, client_models=self.client_models, fedavg_model=fedavg_model, fedavg_models=fedavg_models, flatten_local_models=flatten_local_models, performance=performance)
            if same_arch:
                self.master_model.load_state_dict(list(client_models.values())[0].state_dict())
            for arch, _client_model in client_models.items():
                self.client_models[arch].load_state_dict(_client_model.state_dict())
        else:
            if same_arch:
                self.master_model.load_state_dict(fedavg_model.state_dict())
            for arch, _fedavg_model in fedavg_models.items():
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())
        if same_arch:
            master_utils.do_validation(self.conf, self.coordinator, self.master_model, self.criterion, self.metrics, self.test_loaders, label=f'aggregated_test_loader')
        else:
            for arch, _client_model in self.client_models.items():
                master_utils.do_validation(self.conf, self.coordinator, _client_model, self.criterion, self.metrics, self.test_loaders, label=f'aggregated_test_loader_{arch}')

    def _update_GM_client_performance(self):
        if not self.conf.personal_test:
            return
        for client_id in self.client_ids:
            self.GM_curr_personal_perfs[client_id] = master_utils.do_validation_personal(self.conf, self.GM_client_coordinators[client_id], self.client_models[self.clientid2arch[client_id]], self.criterion, self.metrics, [self.local_test_loaders[client_id]], label=f'GM_local_test_loader_client_{client_id}')
        curr_perf = []
        for curr_personal_perf in self.GM_curr_personal_perfs.values():
            curr_perf.append(curr_personal_perf)
        curr_perf = functools.reduce(lambda a, b: a + b, curr_perf) / len(curr_perf)
        save_average_test_stat(self.conf, curr_perf.dictionary, type='GM_average_test')
        self.GM_personal_avg_coordinator.update_perf(curr_perf)
        for name, best_tracker in self.GM_personal_avg_coordinator.best_trackers.items():
            self.conf.logger.log('GM Personal_test method 2                 -- GM Personal best avg results of {}: \t {} best comm round {:.3f}, current comm_roud {:.3f}'.format(name, best_tracker.best_perf, best_tracker.get_best_perf_loc, self.conf.graph.comm_round))

    def update_local_models(self, selected_client_ids, flatten_local_models):
        _, local_models = master_utils.recover_models(self.conf, self.client_models, flatten_local_models)
        for selected_client_id in selected_client_ids:
            self.local_models[selected_client_id].load_state_dict(local_models[selected_client_id].state_dict())
        return local_models

    def _get_feature_similarity_probe_rounds(self):
        rounds = getattr(self.conf, 'feature_similarity_probe_rounds', None)
        if rounds is None or str(rounds).strip() == '':
            return set()
        return {int(item) for item in re.split('[,;:\\s]+', str(rounds).strip()) if item}

    def _get_feature_similarity_classes(self, local_models):
        classes = getattr(self.conf, 'feature_similarity_classes', None)
        if classes is not None and str(classes).strip() != '':
            return [int(item.strip()) for item in str(classes).split(',') if item.strip()]
        sample_model = next(iter(local_models.values()))
        num_classes = getattr(sample_model, 'num_classes', None)
        if num_classes is None:
            if self.conf.data in ('cifar10', 'svhn'):
                num_classes = 10
            elif self.conf.data == 'cifar100':
                num_classes = 100
            elif 'tiny_imagenet' in self.conf.data:
                num_classes = 200
            else:
                num_classes = 1000
        return list(range(min(10, int(num_classes))))

    def _get_feature_similarity_loader(self):
        if hasattr(self, '_feature_similarity_loader'):
            return self._feature_similarity_loader
        if self.conf.prepare_data == 'combine' and isinstance(self.dataset, dict):
            dataset = self.dataset.get('server_test', None)
            if dataset is None:
                raise RuntimeError('feature similarity probe needs server_test in combine mode.')
            loader, _ = create_dataset.define_data_loader(self.conf, dataset=dataset, is_train=False, shuffle=False)
        elif len(self.test_loaders) > 0:
            loader = self.test_loaders[0]
        else:
            raise RuntimeError('feature similarity probe needs a server test loader.')
        self._feature_similarity_loader = loader
        return loader

    def _extract_probe_outputs(self, model, inputs):
        old_return_features = getattr(model, 'return_features', None)
        if old_return_features is not None:
            model.return_features = True
        try:
            output = model(inputs)
        finally:
            if old_return_features is not None:
                model.return_features = old_return_features
        result = {'projection': None, 'projection_raw': None, 'logits': None, 'pre_classifier': None, 'pre_classifier_raw': None}
        if isinstance(output, (tuple, list)):
            if len(output) > 0 and torch.is_tensor(output[0]):
                result['projection'] = output[0]
            if len(output) > 1 and torch.is_tensor(output[1]):
                result['logits'] = output[1]
            if len(output) > 2 and torch.is_tensor(output[2]):
                result['pre_classifier'] = output[2]
        elif torch.is_tensor(output):
            result['logits'] = output
            activations = getattr(model, 'activations', None)
            if activations:
                result['pre_classifier'] = activations[-1]
        else:
            raise RuntimeError('feature similarity probe received an unsupported model output.')
        for key in ('projection', 'pre_classifier'):
            features = result[key]
            if features is not None:
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                features = features.detach()
                result[f'{key}_raw'] = features
                result[key] = F.normalize(features, dim=1)
        if result['logits'] is not None:
            logits = result['logits']
            if logits.dim() > 2:
                logits = logits.view(logits.size(0), -1)
            result['logits'] = logits.detach()
        return result

    def _select_probe_features(self, outputs, raw=False):
        feature_source = str(getattr(self.conf, 'feature_similarity_feature_source', 'pre_classifier')).lower()
        suffix = '_raw' if raw else ''
        if feature_source == 'projection':
            features = outputs[f'projection{suffix}']
        elif feature_source in ('pre_classifier', 'auto'):
            features = outputs[f'pre_classifier{suffix}']
            if features is None:
                features = outputs[f'projection{suffix}']
                if not raw:
                    self.conf.logger.log('Feature similarity probe could not read pre_classifier features; falling back to projection features.')
        else:
            raise RuntimeError(f'Unknown feature source: {feature_source}')
        if features is None:
            raise RuntimeError(f'Feature similarity probe could not extract {feature_source} features.')
        return features

    def _extract_probe_features(self, model, inputs):
        return self._select_probe_features(self._extract_probe_outputs(model, inputs))

    def _get_probe_shared_anchors(self, model, projection_dim, num_classes, device):
        """Recreate the same fixed reference used by the client."""
        from pcode.shared_anchors import _etf_matrix, _paired_class_matrix, _random_unit_matrix
        mode = str(getattr(self.conf, 'anchor_init', 'etf')).lower()
        if mode == 'random':
            anchors = _random_unit_matrix(
                num_classes, projection_dim, torch.float32, device,
                seed=int(getattr(self.conf, 'manual_seed', 0)),
            )
        elif mode == 'paired_class':
            anchors = _paired_class_matrix(num_classes, projection_dim, torch.float32, device)
        elif mode == 'etf':
            anchors = _etf_matrix(num_classes, projection_dim, torch.float32, device)
        else:
            raise ValueError('Unsupported shared-anchor initialization')
        temperature = float(getattr(self.conf, 'anchor_temperature', 0.2))
        return F.normalize(anchors, dim=1), 1.0 / temperature, 'fedscad'

    @staticmethod
    def _pack_probe_sample_records(sample_records, client_ids):
        if not sample_records:
            return {}
        max_samples = max((int(sample_records[client_id]['labels'].shape[0]) for client_id in client_ids))
        packed = {'sample_mask': np.zeros((len(client_ids), max_samples), dtype=bool), 'sample_indices': np.full((len(client_ids), max_samples), -1, dtype=np.int64), 'sample_labels': np.full((len(client_ids), max_samples), -1, dtype=np.int64)}
        for client_pos, client_id in enumerate(client_ids):
            record = sample_records[client_id]
            n_samples = int(record['labels'].shape[0])
            packed['sample_mask'][client_pos, :n_samples] = True
            packed['sample_indices'][client_pos, :n_samples] = record['indices'].numpy()
            packed['sample_labels'][client_pos, :n_samples] = record['labels'].numpy()
        for key in ('projection_features', 'projection_features_raw', 'pre_classifier_features', 'pre_classifier_features_raw', 'logits', 'anchor_logits'):
            tensors = [sample_records[client_id].get(key) for client_id in client_ids]
            if any((tensor is None for tensor in tensors)):
                continue
            tail_shape = tuple(tensors[0].shape[1:])
            if any((tuple(tensor.shape[1:]) != tail_shape for tensor in tensors)):
                continue
            dtype = tensors[0].numpy().dtype
            array = np.full((len(client_ids), max_samples) + tail_shape, np.nan, dtype=dtype)
            for client_pos, tensor in enumerate(tensors):
                array[client_pos, :tensor.shape[0]] = tensor.numpy()
            packed[f'sample_{key}'] = array
        return packed

    def _compute_client_class_centers(self, local_models, client_ids, class_ids, loader):
        device = torch.device('cuda' if self.conf.graph.on_cuda else 'cpu')
        centers = {}
        mean_features = {}
        within_class_scatter = {}
        raw_mean_features = {}
        raw_within_class_scatter = {}
        counts = {}
        sample_records = {}
        etf_anchors = {}
        etf_logit_scales = {}
        etf_anchor_types = {}
        max_batches = int(getattr(self.conf, 'feature_similarity_max_batches', 0) or 0)
        save_samples = bool(getattr(self.conf, 'feature_similarity_save_samples', True))
        samples_per_class = int(getattr(self.conf, 'feature_similarity_samples_per_class', 200) or 0)
        sample_dtype_name = str(getattr(self.conf, 'feature_similarity_sample_dtype', 'float16')).lower()
        sample_dtype = torch.float16 if sample_dtype_name == 'float16' else torch.float32
        for client_id in client_ids:
            model = local_models[client_id].to(device)
            model.eval()
            sums = None
            raw_sums = None
            raw_squared_norm_sums = None
            class_counts = torch.zeros(len(class_ids), dtype=torch.long, device=device)
            selected_counts = torch.zeros(len(class_ids), dtype=torch.long, device=device)
            record_chunks = {'indices': [], 'labels': [], 'projection_features': [], 'projection_features_raw': [], 'pre_classifier_features': [], 'pre_classifier_features_raw': [], 'logits': [], 'anchor_logits': []}
            sample_offset = 0
            client_anchors = None
            client_anchor_scale = float('nan')
            client_anchor_type = 'unavailable'
            with torch.no_grad():
                for batch_idx, (_input, _target) in enumerate(loader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    data_batch = create_dataset.load_data_batch(self.conf, _input, _target, is_training=False)
                    outputs = self._extract_probe_outputs(model, data_batch['input'])
                    features = self._select_probe_features(outputs)
                    raw_features = self._select_probe_features(outputs, raw=True)
                    targets = data_batch['target']
                    if sums is None:
                        sums = torch.zeros(len(class_ids), features.size(1), dtype=features.dtype, device=device)
                        raw_sums = torch.zeros(len(class_ids), raw_features.size(1), dtype=raw_features.dtype, device=device)
                        raw_squared_norm_sums = torch.zeros(len(class_ids), dtype=raw_features.dtype, device=device)
                        projection = outputs['projection']
                        logits = outputs['logits']
                        if projection is not None and logits is not None:
                            client_anchors, client_anchor_scale, client_anchor_type = self._get_probe_shared_anchors(model, projection.size(1), logits.size(1), device)
                    for class_pos, class_id in enumerate(class_ids):
                        mask = targets == class_id
                        if mask.any():
                            sums[class_pos].add_(features[mask].sum(dim=0))
                            raw_sums[class_pos].add_(raw_features[mask].sum(dim=0))
                            raw_squared_norm_sums[class_pos].add_(raw_features[mask].pow(2).sum(dim=1).sum())
                            class_counts[class_pos] += int(mask.sum().item())
                    if save_samples:
                        selected = torch.zeros_like(targets, dtype=torch.bool)
                        for class_pos, class_id in enumerate(class_ids):
                            class_indices = (targets == class_id).nonzero(as_tuple=False).view(-1)
                            if samples_per_class > 0:
                                remaining = max(0, samples_per_class - int(selected_counts[class_pos].item()))
                                class_indices = class_indices[:remaining]
                            if class_indices.numel() > 0:
                                selected[class_indices] = True
                                selected_counts[class_pos] += class_indices.numel()
                        if selected.any():
                            batch_indices = torch.arange(sample_offset, sample_offset + targets.size(0), device=targets.device)
                            record_chunks['indices'].append(batch_indices[selected].cpu())
                            record_chunks['labels'].append(targets[selected].long().cpu())
                            for key in ('projection', 'projection_raw', 'pre_classifier', 'pre_classifier_raw', 'logits'):
                                tensor = outputs[key]
                                if tensor is not None:
                                    record_key = key if key == 'logits' else f"{key.replace('_raw', '')}_features" + ('_raw' if key.endswith('_raw') else '')
                                    record_chunks[record_key].append(tensor[selected].to(sample_dtype).cpu())
                            if client_anchors is not None and outputs['projection'] is not None:
                                anchor_logits = outputs['projection'][selected].float().mm(client_anchors.t()).mul(client_anchor_scale)
                                record_chunks['anchor_logits'].append(anchor_logits.to(sample_dtype).cpu())
                    sample_offset += targets.size(0)
            if sums is None:
                raise RuntimeError('feature similarity probe received no batches.')
            non_empty = class_counts > 0
            client_means = torch.full_like(sums, float('nan'))
            client_means[non_empty] = sums[non_empty] / class_counts[non_empty].float().unsqueeze(1)
            client_centers = client_means.clone()
            client_centers[non_empty] = F.normalize(client_centers[non_empty], dim=1)
            client_scatter = torch.full((len(class_ids),), float('nan'), dtype=sums.dtype, device=device)
            client_scatter[non_empty] = (1.0 - client_means[non_empty].pow(2).sum(dim=1)).clamp_min(0.0)
            client_raw_means = torch.full_like(raw_sums, float('nan'))
            client_raw_means[non_empty] = raw_sums[non_empty] / class_counts[non_empty].float().unsqueeze(1)
            client_raw_scatter = torch.full((len(class_ids),), float('nan'), dtype=raw_sums.dtype, device=device)
            client_raw_scatter[non_empty] = (raw_squared_norm_sums[non_empty] / class_counts[non_empty].float() - client_raw_means[non_empty].pow(2).sum(dim=1)).clamp_min(0.0)
            centers[client_id] = client_centers.cpu()
            mean_features[client_id] = client_means.cpu()
            within_class_scatter[client_id] = client_scatter.cpu()
            raw_mean_features[client_id] = client_raw_means.cpu()
            raw_within_class_scatter[client_id] = client_raw_scatter.cpu()
            counts[client_id] = class_counts.cpu()
            if save_samples:
                sample_records[client_id] = {key: torch.cat(chunks, dim=0) if chunks else None for key, chunks in record_chunks.items()}
                if sample_records[client_id]['labels'] is None:
                    sample_records[client_id]['labels'] = torch.empty(0, dtype=torch.long)
                    sample_records[client_id]['indices'] = torch.empty(0, dtype=torch.long)
            etf_anchors[client_id] = client_anchors.cpu() if client_anchors is not None else None
            etf_logit_scales[client_id] = client_anchor_scale
            etf_anchor_types[client_id] = client_anchor_type
            model.cpu()
        probe_data = {'mean_features': mean_features, 'within_class_scatter': within_class_scatter, 'raw_mean_features': raw_mean_features, 'raw_within_class_scatter': raw_within_class_scatter, 'samples': self._pack_probe_sample_records(sample_records, client_ids), 'etf_anchors': etf_anchors, 'etf_logit_scales': etf_logit_scales, 'etf_anchor_types': etf_anchor_types, 'sample_dtype': sample_dtype_name, 'samples_per_class': samples_per_class}
        return (centers, counts, probe_data)

    def _build_feature_similarity_matrices(self, centers, client_ids, class_ids):
        n_clients = len(client_ids)
        similarities = torch.full((len(class_ids), n_clients, n_clients), float('nan'))
        for class_pos, _ in enumerate(class_ids):
            class_centers = torch.stack([centers[client_id][class_pos] for client_id in client_ids], dim=0)
            valid = ~torch.isnan(class_centers).any(dim=1)
            if valid.any():
                valid_centers = class_centers[valid]
                valid_sim = valid_centers.mm(valid_centers.t())
                valid_idx = valid.nonzero(as_tuple=False).view(-1)
                for row_pos, row_idx in enumerate(valid_idx):
                    for col_pos, col_idx in enumerate(valid_idx):
                        similarities[class_pos, row_idx, col_idx] = valid_sim[row_pos, col_pos]
        similarities = similarities.numpy()
        avg_similarity = np.nanmean(similarities, axis=0)
        return (similarities, avg_similarity)

    def _save_feature_similarity_plot(self, similarities, avg_similarity, client_ids, class_ids, out_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n_panels = len(class_ids) + 1
        n_cols = min(5, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), dpi=220)
        axes = np.array(axes).reshape(-1)
        panels = [(f'class {class_id}', similarities[idx]) for idx, class_id in enumerate(class_ids)]
        panels.append(('class avg', avg_similarity))
        im = None
        for ax, (title, matrix) in zip(axes, panels):
            im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap='viridis')
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(client_ids)))
            ax.set_yticks(range(len(client_ids)))
            ax.set_xticklabels(client_ids, rotation=90, fontsize=6)
            ax.set_yticklabels(client_ids, fontsize=6)
            ax.set_xlabel('client', fontsize=7)
            ax.set_ylabel('client', fontsize=7)
        for ax in axes[len(panels):]:
            ax.axis('off')
        fig.suptitle(f'{self._feature_probe_algo_name().upper()} Cross-client Same-class Feature Similarity of Local Models, round {self.conf.graph.comm_round}', fontsize=11)
        fig.colorbar(im, ax=axes[:len(panels)].tolist(), fraction=0.025, pad=0.02)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def _save_feature_center_tsne_plot(self, centers_array, client_ids, class_ids, out_path):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            self.conf.logger.log('Feature similarity t-SNE skipped: scikit-learn is not installed.')
            return False
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n_clients, n_classes, feat_dim = centers_array.shape
        features = centers_array.reshape(n_clients * n_classes, feat_dim)
        labels = np.tile(np.array(class_ids), n_clients)
        clients = np.repeat(np.array(client_ids), n_classes)
        finite_mask = np.isfinite(features).all(axis=1)
        features = features[finite_mask]
        labels = labels[finite_mask]
        clients = clients[finite_mask]
        if features.shape[0] < 4:
            self.conf.logger.log('Feature similarity t-SNE skipped: fewer than 4 valid centers.')
            return False
        perplexity = float(getattr(self.conf, 'feature_similarity_tsne_perplexity', 10.0))
        perplexity = max(2.0, min(perplexity, float(features.shape[0] - 1)))
        tsne_kwargs = dict(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=int(getattr(self.conf, 'manual_seed', 6)))
        try:
            embedding = TSNE(**tsne_kwargs).fit_transform(features)
        except TypeError:
            tsne_kwargs.pop('learning_rate', None)
            embedding = TSNE(**tsne_kwargs).fit_transform(features)
        fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=260)
        cmap = plt.get_cmap('tab10' if len(class_ids) <= 10 else 'tab20')
        class_to_color = {class_id: cmap(idx % cmap.N) for idx, class_id in enumerate(class_ids)}
        point_size = float(getattr(self.conf, 'feature_similarity_tsne_point_size', 70.0))
        point_alpha = float(getattr(self.conf, 'feature_similarity_tsne_alpha', 0.88))
        annotate_clients = bool(getattr(self.conf, 'feature_similarity_tsne_annotate_clients', False))
        for class_id in class_ids:
            mask = labels == class_id
            if not mask.any():
                continue
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=point_size, marker='o', c=[class_to_color[class_id]], edgecolors='black', linewidths=0.45, alpha=point_alpha)
        if annotate_clients:
            for point_idx, client_id in enumerate(clients):
                ax.text(embedding[point_idx, 0], embedding[point_idx, 1], str(client_id), fontsize=5, ha='center', va='center', color='black')
        class_handles = [plt.Line2D([0], [0], marker='o', linestyle='', markerfacecolor=class_to_color[class_id], markeredgecolor='black', markersize=6, label=f'class {class_id}') for class_id in class_ids]
        ax.legend(handles=class_handles, title='Class', loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7, title_fontsize=8)
        ax.set_title(f'{self._feature_probe_algo_name().upper()} t-SNE of Cross-client Same-class Feature Centers, round {self.conf.graph.comm_round}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return True

    def _maybe_probe_feature_similarity(self, selected_client_ids, local_models):
        if int(self.conf.graph.comm_round) not in self._get_feature_similarity_probe_rounds():
            return
        try:
            self._run_feature_similarity_probe(selected_client_ids, local_models)
        except Exception:
            self.conf.logger.log('Feature similarity probe failed and was skipped:\n' + traceback.format_exc())

    def _feature_probe_algo_name(self):
        return str(getattr(self.conf, 'algo', 'fedavg')).lower()

    def _run_feature_similarity_probe(self, selected_client_ids, local_models):
        if len(local_models) == 0:
            self.conf.logger.log('Feature similarity probe skipped: no local models.')
            return
        n_clients = int(getattr(self.conf, 'feature_similarity_n_clients', 10))
        client_ids = [client_id for client_id in selected_client_ids if client_id in local_models]
        client_ids = client_ids[:min(n_clients, len(client_ids))]
        if len(client_ids) < 2:
            self.conf.logger.log('Feature similarity probe skipped: fewer than 2 clients.')
            return
        class_ids = self._get_feature_similarity_classes(local_models)
        loader = self._get_feature_similarity_loader()
        self.conf.logger.log(f'Feature similarity probe starts: round={self.conf.graph.comm_round}, clients={client_ids}, classes={class_ids}.')
        centers, counts, probe_data = self._compute_client_class_centers(local_models, client_ids, class_ids, loader)
        global_centers = None
        global_counts = None
        global_probe_data = None
        global_probe_id = -1
        try:
            global_centers, global_counts, global_probe_data = self._compute_client_class_centers({global_probe_id: self.master_model}, [global_probe_id], class_ids, loader)
        except Exception:
            self.conf.logger.log('Global teacher snapshot could not be added to feature probe npz:\n' + traceback.format_exc())
        similarities, avg_similarity = self._build_feature_similarity_matrices(centers, client_ids, class_ids)
        out_dir = getattr(self.conf, 'feature_similarity_out_dir', None)
        if out_dir is None or str(out_dir).strip() == '':
            out_dir = os.path.join(self.conf.checkpoint_root, 'feature_similarity')
        os.makedirs(out_dir, exist_ok=True)
        round_id = int(self.conf.graph.comm_round)
        prefix = f'{self._feature_probe_algo_name()}_cross_client_same_class_feature_similarity_round{round_id}'
        npz_path = os.path.join(out_dir, f'{prefix}.npz')
        png_path = os.path.join(out_dir, f'{prefix}.png')
        tsne_path = os.path.join(out_dir, f'{prefix}_tsne.png')
        centers_array = np.stack([centers[client_id].numpy() for client_id in client_ids], axis=0)
        counts_array = np.stack([counts[client_id].numpy() for client_id in client_ids], axis=0)
        npz_payload = {'schema_version': np.array([2], dtype=np.int64), 'clients': np.array(client_ids), 'classes': np.array(class_ids), 'centers': centers_array, 'mean_features': np.stack([probe_data['mean_features'][client_id].numpy() for client_id in client_ids], axis=0), 'within_class_scatter': np.stack([probe_data['within_class_scatter'][client_id].numpy() for client_id in client_ids], axis=0), 'raw_mean_features': np.stack([probe_data['raw_mean_features'][client_id].numpy() for client_id in client_ids], axis=0), 'raw_within_class_scatter': np.stack([probe_data['raw_within_class_scatter'][client_id].numpy() for client_id in client_ids], axis=0), 'counts': counts_array, 'similarities': similarities, 'avg_similarity': avg_similarity, 'round': np.array([round_id]), 'algorithm': np.array([self._feature_probe_algo_name()]), 'feature_source': np.array([str(getattr(self.conf, 'feature_similarity_feature_source', 'pre_classifier'))]), 'sample_dtype': np.array([probe_data['sample_dtype']]), 'samples_per_class_limit': np.array([probe_data['samples_per_class']], dtype=np.int64)}
        npz_payload.update(probe_data['samples'])
        anchors = [probe_data['etf_anchors'][client_id] for client_id in client_ids]
        if all((anchor is not None for anchor in anchors)):
            anchor_shapes = {tuple(anchor.shape) for anchor in anchors}
            if len(anchor_shapes) == 1:
                npz_payload['etf_anchors'] = np.stack([anchor.numpy() for anchor in anchors], axis=0)
                npz_payload['etf_logit_scales'] = np.array([probe_data['etf_logit_scales'][client_id] for client_id in client_ids], dtype=np.float32)
                npz_payload['etf_anchor_types'] = np.array([probe_data['etf_anchor_types'][client_id] for client_id in client_ids])
        if global_probe_data is not None:
            npz_payload.update({'global_model_role': np.array(['pre_aggregation_teacher_snapshot']), 'global_centers': global_centers[global_probe_id].numpy(), 'global_mean_features': global_probe_data['mean_features'][global_probe_id].numpy(), 'global_within_class_scatter': global_probe_data['within_class_scatter'][global_probe_id].numpy(), 'global_raw_mean_features': global_probe_data['raw_mean_features'][global_probe_id].numpy(), 'global_raw_within_class_scatter': global_probe_data['raw_within_class_scatter'][global_probe_id].numpy(), 'global_counts': global_counts[global_probe_id].numpy()})
            for key, value in global_probe_data['samples'].items():
                npz_payload[f'global_{key}'] = value[0]
            global_anchor = global_probe_data['etf_anchors'][global_probe_id]
            if global_anchor is not None:
                npz_payload['global_etf_anchors'] = global_anchor.numpy()
                npz_payload['global_etf_logit_scale'] = np.array([global_probe_data['etf_logit_scales'][global_probe_id]], dtype=np.float32)
                npz_payload['global_etf_anchor_type'] = np.array([global_probe_data['etf_anchor_types'][global_probe_id]])
        np.savez_compressed(npz_path, **npz_payload)
        saved_paths = [npz_path]
        try:
            self._save_feature_similarity_plot(similarities, avg_similarity, client_ids, class_ids, png_path)
            saved_paths.append(png_path)
        except Exception:
            self.conf.logger.log('Feature similarity heatmap save failed:\n' + traceback.format_exc())
        if bool(getattr(self.conf, 'feature_similarity_save_tsne', True)):
            try:
                if self._save_feature_center_tsne_plot(centers_array, client_ids, class_ids, tsne_path):
                    saved_paths.append(tsne_path)
            except Exception:
                self.conf.logger.log('Feature similarity t-SNE save failed:\n' + traceback.format_exc())
        self.conf.logger.log(f"Feature similarity probe saved artifacts: {', '.join(saved_paths)}.")

    def _check_early_stopping(self):
        meet_flag = False
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0
            if self.coordinator.key_metric.cur_perf is not None and self.coordinator.key_metric.cur_perf > self.conf.target_perf:
                self.conf.logger.log('Master early stopping: meet target perf.')
                self.conf.meet_target = True
                meet_flag = True
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log('Master early stopping: not meet target perf but has no patience.')
                meet_flag = True
        elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
            meet_flag = True
        if meet_flag:
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)
