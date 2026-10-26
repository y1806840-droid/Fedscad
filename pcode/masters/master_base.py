# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import copy
import functools
import os
import numpy as np
import torch
import torch.distributed as dist
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.communication_tracker import DistributedCommunicationTracker
from pcode.utils.logging import save_average_test_stat
from pcode.utils.tensor_buffer import TensorBuffer

class MasterBase(object):

    def __init__(self, conf):
        self.conf = conf
        print('客户端')
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))
        master_arch, self.master_model = create_model.define_model(conf, to_consistent_model=False)
        self.used_client_archs = set((create_model.determine_arch(conf, client_id, use_complex_arch=True) for client_id in range(1, 1 + conf.n_clients)))
        self.conf.used_client_archs = self.used_client_archs
        conf.logger.log(f'客户端将使用架构={self.used_client_archs}。')
        conf.logger.log('主节点已为客户端模型创建模板。')
        self.client_models = dict(((arch, copy.deepcopy(self.master_model)) if arch == master_arch else create_model.define_model(conf, to_consistent_model=False, arch=arch) for arch in self.used_client_archs))
        self.clientid2arch = {client_id: create_model.determine_arch(conf, client_id=client_id, use_complex_arch=True) for client_id in range(1, 1 + conf.n_clients)}
        self.conf.clientid2arch = self.clientid2arch
        conf.logger.log(f'主节点初始化了客户端ID到架构的映射关系: {self.clientid2arch}。')
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        if self.conf.prepare_data == 'combine':
            if getattr(self.conf, 'server_eval_global_test', True) and isinstance(self.dataset, dict) and ('server_test' in self.dataset):
                test_loader, _ = create_dataset.define_data_loader(conf, self.dataset['server_test'], is_train=False)
                self.test_loaders = [test_loader]
            else:
                self.test_loaders = []
            self.local_test_loaders = {}
            if getattr(self.conf, 'server_eval_client_tests', False) and isinstance(self.dataset, dict) and ('client_local' in self.dataset):
                for client_id in self.client_ids:
                    splits = self.dataset['client_local'].get(client_id)
                    if splits is not None:
                        loader, _ = create_dataset.define_data_loader(self.conf, dataset=splits['test'], is_train=False, data_partitioner=None)
                        self.local_test_loaders[client_id] = loader
            conf.logger.log('主节点在 combine 模式下完成测试数据加载器初始化。')
        else:
            _, self.data_partitioner = create_dataset.define_data_loader(self.conf, dataset=self.dataset['train'], localdata_id=0, is_train=True, data_partitioner=None)
            conf.logger.log(f'主节点初始化了本地训练数据加载器。')
            if self.dataset['val'] is not None:
                assert not conf.partitioned_by_user
                self.val_loader, _ = create_dataset.define_data_loader(conf, self.dataset['val'], is_train=False)
                conf.logger.log(f'主节点初始化了验证数据加载器。')
            else:
                self.val_loader = None
            if conf.partitioned_by_user:
                self.test_loaders = []
                for localdata_id in self.client_ids:
                    test_loader, _ = create_dataset.define_data_loader(conf, self.dataset['test'], localdata_id=localdata_id - 1, is_train=False, shuffle=False)
                    self.test_loaders.append(copy.deepcopy(test_loader))
            else:
                test_loader, _ = create_dataset.define_data_loader(conf, self.dataset['test'], is_train=False)
                self.test_loaders = [test_loader]
            if self.conf.personal_test:
                dist.barrier()
                _, self.personal_test_data_partitioner = create_dataset.define_data_loader(conf, self.dataset['test'], localdata_id=0, is_train=False, data_partitioner=None, personal_partition=True, known_distribution=self.data_partitioner.class_distribution)
                self.local_test_loaders = {}
                for client_id in self.client_ids:
                    self.local_test_loaders[client_id], _ = create_dataset.define_data_loader(conf, dataset=self.dataset['test'], localdata_id=client_id - 1, is_train=False, data_partitioner=self.personal_test_data_partitioner, personal_partition=True, known_distribution=self.data_partitioner.class_distribution)
                conf.logger.log(f'主节点初始化了本地测试数据加载器。')
        self.criterion = cross_entropy.CrossEntropyLoss(reduction='mean')
        self.metrics = create_metrics.Metrics(self.master_model, task='classification')
        conf.logger.log(f'主节点初始化了模型、数据集、损失函数和评估指标。')
        if self.conf.personal_test:
            self.client_coordinators = {}
            for client_id in self.client_ids:
                self.client_coordinators[client_id] = create_coordinator.Coordinator(conf, self.metrics)
            self.curr_personal_perfs = {}
            self.personal_avg_coordinator = create_coordinator.Coordinator(conf, self.metrics)
            conf.logger.log(f'主节点为每个客户端初始化了个性化平均协调器。\n')
        self.communication_tracker = None
        if bool(getattr(self.conf, 'track_communication', False)):
            self.communication_tracker = DistributedCommunicationTracker(self.conf)
            self.communication_tracker.install()
            self.conf.logger.log(f'Application-level communication tracking enabled; point-to-point training payload and broadcast control bytes are recorded separately. Outputs: {os.path.abspath(self.communication_tracker.csv_path)} and {os.path.abspath(self.communication_tracker.summary_path)}.')

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(self.client_ids, self.conf.n_participated, replace=False).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(f'Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}.')
        return selected_client_ids

    def _activate_selected_clients(self, selected_client_ids, comm_round, list_of_local_n_epochs):
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((3, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)
        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f'Master activated the selected clients.')
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f'Master send the models to workers.')
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(f'\tMaster send the global model={arch} to process_id={worker_rank}.')
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f'Master waits to receive the local models.')
        dist.barrier()
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(list(self.client_models[arch].state_dict().values()))
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(tensor=flatten_local_models[client_id].buffer, src=world_id)
            reqs.append(req)
        for req in reqs:
            req.wait()
        dist.barrier()
        self.conf.logger.log(f'Master received all local models.')
        return flatten_local_models

    def _compute_best_mean_client_performance(self):
        curr_perf = []
        for curr_personal_perf in self.curr_personal_perfs.values():
            curr_perf.append(curr_personal_perf)
        curr_perf = functools.reduce(lambda a, b: a + b, curr_perf) / len(curr_perf)
        save_average_test_stat(self.conf, curr_perf.dictionary)
        self.personal_avg_coordinator.update_perf(curr_perf)
        for name, best_tracker in self.personal_avg_coordinator.best_trackers.items():
            self.conf.logger.log('Personal_test method 2                 -- Personal best avg results of {}: \t {} best comm round {:.3f}, current comm_roud {:.3f}'.format(name, best_tracker.best_perf, best_tracker.get_best_perf_loc, self.conf.graph.comm_round))

    def _finishing(self, _comm_round=None):
        if self.communication_tracker is not None:
            self.communication_tracker.close()
        self.conf.logger.save_json()
        self.conf.logger.log(f'Master finished the federated learning.')
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f'echo {self.conf.checkpoint_root} >> {self.conf.job_id}')
