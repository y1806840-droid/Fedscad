# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import copy
import json
import pcode.aggregation.fedavg as fedavg

class Aggregator(object):

    def __init__(self, conf, model, criterion, metrics, dataset, test_loaders, clientid2arch):
        self.conf = conf
        self.logger = conf.logger
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.metrics = metrics
        self.dataset = dataset
        self.test_loaders = test_loaders
        self.clientid2arch = clientid2arch
        self._define_aggregate_fn()

    def _define_aggregate_fn(self):
        if self.conf.fl_aggregate and self.conf.fl_aggregate['scheme'] != 'federated_average':
            raise ValueError('This release supports the FedSCAD/FedAvg aggregation path only.')
        self.aggregate_fn = None

    def _s1_federated_average(self):

        def f(**kwargs):
            valid_flatten_models = {k: v for k, v in kwargs['flatten_local_models'].items() if v.buffer.sum() != 0}
            if len(valid_flatten_models) == 0:
                return kwargs['master_model']
            return fedavg.fedavg(conf=self.conf, clientid2arch=self.clientid2arch, n_selected_clients=len(valid_flatten_models), flatten_local_models=valid_flatten_models, client_models=kwargs['client_models'], criterion=self.criterion, metrics=self.metrics)
        return f

    def aggregate(self, master_model, client_models, flatten_local_models, aggregate_fn_name=None, **kwargs):
        n_selected_clients = len(flatten_local_models)
        self.logger.log(f"Aggregator via {(aggregate_fn_name if aggregate_fn_name is not None else self.conf.fl_aggregate['scheme'])}: {(f'scheme={json.dumps(self.conf.fl_aggregate)}' if self.conf.fl_aggregate is not None else '')}")
        _aggregate_fn = self.aggregate_fn if aggregate_fn_name is None else getattr(self, aggregate_fn_name)()
        return _aggregate_fn(master_model=master_model, client_models=client_models, flatten_local_models=flatten_local_models, n_selected_clients=n_selected_clients, **kwargs)
