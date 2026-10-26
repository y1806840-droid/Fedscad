# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import torch.distributed as dist
import pcode.models as models
'define loaders for different models.'

def determine_arch(conf, client_id, use_complex_arch):
    _id = client_id if client_id is not None else 0
    if use_complex_arch:
        if _id == 0:
            arch = conf.arch_info['master']
        else:
            archs = conf.arch_info['worker']
            if len(conf.arch_info['worker']) == 1:
                arch = archs[0]
            else:
                assert 'num_clients_per_model' in conf.arch_info
                assert conf.arch_info['num_clients_per_model'] * len(archs) == conf.n_clients
                arch = archs[int((_id - 1) / conf.arch_info['num_clients_per_model'])]
    else:
        arch = conf.arch
    return arch
'some utilities functions.'

def get_model_stat(conf, model, arch):
    conf.logger.log("\t=> {} created model '{}. Total params: {}M".format('Master' if conf.graph.rank == 0 else f'Worker-{conf.graph.worker_id} (client-{conf.graph.client_id})', arch, sum((p.numel() for p in model.parameters() if p.requires_grad)) / 1000000.0))

def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    conf.logger.log('\tconsistent model for process (rank {})'.format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)

def define_model(conf, show_stat=True, to_consistent_model=True, use_complex_arch=True, client_id=None, arch=None):
    arch, model = define_cv_classification_model(conf, client_id, use_complex_arch, arch)
    if to_consistent_model:
        consistent_model(conf, model)
    if show_stat:
        get_model_stat(conf, model, arch)
    return (arch, model)

def define_cv_classification_model(conf, client_id, use_complex_arch, arch):
    arch = determine_arch(conf, client_id, use_complex_arch) if arch is None else arch
    if arch not in ('cnn_mlp', 'resnet18_mlp'):
        raise ValueError('Supported architectures: cnn_mlp, resnet18_mlp')
    return (arch, models.__dict__[arch](conf))
