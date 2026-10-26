# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import os
import random
import socket
import sys
import traceback
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import pcode.utils.topology as topology
from parameters import get_args
from pcode.master import Master
from pcode.worker import Worker

def main(rank, size, conf, port):
    print(f'[Rank {rank}] >>> Attempting to init process group...', flush=True)
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('gloo', rank=rank, world_size=size, timeout=timedelta(minutes=300))
        print(f'Rank {rank}: distributed world initialized, world_size={dist.get_world_size()}', flush=True)
    except Exception:
        print(f'[Rank {rank}] failed to init the distributed world:\n{traceback.format_exc()}', file=sys.stderr, flush=True)
        raise
    completed_normally = False
    try:
        init_config(conf)
        if conf.graph.rank == 0:
            process = Master(conf)
        else:
            process = Worker(conf)
        process.run()
        completed_normally = True
    except Exception:
        trace = traceback.format_exc()
        print(f'[Rank {rank}] Unhandled exception:\n{trace}', file=sys.stderr, flush=True)
        logger = getattr(conf, 'logger', None)
        if logger is not None:
            try:
                logger.log(f'[Rank {rank}] Unhandled exception:\n{trace}')
                logger.save_json()
            except Exception as log_error:
                print(f'[Rank {rank}] failed to write exception to log: {log_error}', file=sys.stderr, flush=True)
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                if completed_normally:
                    dist.barrier()
            except Exception as e:
                print(f'[Rank {rank}] final barrier failed: {e}', flush=True)
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f'[Rank {rank}] destroy_process_group failed: {e}', flush=True)

def init_config(conf):
    conf.graph = topology.define_graph_topology(world=conf.world, world_conf=conf.world_conf, n_participated=conf.n_participated, on_cuda=conf.on_cuda)
    conf.graph.rank = dist.get_rank()
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    random.seed(conf.manual_seed)
    if conf.graph.on_cuda:
        torch.cuda.manual_seed(conf.manual_seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(conf.graph.rank % torch.cuda.device_count())
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(conf.manual_seed)
    conf.arch_info = param_parser.dict_parser(conf.complex_arch) if conf.complex_arch is not None else {'master': conf.arch, 'worker': conf.arch}
    conf.arch_info['worker'] = conf.arch_info['worker'].split(':')
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = param_parser.dict_parser(conf.fl_aggregate) if conf.fl_aggregate is not None else conf.fl_aggregate
    [setattr(conf, f'fl_aggregate_{k}', v) for k, v in conf.fl_aggregate.items()]
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))
    conf.logger = logging.Logger(conf.checkpoint_dir)
    if conf.graph.rank == 0:
        logging.display_args(conf)
    dist.barrier()

def _is_port_free(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()

def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 0))
        return str(sock.getsockname()[1])
    finally:
        sock.close()
if __name__ == '__main__':
    print('start', flush=True)
    conf = get_args()
    if not _is_port_free(conf.port):
        conf.port = _get_free_port()
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    size = conf.n_participated + 1
    processes = []
    mp.set_start_method('spawn', force=True)
    print('after spawn', flush=True)
    for rank in range(size):
        process = mp.Process(target=main, args=(rank, size, conf, conf.port))
        print(rank, flush=True)
        process.start()
        print(f'process{rank}.............................', flush=True)
        processes.append(process)
    print('before processes', flush=True)
    failed_processes = []
    for rank, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed_processes.append((rank, process.pid, process.exitcode))
    if failed_processes:
        details = ', '.join((f'rank={rank}, pid={pid}, exitcode={exitcode}' for rank, pid, exitcode in failed_processes))
        raise RuntimeError(f'One or more distributed child processes failed: {details}')
