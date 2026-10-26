# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

"""Append-only, per-worker CUDA allocator diagnostics (no tensors retained)."""
import json
import os
from pathlib import Path
import time
import warnings

class GpuMemoryTracker:

    def __init__(self, conf, cuda):
        self.conf = conf
        self.cuda = cuda
        self.enabled = bool(getattr(conf, 'track_gpu_memory', False)) and bool(conf.graph.on_cuda)
        self.interval = max(1, int(getattr(conf, 'gpu_memory_interval', 10)))
        self.path = None
        self.peak_round = None

    def record(self, stage, reset_peak=False):
        if not self.enabled:
            return
        round_id = int(self.conf.graph.comm_round)
        if round_id < 1 or (round_id > 3 and round_id % self.interval != 0):
            return
        if not self.cuda.is_initialized():
            return
        try:
            device = self.cuda.current_device()
            if reset_peak:
                self.cuda.reset_peak_memory_stats(device)
                self.peak_round = round_id
            stats = self.cuda.memory_stats(device)
            mib = 1024 ** 2
            allocated = self.cuda.memory_allocated(device)
            reserved = self.cuda.memory_reserved(device)
            record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(), 'rank': int(self.conf.graph.rank), 'client_id': int(self.conf.graph.client_id), 'comm_round': round_id, 'stage': stage, 'algo': str(getattr(self.conf, 'algo', '')), 'cuda_device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''), 'allocated_mib': allocated / mib, 'reserved_mib': reserved / mib, 'reserved_minus_allocated_mib': (reserved - allocated) / mib, 'peak_allocated_mib': self.cuda.max_memory_allocated(device) / mib, 'peak_reserved_mib': self.cuda.max_memory_reserved(device) / mib, 'peak_reset_round': self.peak_round, 'inactive_split_mib': stats.get('inactive_split_bytes.all.current', 0) / mib, 'allocation_retries_total': stats.get('num_alloc_retries', 0), 'ooms_total': stats.get('num_ooms', 0)}
            if self.path is None:
                folder = Path(self.conf.checkpoint_root) / 'gpu_memory'
                folder.mkdir(parents=True, exist_ok=True)
                self.path = folder / 'worker-{}_pid-{}.jsonl'.format(self.conf.graph.rank, os.getpid())
                print('[GPU memory] saving to {}'.format(self.path), flush=True)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        except (OSError, RuntimeError, ValueError) as exc:
            self.enabled = False
            warnings.warn('GPU memory logging disabled after error: {}'.format(exc))
