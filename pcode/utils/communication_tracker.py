# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

"""Application-level communication accounting for the server process.

The tracker counts bytes in tensors passed to ``torch.distributed``. It does
not estimate transport headers, retransmissions, or backend-dependent network
overhead. Point-to-point traffic is the training payload; broadcast control
messages are recorded separately.
"""
import atexit
import csv
import json
import os
import torch
import torch.distributed as dist

def tensor_nbytes(tensor):
    """Return the actual tensor payload in bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor, got {type(tensor)!r}.')
    return int(tensor.numel()) * int(tensor.element_size())

class DistributedCommunicationTracker:
    """Count server-side distributed tensor traffic by communication round."""
    _PATCHED_ATTR = '_pcode_communication_tracker'

    def __init__(self, conf):
        self.conf = conf
        self.output_dir = conf.checkpoint_dir
        self.csv_path = os.path.join(self.output_dir, 'communication_rounds.csv')
        self.summary_path = os.path.join(self.output_dir, 'communication_summary.json')
        self.display = bool(getattr(conf, 'display_communication', False))
        self.current_round = None
        self.current = self._empty_round()
        self.rows = []
        self._originals = {}
        self._installed = False
        self._closed = False
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=self._fieldnames()).writeheader()
        self._write_summary(include_current=False)
        atexit.register(self.close)

    @staticmethod
    def _empty_round():
        return {'downlink_p2p_bytes': 0, 'uplink_p2p_bytes': 0, 'control_broadcast_bytes': 0, 'downlink_messages': 0, 'uplink_messages': 0, 'broadcast_calls': 0}

    @staticmethod
    def _fieldnames():
        return ['round', 'downlink_p2p_bytes', 'uplink_p2p_bytes', 'control_broadcast_bytes', 'payload_total_bytes', 'application_total_bytes', 'cumulative_payload_bytes', 'cumulative_application_bytes', 'downlink_messages', 'uplink_messages', 'broadcast_calls']

    def _round(self):
        graph = getattr(self.conf, 'graph', None)
        return int(getattr(graph, 'comm_round', 0) or 0)

    def _ensure_round(self):
        comm_round = self._round()
        if self.current_round is None:
            self.current_round = comm_round
        elif comm_round != self.current_round:
            self._flush_current_round()
            self.current_round = comm_round
            self.current = self._empty_round()

    def record_p2p(self, direction, tensor):
        self._ensure_round()
        nbytes = tensor_nbytes(tensor)
        if direction == 'downlink':
            self.current['downlink_p2p_bytes'] += nbytes
            self.current['downlink_messages'] += 1
        elif direction == 'uplink':
            self.current['uplink_p2p_bytes'] += nbytes
            self.current['uplink_messages'] += 1
            n_participated = max(1, int(getattr(self.conf, 'n_participated', 1)))
            if self.current['uplink_messages'] % n_participated == 0:
                self._write_summary(include_current=True)
                self._write_csv(include_current=True)
        else:
            raise ValueError(f'Unknown communication direction: {direction!r}.')

    def record_broadcast(self, tensor, peers):
        self._ensure_round()
        self.current['control_broadcast_bytes'] += tensor_nbytes(tensor) * int(peers)
        self.current['broadcast_calls'] += 1

    def _flush_current_round(self):
        if self.current_round is None:
            return
        if self.current_round <= 0 and (not any(self.current.values())):
            return
        downlink = self.current['downlink_p2p_bytes']
        uplink = self.current['uplink_p2p_bytes']
        control = self.current['control_broadcast_bytes']
        payload = downlink + uplink
        application = payload + control
        previous_payload = self.rows[-1]['cumulative_payload_bytes'] if self.rows else 0
        previous_application = self.rows[-1]['cumulative_application_bytes'] if self.rows else 0
        row = {'round': self.current_round, **self.current, 'payload_total_bytes': payload, 'application_total_bytes': application, 'cumulative_payload_bytes': previous_payload + payload, 'cumulative_application_bytes': previous_application + application}
        self.rows.append(row)
        self._write_csv(include_current=False)
        self._write_summary(include_current=False)
        if self.display:
            mib = 2 ** 20
            self.conf.logger.log('Communication payload at round {}: downlink={:.3f} MiB, uplink={:.3f} MiB, total={:.3f} MiB.'.format(self.current_round, downlink / mib, uplink / mib, payload / mib))

    def _write_csv(self, include_current=False):
        output_rows = list(self.rows)
        if include_current and self.current_round is not None and any(self.current.values()):
            downlink = self.current['downlink_p2p_bytes']
            uplink = self.current['uplink_p2p_bytes']
            control = self.current['control_broadcast_bytes']
            previous_payload = output_rows[-1]['cumulative_payload_bytes'] if output_rows else 0
            previous_application = output_rows[-1]['cumulative_application_bytes'] if output_rows else 0
            payload = downlink + uplink
            application = payload + control
            output_rows.append({'round': self.current_round, **self.current, 'payload_total_bytes': payload, 'application_total_bytes': application, 'cumulative_payload_bytes': previous_payload + payload, 'cumulative_application_bytes': previous_application + application})
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames())
            writer.writeheader()
            for row in output_rows:
                writer.writerow({name: row[name] for name in self._fieldnames()})

    def _write_summary(self, include_current=False):
        summary_rows = list(self.rows)
        if include_current and self.current_round is not None and any(self.current.values()):
            downlink = self.current['downlink_p2p_bytes']
            uplink = self.current['uplink_p2p_bytes']
            control = self.current['control_broadcast_bytes']
            summary_rows.append({'round': self.current_round, **self.current, 'payload_total_bytes': downlink + uplink, 'application_total_bytes': downlink + uplink + control})
        payloads = [row['payload_total_bytes'] for row in summary_rows]
        downlinks = [row['downlink_p2p_bytes'] for row in summary_rows]
        uplinks = [row['uplink_p2p_bytes'] for row in summary_rows]
        applications = [row['application_total_bytes'] for row in summary_rows]
        n_rounds = len(summary_rows)
        total_payload = sum(payloads)
        total_application = sum(applications)
        summary = {'definition': 'Application-level tensor bytes. payload excludes activation/control broadcasts; application_total includes them. Transport overhead is excluded.', 'algorithm': str(getattr(self.conf, 'algo', 'unknown')), 'data': str(getattr(self.conf, 'data', 'unknown')), 'architecture': str(getattr(self.conf, 'arch', 'unknown')), 'n_clients': int(getattr(self.conf, 'n_clients', 0)), 'n_participated': int(getattr(self.conf, 'n_participated', 0)), 'rounds_recorded': n_rounds, 'contains_live_round': bool(include_current and len(summary_rows) > len(self.rows)), 'total_payload_bytes': total_payload, 'total_application_bytes': total_application, 'mean_downlink_payload_bytes_per_round': sum(downlinks) / n_rounds if n_rounds else 0, 'mean_uplink_payload_bytes_per_round': sum(uplinks) / n_rounds if n_rounds else 0, 'mean_payload_bytes_per_round': total_payload / n_rounds if n_rounds else 0, 'mean_application_bytes_per_round': total_application / n_rounds if n_rounds else 0, 'min_payload_bytes_per_round': min(payloads) if payloads else 0, 'max_payload_bytes_per_round': max(payloads) if payloads else 0, 'csv': os.path.basename(self.csv_path)}
        temporary_path = self.summary_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        os.replace(temporary_path, self.summary_path)

    @staticmethod
    def _extract_tensor(args, kwargs):
        if 'tensor' in kwargs:
            return kwargs['tensor']
        if args:
            return args[0]
        raise TypeError('A distributed tensor operation was called without a tensor.')

    def install(self):
        if self._installed:
            return
        existing = getattr(dist, self._PATCHED_ATTR, None)
        if existing is not None and existing is not self:
            raise RuntimeError('torch.distributed is already tracked by another instance.')
        for name in ('send', 'isend', 'recv', 'irecv', 'broadcast'):
            self._originals[name] = getattr(dist, name)
        tracker = self

        def tracked_send(*args, **kwargs):
            result = tracker._originals['send'](*args, **kwargs)
            tracker.record_p2p('downlink', tracker._extract_tensor(args, kwargs))
            return result

        def tracked_isend(*args, **kwargs):
            result = tracker._originals['isend'](*args, **kwargs)
            tracker.record_p2p('downlink', tracker._extract_tensor(args, kwargs))
            return result

        def tracked_recv(*args, **kwargs):
            result = tracker._originals['recv'](*args, **kwargs)
            tracker.record_p2p('uplink', tracker._extract_tensor(args, kwargs))
            return result

        def tracked_irecv(*args, **kwargs):
            result = tracker._originals['irecv'](*args, **kwargs)
            tracker.record_p2p('uplink', tracker._extract_tensor(args, kwargs))
            return result

        def tracked_broadcast(*args, **kwargs):
            result = tracker._originals['broadcast'](*args, **kwargs)
            src = kwargs.get('src', args[1] if len(args) > 1 else None)
            if src == dist.get_rank():
                tracker.record_broadcast(tracker._extract_tensor(args, kwargs), max(0, dist.get_world_size() - 1))
            return result
        dist.send = tracked_send
        dist.isend = tracked_isend
        dist.recv = tracked_recv
        dist.irecv = tracked_irecv
        dist.broadcast = tracked_broadcast
        setattr(dist, self._PATCHED_ATTR, self)
        self._installed = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._flush_current_round()
        if self._installed:
            for name, original in self._originals.items():
                setattr(dist, name, original)
            if getattr(dist, self._PATCHED_ATTR, None) is self:
                delattr(dist, self._PATCHED_ATTR)
            self._installed = False
