"""Run two CPU rounds on synthetic data without downloading a dataset."""

import importlib.util
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

import pcode.create_dataset as create_dataset
from pcode.datasets.partition_data import Partition
from parameters import get_args
import run_gloo


class SyntheticDataset(Dataset):
    def __init__(self, n=32):
        generator = torch.Generator().manual_seed(7)
        self.data = torch.randn(n, 3, 32, 32, generator=generator)
        self.targets = (torch.arange(n) % 10).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def synthetic_data(conf, data, display_log=True):
    conf.partitioned_by_user = False
    base = SyntheticDataset()
    local = {
        cid: {"train": Partition(base, list(range((cid - 1) * 16, cid * 16))),
              "test": Partition(base, [])}
        for cid in (1, 2)
    }
    return {"client_local": local, "server_test": base}


def rank_main(rank, conf):
    torch.set_num_threads(1)
    create_dataset.define_dataset = synthetic_data
    run_gloo.main(rank, 3, conf, conf.port)


def main():
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/run_fedscad.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with tempfile.TemporaryDirectory(prefix="fedscad-smoke-") as output:
        command, _ = launcher.build_command([
            "--cpu", "--clients", "2", "--participation", "1", "--rounds", "2",
            "--local_n_epochs", "1", "--batch_size", "4", "--num_workers", "0",
            "--pin_memory", "False", "--checkpoint", output,
            "--feature_similarity_save_tsne", "False",
        ])
        sys.argv = command[1:]
        conf = get_args()
        conf.port = run_gloo._get_free_port()
        conf.n_participated = 2
        ctx = mp.get_context("spawn")
        processes = [ctx.Process(target=rank_main, args=(rank, conf)) for rank in range(3)]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=120)
            if any(p.is_alive() or p.exitcode != 0 for p in processes):
                raise RuntimeError("A smoke-test process failed or timed out.")
            if not list(Path(output).rglob("communication_summary.json")):
                raise RuntimeError("The communication tracker did not write its summary.")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
        print("PASS: two CPU rounds, CE warmup, auxiliary training, HC and communication logging.")


if __name__ == "__main__":
    main()
