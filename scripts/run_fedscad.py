"""Resolve the manuscript settings and launch the original Gloo runtime."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--partition", choices=["dirichlet", "pathological"], default="dirichlet")
    parser.add_argument("--shards-per-client", type=int)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--participation", type=float, default=0.1)
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--anchor-init", choices=["etf", "random", "paired_class"], default="etf")
    parser.add_argument("--hc-weight", type=float)
    parser.add_argument("--kd-weight", type=float)
    parser.add_argument("--agce-weight", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if not 0 < args.participation <= 1 or int(args.clients * args.participation + 0.5) < 1:
        parser.error("Select at least one client per round.")
    if args.dataset == "tiny_imagenet" and args.partition == "pathological":
        parser.error("The released shard-based experiments cover CIFAR-10 and CIFAR-100.")
    with (ROOT / "configs" / f"{args.dataset}.json").open(encoding="utf-8") as stream:
        settings = json.load(stream)
    if args.hc_weight is not None:
        settings["hc_weight"] = args.hc_weight
    if args.kd_weight is not None:
        settings["kd_weight"] = args.kd_weight
    if args.shards_per_client is not None:
        settings["shard_per_user"] = args.shards_per_client
    arch = settings["arch"]
    settings.update({
        "algo": "fedscad", "data": args.dataset,
        "data_dir": str(args.data_dir.resolve()), "prepare_data": "combine",
        "complex_arch": f"master={arch},worker={arch}",
        "partition_data": "non_iid_dirichlet" if args.partition == "dirichlet" else "pathological",
        "non_iid_alpha": args.alpha, "manual_seed": args.seed,
        "n_clients": args.clients, "participation_ratio": args.participation,
        "n_comm_rounds": args.rounds, "batch_size": 64, "num_workers": 1,
        "train_data_ratio": 1.0, "val_data_ratio": 0.0, "test_data_ratio": 0.0,
        "world_conf": f"0,0,1,1,{args.clients}", "on_cuda": not args.cpu,
        "fl_aggregate": "scheme=federated_average",
        "optimizer": "sgd", "lr": 0.01, "momentum_factor": 0.9, "use_nesterov": True,
        "lr_warmup": True, "lr_warmup_epochs": 15, "lr_warmup_epochs_upper_bound": 300,
        "lr_scheduler": "MultiStepLR", "lr_decay": 0.1, "lr_milestones": "100,150",
        "use_hc": settings["hc_weight"] > 0,
        "use_logit_adjust": True, "ce_weight": 1.0,
        "anchor_temperature": 0.2, "agce_weight": args.agce_weight,
        "kd_temperature": 3, "self_distillation": 0,
        "kd_warmup_rounds": 0, "agce_warmup_rounds": 0,
        "anchor_init": args.anchor_init, "cnn_mlp_cifar_simple": True,
        "track_time": True, "display_tracked_time": True,
        "track_communication": True, "display_communication": True,
        "same_seed_process": True, "pn_normalize": True, "personal_test": False,
        "plot_data_distribution": False, "port": args.port,
        "python_path": sys.executable,
        "experiment": f"fedscad_{args.dataset}_seed{args.seed}_{args.partition}_{args.anchor_init}",
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
    })
    command = [sys.executable, str(ROOT / "run_gloo.py")]
    for key, val in settings.items():
        command.extend([f"--{key}", str(val)])
    command.extend(extra)
    return command, args.dry_run


if __name__ == "__main__":
    command, dry_run = build_command()
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)
