# FedSCAD

Implementation of **Learning from Shared Class Anchors: Knowledge Distillation
for Federated Learning**.

FedSCAD learns from fixed shared class anchors through an auxiliary projection
branch. Anchor-guided cross-entropy shapes this branch, and knowledge
distillation transfers its class scores to the task classifier. Accumulated
client-update history supplies the linear history-correction term.

## Setup

Use Python 3.10 or later. Install matching **PyTorch and torchvision** builds
for your CUDA environment, then install the remaining dependencies:

~~~bash
pip install -r requirements.txt
~~~

Training uses PyTorch multiprocessing and Gloo: one server process and one
worker process per participating client, on the same machine. The full
experiments used an RTX 5090. CPU execution is available with --cpu.

## Datasets

- **CIFAR-10 / CIFAR-100:** torchvision downloads the data when needed.
  --data-dir ./data creates data/cifar10/ and data/cifar100/.
- **Tiny-ImageNet:** place the dataset in data/tiny_imagenet/, containing
  wnids.txt, train/ and val/val_annotations.txt. The labeled validation
  split is used for evaluation.

Datasets, trained checkpoints and experiment logs are not included.

## Training

Run from the repository root:

~~~bash
python scripts/run_fedscad.py --dataset cifar10 --data-dir ./data --alpha 0.1 --seed 100
python scripts/run_fedscad.py --dataset cifar100 --data-dir ./data --alpha 0.1 --seed 100
python scripts/run_fedscad.py --dataset tiny_imagenet --data-dir ./data --alpha 0.1 --seed 100
~~~

Select a device with CUDA_VISIBLE_DEVICES, for example:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_fedscad.py --dataset cifar10 --data-dir ./data
~~~

Use --dry-run to print the resolved command without loading PyTorch.
Additional run_gloo.py arguments may follow the launcher's options.

The common setup is 100 clients, 10% participation, 500 rounds, batch size
64, and SGD with learning rate 0.01 and momentum 0.9. Dataset settings are
stored in configs/:

| Setting | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|---|---:|---:|---:|
| Backbone + auxiliary head | Simple-CNN + MLP | ResNet-18 + MLP | ResNet-18 + MLP |
| Local epochs | 5 | 3 | 2 |
| Projection dimension | 128 | 128 | 256 |
| Weight decay | 0.00001 | 0.0005 | 0.0005 |
| HC coefficient | 0.15 | 0.25 | 0.25 |
| KD coefficient | 0.4 | 1.0 | 1.0 |

The AGCE weight is 0.3, anchor temperature is 0.2, and KD temperature is 3.
Logit adjustment is enabled. The first round uses task cross-entropy;
auxiliary supervision and KD start from round two. The partition seed is
fixed to 7; the training seed controls model initialization, client sampling,
minibatch order and augmentation.

For shard-based label skew:

~~~bash
python scripts/run_fedscad.py --dataset cifar10 --partition pathological --shards-per-client 2
python scripts/run_fedscad.py --dataset cifar100 --partition pathological --shards-per-client 10
~~~

Shards have a single class label; a client's assigned labels may repeat.

## Controls and diagnostics

~~~bash
# Fixed shared random unit anchors instead of the ETF.
python scripts/run_fedscad.py --dataset cifar10 --anchor-init random

# Adjacent class pairs share an anchor.
python scripts/run_fedscad.py --dataset cifar10 --anchor-init paired_class

# Disable task-classifier KD or history correction.
python scripts/run_fedscad.py --dataset cifar10 --kd-weight 0
python scripts/run_fedscad.py --dataset cifar10 --hc-weight 0

# Save round-500 client representations for geometry analysis.
python scripts/run_fedscad.py --dataset cifar10 --feature_similarity_probe_rounds 500
~~~

Time and application-level communication tracking are enabled by the launcher.
Results are written below checkpoint/, including text/JSON logs and
communication summaries. Optional t-SNE output requires
pip install -r requirements-analysis.txt.

## Code map

- run_gloo.py: distributed process entry point.
- pcode/workers/worker_fedscad.py: local CE, AGCE, detached-teacher KD and HC.
- pcode/masters/master_fedscad.py: model exchange and accumulated history correction.
- pcode/shared_anchors.py: fixed ETF, random and paired anchors, and normalized scores.
- pcode/models/: the two backbone/projection-head architectures.
- pcode/datasets/: loading and the original partition generator.
- pcode/utils/: checkpointing, metrics, timing and communication tracking.

Source files, classes, configuration keys and logs use FedSCAD naming.
Older command-line options remain accepted as aliases for compatibility.
The released method is FedSCAD. Other baselines and the later security/client
selection experiments are not included.

## Checks

~~~bash
# No third-party packages: syntax, local imports and launch arguments.
python tests/check_release.py

# Requires PyTorch/torchvision: model outputs, anchors, KD and HC gradients.
python -m unittest discover -s tests -p "test_*.py"

# Two-round CPU check with synthetic data; no dataset download.
python tests/smoke_distributed.py
~~~

The release was statically checked locally. The dependency-based tests are
provided for execution in a configured training environment; they were not
run in the local packaging environment.

## Citation

The manuscript citation will be added when its author list and publication
information are available.

## Acknowledgements and license

The skeleton codebase was adapted from FedDF:

T. Lin, L. Kong, S. U. Stich, and M. Jaggi,
[Ensemble Distillation for Robust Model Fusion in Federated Learning](https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf),
NeurIPS, 2020.

This repository uses Apache License 2.0. See LICENSE, NOTICE, and
LICENSES/FedDF.txt. Modified framework files carry a release notice.
