"""Small CPU checks for the released models and the active local objective."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F

from pcode.models.cnn_mlp import cnn_mlp
from pcode.models.resnet18_mlp import resnet18_mlp
from pcode.shared_anchors import SharedAnchorScores
from pcode.workers.worker_fedscad import WorkerFedSCAD


class Components(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_model_outputs(self):
        for data, factory, dim, classes, size in (
            ("cifar10", cnn_mlp, 128, 10, 32),
            ("cifar100", resnet18_mlp, 128, 100, 32),
            ("tiny_imagenet", resnet18_mlp, 256, 200, 64),
        ):
            model = factory(SimpleNamespace(data=data, feat_dim=dim)).eval()
            with torch.no_grad():
                projection, logits, backbone = model(torch.randn(2, 3, size, size))
            self.assertEqual(projection.shape, (2, dim))
            self.assertEqual(logits.shape, (2, classes))
            self.assertEqual(backbone.shape, (2, 512))

    def test_anchor_geometry(self):
        etf = SharedAnchorScores(128).estimator_old.Ave
        gram = etf @ etf.T
        expected = torch.full((10, 10), -1 / 9)
        expected.fill_diagonal_(1)
        torch.testing.assert_close(gram, expected, atol=1e-5, rtol=1e-5)
        a = SharedAnchorScores(128, anchor_init="random", anchor_seed=100)
        b = SharedAnchorScores(128, anchor_init="random", anchor_seed=100)
        torch.testing.assert_close(a.estimator_old.Ave, b.estimator_old.Ave)
        paired = SharedAnchorScores(128, anchor_init="paired_class")
        scores = paired(torch.randn(3, 128))
        torch.testing.assert_close(scores[:, 0], scores[:, 1])

    def test_detached_kd_target(self):
        features = torch.randn(4, 128, requires_grad=True)
        logits = torch.randn(4, 10, requires_grad=True)
        teacher_scores = SharedAnchorScores(128)(features)
        loss = F.kl_div(F.log_softmax(logits / 3, dim=-1),
                        F.log_softmax(teacher_scores.detach() / 3, dim=-1),
                        log_target=True, reduction="batchmean") * 9
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertGreater(logits.grad.norm().item(), 0)

    def test_agce_reaches_backbone(self):
        model = cnn_mlp(SimpleNamespace(data="cifar10", feat_dim=128)).train()
        features, _, _ = model(torch.randn(4, 3, 32, 32))
        F.cross_entropy(SharedAnchorScores(128)(features), torch.arange(4)).backward()
        self.assertGreater(model.conv1.weight.grad.norm().item(), 0)

    def test_local_objective(self):
        worker = WorkerFedSCAD.__new__(WorkerFedSCAD)
        worker.conf = SimpleNamespace(
            graph=SimpleNamespace(comm_round=2), ce_weight=1.0, agce_weight=0.3,
            kd_weight=0.4, kd_temperature=3,
            use_hc=True, hc_weight=0.15, kd_warmup_rounds=0,
            agce_warmup_rounds=0,
        )
        worker.model = cnn_mlp(SimpleNamespace(data="cifar10", feat_dim=128)).eval()
        worker.anchor_scores = SharedAnchorScores(128)
        worker.criterion = nn.CrossEntropyLoss()
        worker.agce_criterion = nn.CrossEntropyLoss()
        worker._enable_aux = True
        worker.tracker = None
        worker.metrics = SimpleNamespace(evaluate=lambda *args: [])
        params = list(worker.model.parameters())
        worker.dual_variable_list = [torch.randn_like(p) * 0.001 for p in params]
        worker.regular_params_list = [p.detach().clone() for p in params]
        x, y = torch.randn(4, 3, 32, 32), torch.arange(4)
        features, logits, _ = worker.model(x)
        scores = worker.anchor_scores(features)
        kd = F.kl_div(F.log_softmax(logits / 3, dim=-1),
                      F.log_softmax(scores.detach() / 3, dim=-1),
                      log_target=True, reduction="batchmean") * 9
        hc = sum(-0.15 * (p * d).sum() for p, d in zip(params, worker.dual_variable_list))
        expected = F.cross_entropy(logits, y) + 0.3 * F.cross_entropy(scores, y) + 0.4 * kd + hc
        actual, _ = worker._inference({"input": x, "target": y}, update_tracker=False)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
