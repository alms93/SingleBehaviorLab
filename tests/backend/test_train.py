"""Tests for loss functions, samplers, and class weights (singlebehaviorlab.backend.train)."""

import torch
import numpy as np
import pytest

from singlebehaviorlab.backend.train import (
    FocalLoss,
    BinaryFocalLoss,
    AsymmetricLoss,
    BalancedBatchSampler,
    ConfusionAwareSampler,
    compute_class_weights,
)


class TestComputeClassWeights:
    def test_balanced_classes_equal_weights(self):
        labels = [0, 0, 1, 1, 2, 2]
        w = compute_class_weights(labels, num_classes=3)
        assert w.shape == (3,)
        assert torch.allclose(w, w[0].expand(3), atol=1e-5)

    def test_imbalanced_classes_rare_gets_higher_weight(self):
        labels = [0] * 100 + [1] * 10
        w = compute_class_weights(labels, num_classes=2)
        assert w[1] > w[0]

    def test_empty_labels_returns_ones(self):
        w = compute_class_weights([], num_classes=3)
        assert torch.allclose(w, torch.ones(3))


class TestFocalLoss:
    def test_easy_vs_hard_predictions(self):
        fl = FocalLoss(gamma=2.0)
        easy_logits = torch.tensor([[5.0, -5.0]])
        hard_logits = torch.tensor([[0.1, -0.1]])
        targets = torch.tensor([0])
        easy_loss = fl(easy_logits, targets)
        hard_loss = fl(hard_logits, targets)
        assert hard_loss.item() > easy_loss.item()


class TestBinaryFocalLoss:
    def test_output_shape(self):
        bfl = BinaryFocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.zeros(4, 3)
        targets[0, 0] = 1.0
        loss = bfl(logits, targets)
        assert loss.shape == (4, 3)

    def test_mean_is_scalar(self):
        bfl = BinaryFocalLoss(gamma=2.0, reduction="mean")
        loss = bfl(torch.randn(4, 3), torch.zeros(4, 3))
        assert loss.dim() == 0


class TestAsymmetricLoss:
    def test_neg_suppression(self):
        """Easy negatives should produce lower loss than hard negatives."""
        asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05, reduction="none")
        targets = torch.zeros(1, 2)
        easy_neg = torch.tensor([[-5.0, -5.0]])  # sigmoid ≈ 0 → easy negative
        hard_neg = torch.tensor([[0.0, 0.0]])     # sigmoid = 0.5 → hard negative
        loss_easy = asl(easy_neg, targets)
        loss_hard = asl(hard_neg, targets)
        assert loss_easy.sum().item() < loss_hard.sum().item()


class TestBalancedBatchSampler:
    def test_all_classes_appear_in_batch(self):
        labels = [0]*20 + [1]*20 + [2]*20
        sampler = BalancedBatchSampler(labels, batch_size=12, min_samples_per_class=2, seed=42)
        first_batch = next(iter(sampler))
        batch_labels = {labels[i] for i in first_batch}
        assert len(batch_labels) >= 2

    def test_batch_size_respected(self):
        labels = [0]*10 + [1]*10
        sampler = BalancedBatchSampler(labels, batch_size=8, min_samples_per_class=2, seed=42)
        for batch in sampler:
            assert len(batch) == 8

    def test_no_eligible_classes_yields_nothing(self):
        labels = [0]
        sampler = BalancedBatchSampler(labels, batch_size=4, min_samples_per_class=2, seed=42)
        batches = list(sampler)
        assert len(batches) == 0


class TestConfusionAwareSampler:
    def test_update_weights_changes_sampling(self):
        labels = [0]*20 + [1]*20
        sampler = ConfusionAwareSampler(labels, batch_size=8, min_samples_per_class=2, seed=42)
        scores = np.zeros(40, dtype=np.float32)
        scores[:5] = 1.0  # first 5 samples of class 0 are "confused"
        sampler.update_weights(scores)
        assert sampler._confusion_scores[:5].sum() > 0
