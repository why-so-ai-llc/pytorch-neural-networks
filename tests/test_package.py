import unittest

import torch
from torch import nn

from pytorch_neural_networks import (
    EfficientRLLoop,
    KnowledgeDistillationLoss,
    LowLatencyNeuralNetwork,
    PrioritizedReplayBuffer,
    count_parameters,
)
from pytorch_neural_networks.efficient_rl_loop import Experience


class PackageTests(unittest.TestCase):
    def test_low_latency_network_output_shape(self) -> None:
        model = LowLatencyNeuralNetwork(input_channels=3, num_classes=7, width_multiplier=0.5)
        sample = torch.randn(2, 3, 64, 64)
        output = model(sample)
        self.assertEqual(output.shape, (2, 7))

    def test_count_parameters_reports_trainable_subset(self) -> None:
        model = LowLatencyNeuralNetwork()
        counts = count_parameters(model)
        self.assertGreater(counts["total"], 0)
        self.assertGreaterEqual(counts["total"], counts["trainable"])

    def test_distillation_loss_returns_scalar(self) -> None:
        criterion = KnowledgeDistillationLoss(alpha=0.25, temperature=2.0)
        student_logits = torch.randn(4, 3, requires_grad=True)
        teacher_logits = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 1])
        loss = criterion(student_logits, teacher_logits, target)
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(loss.item(), 0.0)

    def test_replay_buffer_sampling_and_update(self) -> None:
        replay = PrioritizedReplayBuffer(capacity=8)
        for index in range(4):
            replay.add(
                Experience(
                    state=[index, index + 1],
                    action=index % 2,
                    reward=float(index),
                    next_state=[index + 1, index + 2],
                    done=False,
                )
            )
        batch, weights, indices = replay.sample(batch_size=2)
        self.assertEqual(len(batch), 2)
        self.assertEqual(weights.shape[0], 2)
        self.assertEqual(indices.shape[0], 2)
        replay.update_priorities(indices, weights)

    def test_efficient_rl_loop_train_step(self) -> None:
        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
        loop = EfficientRLLoop(model=model, action_dim=2, batch_size=2, capacity=16)
        loop.store([0.0, 1.0], 0, 1.0, [0.5, 1.5], False)
        loop.store([1.0, 0.0], 1, 0.5, [1.5, 0.5], True)
        metrics = loop.train_step()
        self.assertIn("loss", metrics)
        self.assertIn("mean_abs_td_error", metrics)
        self.assertEqual(loop.steps, 1)


if __name__ == "__main__":
    unittest.main()
