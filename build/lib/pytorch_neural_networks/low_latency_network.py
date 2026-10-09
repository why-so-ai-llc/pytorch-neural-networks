"""Low-latency PyTorch neural network components."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise + pointwise convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu6(self.norm(self.pointwise(self.depthwise(x))), inplace=True)


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style block with an optional residual connection."""

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4, stride: int = 1) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        return x + result if self.use_residual else result


class LowLatencyNeuralNetwork(nn.Module):
    """Compact classifier designed for low-latency inference."""

    def __init__(self, input_channels: int = 3, num_classes: int = 10, width_multiplier: float = 0.5) -> None:
        super().__init__()

        def channels(value: int) -> int:
            return max(8, int(value * width_multiplier))

        c1, c2, c3 = channels(32), channels(64), channels(128)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
            InvertedResidualBlock(c1, c2, stride=2),
            InvertedResidualBlock(c2, c2),
            InvertedResidualBlock(c2, c3, stride=2),
            InvertedResidualBlock(c3, c3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_inference_time(model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100, device: str = "cpu") -> Dict[str, float]:
    """Measure warm-start inference latency in milliseconds."""
    model = model.to(device).eval()
    sample = torch.randn(input_shape, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    values = np.asarray(timings)
    return {
        "mean_latency_ms": float(values.mean()),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
    }
