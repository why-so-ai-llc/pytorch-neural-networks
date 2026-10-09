"""Backward-compatible exports for low_latency_network."""

from pytorch_neural_networks.low_latency_network import (
    DepthwiseSeparableConv,
    InvertedResidualBlock,
    LowLatencyNeuralNetwork,
    count_parameters,
    measure_inference_time,
)


if __name__ == "__main__":
    network = LowLatencyNeuralNetwork()
    print(count_parameters(network))
    print(measure_inference_time(network, (1, 3, 224, 224), num_runs=20))
