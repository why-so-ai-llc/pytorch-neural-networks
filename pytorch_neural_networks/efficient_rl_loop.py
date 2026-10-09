"""Sample-efficient Double-DQN training utilities."""

from __future__ import annotations

import copy
import random
from collections import deque, namedtuple
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

Experience = namedtuple("Experience", "state action reward next_state done")


class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha, self.beta = alpha, beta
        self.max_priority = 1.0

    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        priority = self.max_priority if priority is None else abs(float(priority)) + 1e-6
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences to sample this batch size")
        priorities = np.asarray(self.priorities, dtype=np.float64)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.buffer[index] for index in indices], weights.astype(np.float32), indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for index, error in zip(indices, td_errors):
            raw = abs(float(error)) + 1e-6
            self.priorities[index] = raw ** self.alpha
            self.max_priority = max(self.max_priority, raw)

    def __len__(self) -> int:
        return len(self.buffer)


class EfficientRLLoop:
    """Double-DQN loop with warm-up, PER, target updates, and gradient clipping."""

    def __init__(self, model: nn.Module, action_dim: int, batch_size: int = 64, gamma: float = 0.99, tau: float = 0.005, capacity: int = 100_000, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.q_network = model.to(self.device)
        self.target_network = copy.deepcopy(model).to(self.device).eval()
        self.action_dim = action_dim
        self.batch_size, self.gamma, self.tau = batch_size, gamma, tau
        self.replay = PrioritizedReplayBuffer(capacity)
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=1e-4, weight_decay=1e-5)
        self.steps = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            return int(self.q_network(state_tensor).argmax(dim=1).item())

    def store(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.replay.add(Experience(state, action, reward, next_state, done))

    def train_step(self) -> Dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {}
        experiences, weights, indices = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.asarray([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([e.action for e in experiences], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.asarray([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)
        importance = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current = self.q_network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_values = self.target_network(next_states).gather(1, next_actions[:, None]).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * next_values

        td_errors = target - current
        loss = (importance * F.smooth_l1_loss(current, target, reduction="none")).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, online_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.lerp_(online_param, self.tau)
        self.replay.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.steps += 1
        return {"loss": float(loss.item()), "mean_abs_td_error": float(td_errors.abs().mean().item())}
