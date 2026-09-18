"""Knowledge-distillation loss for compressing a teacher into a student."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, target: Tensor) -> Tensor:
        hard = self.hard_loss(student_logits, target)
        temperature = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits.detach() / temperature, dim=1),
            reduction="batchmean",
        ) * temperature ** 2
        return (1.0 - self.alpha) * hard + self.alpha * soft
