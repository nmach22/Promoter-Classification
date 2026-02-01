import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """Thin wrapper around `nn.CrossEntropyLoss` to keep interface consistent."""
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy.
        logits: (N, C)
        targets: (N,) long
        """
        return self.loss(logits, targets)
