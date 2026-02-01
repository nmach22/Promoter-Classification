import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for classification.
    Args:
        alpha: balancing factor, either float or list of per-class weights
        gamma: focusing parameter
        reduction: 'mean'|'sum'|'none'
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), targets: (N,)
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = torch.log(pt + 1e-12)
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha[targets].to(logits.device)
        else:
            at = self.alpha
        loss = -at * ((1 - pt) ** self.gamma) * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
