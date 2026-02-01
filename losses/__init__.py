"""Loss functions for Promoter-Classification."""

from .cross_entropy import CrossEntropyLoss
from .focal_loss import FocalLoss

__all__ = ["CrossEntropyLoss", "FocalLoss"]
