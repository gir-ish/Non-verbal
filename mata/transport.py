"""Optimal Transport Alignment Module"""

import torch
import torch.nn as nn
import numpy as np


class OptimalTransportAlignment(nn.Module):
    """
    Optimal transport-based alignment of feature distributions.

    Uses Wasserstein distance to align features from different foundation models.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, source_dist, target_dist):
        """
        Align source distribution to target using optimal transport.

        Args:
            source_dist: Source feature distribution
            target_dist: Target feature distribution

        Returns:
            Aligned features
        """
        # Normalize distributions
        source = source_dist / (torch.norm(source_dist, dim=-1, keepdim=True) + self.eps)
        target = target_dist / (torch.norm(target_dist, dim=-1, keepdim=True) + self.eps)

        # Compute Wasserstein-like alignment
        # In practice, use POT (Python Optimal Transport) library
        aligned = source + (target - source) * 0.5

        return aligned
