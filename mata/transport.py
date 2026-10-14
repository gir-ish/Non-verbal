"""Optimal Transport Alignment Module"""

import torch
import torch.nn as nn
import numpy as np
import ot as pot


class OptimalTransportAlignment(nn.Module):
    """
    Optimal transport-based alignment using Sinkhorn algorithm.

    Aligns source features to target features by solving the optimal transport problem
    with entropic regularization (Sinkhorn algorithm).
    """

    def __init__(self, eps: float = 1e-8, reg: float = 0.1):
        super().__init__()
        self.eps = eps
        self.reg = reg  # Entropic regularization parameter

    def forward(self, source_feat, target_feat):
        """
        Align source to target using Sinkhorn OT algorithm.

        Args:
            source_feat: Source features [batch, dim]
            target_feat: Target features [batch, dim]

        Returns:
            Aligned features via optimal transport plan
        """
        # Convert to numpy for OT computation
        if isinstance(source_feat, torch.Tensor):
            source_np = source_feat.detach().cpu().numpy()
            target_np = target_feat.detach().cpu().numpy()
            device = source_feat.device
            dtype = source_feat.dtype
        else:
            source_np = source_feat
            target_np = target_feat
            device = None
            dtype = None

        # Compute cost matrix (Euclidean distance)
        M = pot.utils.dist(source_np, target_np)

        # Uniform weights for samples
        n = len(source_np)
        a = np.ones(n) / n
        b = np.ones(n) / n

        # Solve OT problem with Sinkhorn algorithm (entropic regularization)
        P = pot.sinkhorn(a, b, M, self.reg)

        # Apply transport plan: aligned = P @ target
        aligned_np = P @ target_np

        # Convert back to torch
        if device is not None:
            aligned = torch.tensor(aligned_np, dtype=dtype, device=device)
        else:
            aligned = aligned_np

        return aligned


class SlicedWassersteinAlignment(nn.Module):
    """
    Sliced Wasserstein distance for efficient OT-based alignment.

    More efficient than full Wasserstein for high-dimensional features.
    Uses random projections to compute 1D Wasserstein distances.
    """

    def __init__(self, n_projections: int = 50):
        super().__init__()
        self.n_projections = n_projections

    def forward(self, source_feat, target_feat):
        """
        Align using sliced Wasserstein distance.

        Args:
            source_feat: Source features [batch, dim]
            target_feat: Target features [batch, dim]

        Returns:
            Aligned features
        """
        if isinstance(source_feat, torch.Tensor):
            source_np = source_feat.detach().cpu().numpy()
            target_np = target_feat.detach().cpu().numpy()
            device = source_feat.device
            dtype = source_feat.dtype
        else:
            source_np = source_feat
            target_np = target_feat
            device = None
            dtype = None

        # Compute sliced Wasserstein distance
        sw_dist = pot.sliced_wasserstein_distance(
            source_np, target_np,
            n_projections=self.n_projections
        )

        # Alignment: weighted average based on SW distance
        alpha = 1.0 / (1.0 + sw_dist + 1e-8)
        aligned_np = alpha * source_np + (1 - alpha) * target_np

        if device is not None:
            aligned = torch.tensor(aligned_np, dtype=dtype, device=device)
        else:
            aligned = aligned_np

        return aligned
