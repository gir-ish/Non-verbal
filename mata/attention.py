"""Temporal Attention Module for Non-Verbal Audio"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention for non-verbal audio sequences.

    Captures temporal dependencies in vocalizations.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Apply temporal attention.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            Attended features
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)
