"""Cross-Foundation Model Fusion"""

import torch
import torch.nn as nn
from .attention import TemporalAttention
from .transport import OptimalTransportAlignment


class CrossFoundationFusion(nn.Module):
    """
    Fuses LanguageBind and ImageBind through optimal transport and attention.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model

        # Temporal attention for each foundation model
        self.language_attention = TemporalAttention(d_model, num_heads)
        self.image_attention = TemporalAttention(d_model, num_heads)

        # Optimal transport alignment
        self.ot_alignment = OptimalTransportAlignment()

        # Gating networks for adaptive weighting
        self.gate_language = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.gate_image = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, language_feat, image_feat):
        """Fuse modality-binding foundation models"""
        # Add temporal dimension if needed
        if language_feat.dim() == 2:
            language_feat = language_feat.unsqueeze(1)
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(1)

        # Apply temporal attention
        language = self.language_attention(language_feat)
        image = self.image_attention(image_feat)

        # Optimal transport alignment
        aligned = self.ot_alignment(language, image)

        # Gated fusion
        if language_feat.size(1) == 1:
            gate_l = self.gate_language(language.squeeze(1))
            gate_i = self.gate_image(image.squeeze(1))
            fused = gate_l.unsqueeze(-1) * language.squeeze(1) + (1 - gate_i.unsqueeze(-1)) * image.squeeze(1)
        else:
            gate_l = self.gate_language(language)
            gate_i = self.gate_image(image)
            fused = gate_l * language + (1 - gate_i) * image

        return fused
