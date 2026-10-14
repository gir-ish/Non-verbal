"""MATA Model Architecture"""

import torch
import torch.nn as nn
from .fusion import CrossFoundationFusion


class MATAModel(nn.Module):
    """
    MATA: Modality-binding Audio Temporal Attention

    Synergizes modality-binding foundation models for non-verbal emotion recognition.

    Args:
        num_emotions: Number of emotion classes
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        num_emotions: int = 8,
        hidden_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        # Foundation model projections
        self.language_bind_proj = nn.Linear(768, hidden_dim)
        self.image_bind_proj = nn.Linear(768, hidden_dim)

        # Cross-foundation fusion
        self.fusion = CrossFoundationFusion(hidden_dim, num_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, language_bind_feat, image_bind_feat):
        """Forward pass for non-verbal emotion recognition"""
        # Project foundation model features
        language = self.language_bind_proj(language_bind_feat)
        image = self.image_bind_proj(image_bind_feat)

        # Fuse through cross-foundation module
        fused = self.fusion(language, image)

        # Global pooling if needed
        if fused.dim() > 2:
            fused = fused.mean(dim=1)

        # Classify
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=1),
            "confidence": torch.softmax(logits, dim=1).max(dim=1).values,
            "emotion": torch.argmax(logits, dim=1),
        }
