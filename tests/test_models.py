"""Unit tests for MATA models"""

import torch
from mata import MATAModel


def test_mata_model():
    """Test MATA model initialization and forward pass"""
    model = MATAModel(num_emotions=8)

    # Create dummy inputs
    language_bind = torch.randn(4, 768)
    image_bind = torch.randn(4, 768)

    # Forward pass
    output = model(language_bind, image_bind)

    assert output['logits'].shape == (4, 8)
    assert output['predictions'].shape == (4,)
    assert output['confidence'].shape == (4,)
    print("✓ MATA model test passed")


if __name__ == '__main__':
    test_mata_model()
