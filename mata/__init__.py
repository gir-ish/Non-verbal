"""MATA: Non-Verbal Emotion Recognition via Modality-Binding Foundation Models"""

__version__ = "1.0.0"

from .models import MATAModel
from .fusion import CrossFoundationFusion
from .transport import OptimalTransportAlignment

__all__ = ["MATAModel", "CrossFoundationFusion", "OptimalTransportAlignment"]
