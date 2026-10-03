"""Frame Interpolation module for satellite TIR imagery.
Provides: dataset loading, model, training, inference and I/O utilites for
AI/ML- based temporal super-resolution of geostationary satellite imagery.
"""

from .models.rife_model import SatelliteRIFE
__all__=["SatelliteRIFE"]
