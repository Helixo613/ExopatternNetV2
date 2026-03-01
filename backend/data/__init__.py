"""
Data ingestion module for stellar light curves.
Handles FITS and CSV file formats, plus MAST archive downloads.
"""

from .loader import LightCurveLoader

try:
    from .acquisition import MastDataAcquisitor
    __all__ = ['LightCurveLoader', 'MastDataAcquisitor']
except ImportError:
    # lightkurve/astroquery not installed — acquisition not available
    __all__ = ['LightCurveLoader']
