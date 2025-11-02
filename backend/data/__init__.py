"""
Data ingestion module for stellar light curves.
Handles FITS and CSV file formats.
"""

from .loader import LightCurveLoader

__all__ = ['LightCurveLoader']
