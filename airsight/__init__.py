"""
AirSight - Visual Visibility & Haze Intelligence System

A sensor-free image and video processing system that estimates visual visibility
degradation by measuring contrast loss, edge attenuation, color distortion, haze,
and structural decay.
"""

__version__ = "1.0.0"
__author__ = "AirSight Team"

from .core.visibility_analyzer import VisibilityAnalyzer
from .core.video_processor import VideoProcessor

__all__ = ['VisibilityAnalyzer', 'VideoProcessor']
