"""
Motion Feature Extractor Module
===============================

This module provides motion feature extraction capabilities using I3D.
"""

from .i3d_extractor import I3DMotionExtractor, create_motion_extractor

__all__ = [
    'I3DMotionExtractor',
    'create_motion_extractor'
]
