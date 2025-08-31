"""
Semantic Feature Extractor Module
=================================

This module provides semantic feature extraction capabilities using DINOv3.
"""

from .dinov3_extractor import DINOv3SemanticExtractor, create_semantic_extractor

__all__ = [
    'DINOv3SemanticExtractor',
    'create_semantic_extractor'
]
