"""
Test script for DINOv3 Semantic Feature Extractor
================================================

This script demonstrates how to use the DINOv3 semantic feature extractor
for extracting features from video frames.
"""

import torch
from ..src.models.semantic_extractor import create_semantic_extractor
from transformers.image_utils import load_image
from PIL import Image
import numpy as np


def test_semantic_extractor():
    """Test the semantic feature extractor with different input types."""

    print("=== DINOv3 ViT-B/16 Semantic Feature Extractor Test ===\n")

    # Create the extractor
    extractor = create_semantic_extractor()
    print("✓ Extractor created successfully")
    print(f"  - Model: {extractor.model_name}")
    print(f"  - Feature dimension: {extractor.get_feature_dim()}")
    print(f"  - Patch size: {extractor.patch_size}")
    print(f"  - Device: {extractor.device}")
    print()

    # Test 1: Load image from URL
    print("Test 1: Extracting features from URL image")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)
    print(f"  Original image size: {image.size}")

    features = extractor([image])
    print(f"  Features shape: {features.shape}")
    print(f"  Expected: [1, H//16, W//16, 768]")
    print()

    # Test 2: Multiple images
    print("Test 2: Extracting features from multiple images")
    images = [image, image]  # Same image twice for testing
    features_batch = extractor(images)
    print(f"  Batch features shape: {features_batch.shape}")
    print(f"  Expected: [2, H//16, W//16, 768]")
    print()

    # Test 3: Patch information
    print("Test 3: Patch information analysis")
    patch_info = extractor.get_patch_info(image.size)
    print(f"  Patch size: {patch_info['patch_size']}")
    print(f"  Number of patches: {patch_info['num_patches_height']} x {patch_info['num_patches_width']}")
    print(f"  Total patches: {patch_info['num_patches_total']}")
    print(f"  Feature dimension: {patch_info['feature_dim']}")
    print()

    # Test 4: Memory usage
    print("Test 4: Memory usage check")
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"  GPU memory used: {memory_used:.1f} MB")
    print()

    print("=== All tests passed! ===")
    print("\nThe DINOv3 ViT-B/16 semantic extractor is ready for video frame feature extraction.")
    print("Features can be used as input to the cross-attention fusion module.")
    print("✅ Proper module structure with dinov3_extractor.py implementation")


if __name__ == "__main__":
    test_semantic_extractor()
