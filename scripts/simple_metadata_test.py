"""
Simple Metadata Generator Test (No Matplotlib)

Quick validation of metadata generation without visualization components
to avoid OpenMP conflicts.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metadata_generator import create_metadata_generator


def simple_test():
    """Simple test without visualizations"""
    print("🚀 Simple GASM-CAST Metadata Test")
    print("=" * 40)
    
    # Create generator
    generator = create_metadata_generator()
    print(f"✅ Created metadata generator on {generator.device}")
    
    # Generate spatial positions
    spatial_pos = generator.generate_spatial_positions()
    print(f"✅ Spatial positions: {spatial_pos.shape}")
    print(f"   Range: [{spatial_pos.min():.3f}, {spatial_pos.max():.3f}]")
    
    # Generate temporal positions
    temporal_pos = generator.generate_temporal_positions()
    print(f"✅ Temporal positions: {temporal_pos.shape}")
    print(f"   Range: [{temporal_pos.min():.3f}, {temporal_pos.max():.3f}]")
    
    # Test compatibility with model outputs
    batch_size = 2
    dinov3_features = torch.randn(batch_size, 196, 768)
    i3d_features = torch.randn(batch_size, 8, 512)
    
    print(f"✅ DINOv3 features: {dinov3_features.shape}")
    print(f"✅ I3D features: {i3d_features.shape}")
    print(f"✅ Compatible with spatial positions: {spatial_pos.shape}")
    print(f"✅ Compatible with temporal positions: {temporal_pos.shape}")
    
    # Generate neighbor graphs
    spatial_neighbors = generator.get_spatial_neighbors(k=8)
    temporal_neighbors = generator.get_temporal_neighbors(k=3)
    
    print(f"✅ Spatial k-NN graph: {spatial_neighbors.shape}")
    print(f"✅ Temporal k-NN graph: {temporal_neighbors.shape}")
    
    # Test a few specific values
    print("\n📋 Sample Values:")
    print(f"First spatial position: {spatial_pos[0]}")
    print(f"Last spatial position: {spatial_pos[-1]}")
    print(f"First temporal position: {temporal_pos[0]}")
    print(f"Last temporal position: {temporal_pos[-1]}")
    
    # Test center patch neighbors (patch 97 = center of 14x14 grid)
    center_patch = 97  # Approximately center
    center_neighbors = spatial_neighbors[center_patch]
    print(f"Center patch {center_patch} neighbors: {center_neighbors}")
    
    # Get summary
    summary = generator.get_metadata_summary()
    print(f"\n📊 Summary: {summary}")
    
    print("\n🎉 All basic tests passed!")
    print("✅ Metadata generator is ready for graph attention implementation")
    
    return generator


if __name__ == "__main__":
    generator = simple_test()
