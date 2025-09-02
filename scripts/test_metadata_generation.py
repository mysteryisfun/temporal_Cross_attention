"""
Comprehensive Tests for GASM-CAST Metadata Generator

This script thoroughly tests the metadata generation functionality to ensure
correctness and compatibility with the GASM-CAST architecture.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metadata_generator import create_metadata_generator


def test_spatial_positions():
    """Test spatial position generation for DINOv3 patches"""
    print("\n🔍 Testing Spatial Position Generation")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Test basic generation
    positions = generator.generate_spatial_positions()
    
    # Validate shape
    expected_shape = (196, 2)  # 14*14 patches, 2D coordinates
    assert positions.shape == expected_shape, f"Expected shape {expected_shape}, got {positions.shape}"
    print(f"✅ Shape validation passed: {positions.shape}")
    
    # Validate coordinate ranges (normalized to [0, 1])
    assert positions.min() >= 0.0, f"Minimum coordinate should be >= 0, got {positions.min()}"
    assert positions.max() <= 1.0, f"Maximum coordinate should be <= 1, got {positions.max()}"
    print(f"✅ Coordinate range validation passed: [{positions.min():.3f}, {positions.max():.3f}]")
    
    # Validate grid structure (should form perfect 14x14 grid)
    positions_cpu = positions.cpu().numpy()
    unique_x = len(np.unique(positions_cpu[:, 0]))
    unique_y = len(np.unique(positions_cpu[:, 1]))
    assert unique_x == 14, f"Expected 14 unique X coordinates, got {unique_x}"
    assert unique_y == 14, f"Expected 14 unique Y coordinates, got {unique_y}"
    print(f"✅ Grid structure validation passed: {unique_x}×{unique_y} unique coordinates")
    
    # Validate caching (second call should return same tensor)
    positions_2 = generator.generate_spatial_positions()
    assert torch.equal(positions, positions_2), "Cached positions should be identical"
    print(f"✅ Caching validation passed")
    
    # Test device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_device = str(positions.device).split(':')[0]  # Handle cuda:0 -> cuda
    assert actual_device == expected_device, f"Expected device {expected_device}, got {positions.device}"
    print(f"✅ Device placement validation passed: {positions.device}")


def test_temporal_positions():
    """Test temporal position generation for I3D segments"""
    print("\n🔍 Testing Temporal Position Generation")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Test basic generation
    positions = generator.generate_temporal_positions()
    
    # Validate shape
    expected_shape = (8, 1)  # 8 segments, 1D temporal coordinate
    assert positions.shape == expected_shape, f"Expected shape {expected_shape}, got {positions.shape}"
    print(f"✅ Shape validation passed: {positions.shape}")
    
    # Validate coordinate ranges (normalized to [0, 1])
    assert positions.min() >= 0.0, f"Minimum coordinate should be >= 0, got {positions.min()}"
    assert positions.max() <= 1.0, f"Maximum coordinate should be <= 1, got {positions.max()}"
    print(f"✅ Coordinate range validation passed: [{positions.min():.3f}, {positions.max():.3f}]")
    
    # Validate temporal ordering (should be monotonically increasing)
    positions_flat = positions.cpu().numpy().flatten()
    is_sorted = np.all(positions_flat[:-1] <= positions_flat[1:])
    assert is_sorted, "Temporal positions should be monotonically increasing"
    print(f"✅ Temporal ordering validation passed")
    
    # Validate segment mapping (8 segments from 16 frames)
    # Each segment should be centered at frame (2*i + 0.5) normalized by 15
    expected_positions = (np.arange(8) * 2 + 0.5) / 15
    actual_positions = positions_flat
    np.testing.assert_allclose(actual_positions, expected_positions, rtol=1e-5)
    print(f"✅ Segment mapping validation passed")
    
    # Test device placement
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_device = str(positions.device).split(':')[0]  # Handle cuda:0 -> cuda
    assert actual_device == expected_device, f"Expected device {expected_device}, got {positions.device}"
    print(f"✅ Device placement validation passed: {positions.device}")


def test_distance_matrices():
    """Test spatial and temporal distance matrix computation"""
    print("\n🔍 Testing Distance Matrix Computation")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Test spatial distances
    spatial_distances = generator.compute_spatial_distances()
    expected_shape = (196, 196)
    assert spatial_distances.shape == expected_shape, f"Expected shape {expected_shape}, got {spatial_distances.shape}"
    print(f"✅ Spatial distance matrix shape: {spatial_distances.shape}")
    
    # Validate symmetry
    assert torch.allclose(spatial_distances, spatial_distances.T), "Distance matrix should be symmetric"
    print(f"✅ Spatial distance matrix symmetry validated")
    
    # Validate diagonal (self-distances should be 0)
    diagonal = torch.diag(spatial_distances)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal)), "Diagonal should be all zeros"
    print(f"✅ Spatial distance matrix diagonal validated")
    
    # Test temporal distances
    temporal_distances = generator.compute_temporal_distances()
    expected_shape = (8, 8)
    assert temporal_distances.shape == expected_shape, f"Expected shape {expected_shape}, got {temporal_distances.shape}"
    print(f"✅ Temporal distance matrix shape: {temporal_distances.shape}")
    
    # Validate symmetry
    assert torch.allclose(temporal_distances, temporal_distances.T), "Distance matrix should be symmetric"
    print(f"✅ Temporal distance matrix symmetry validated")
    
    # Validate diagonal
    diagonal = torch.diag(temporal_distances)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal)), "Diagonal should be all zeros"
    print(f"✅ Temporal distance matrix diagonal validated")


def test_neighbor_graphs():
    """Test k-nearest neighbor graph generation"""
    print("\n🔍 Testing K-Nearest Neighbor Graphs")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Test spatial neighbors
    k_spatial = 8
    spatial_neighbors = generator.get_spatial_neighbors(k=k_spatial, include_self=False)
    expected_shape = (196, k_spatial)
    assert spatial_neighbors.shape == expected_shape, f"Expected shape {expected_shape}, got {spatial_neighbors.shape}"
    print(f"✅ Spatial neighbors shape: {spatial_neighbors.shape}")
    
    # Validate neighbor indices are within valid range
    assert spatial_neighbors.min() >= 0, f"Neighbor indices should be >= 0, got {spatial_neighbors.min()}"
    assert spatial_neighbors.max() < 196, f"Neighbor indices should be < 196, got {spatial_neighbors.max()}"
    print(f"✅ Spatial neighbor indices range: [{spatial_neighbors.min()}, {spatial_neighbors.max()}]")
    
    # Test temporal neighbors
    k_temporal = 3
    temporal_neighbors = generator.get_temporal_neighbors(k=k_temporal, include_self=False)
    expected_shape = (8, k_temporal)
    assert temporal_neighbors.shape == expected_shape, f"Expected shape {expected_shape}, got {temporal_neighbors.shape}"
    print(f"✅ Temporal neighbors shape: {temporal_neighbors.shape}")
    
    # Validate neighbor indices are within valid range
    assert temporal_neighbors.min() >= 0, f"Neighbor indices should be >= 0, got {temporal_neighbors.min()}"
    assert temporal_neighbors.max() < 8, f"Neighbor indices should be < 8, got {temporal_neighbors.max()}"
    print(f"✅ Temporal neighbor indices range: [{temporal_neighbors.min()}, {temporal_neighbors.max()}]")
    
    # Test with self inclusion
    spatial_neighbors_self = generator.get_spatial_neighbors(k=k_spatial, include_self=True)
    # Should have same shape but include self-connections
    expected_shape_self = (196, k_spatial)  # Same as spatial neighbors
    assert spatial_neighbors_self.shape == expected_shape_self, f"Expected shape {expected_shape_self}, got {spatial_neighbors_self.shape}"
    print(f"✅ Spatial neighbors with self inclusion: {spatial_neighbors_self.shape}")


def test_compatibility_with_existing_models():
    """Test compatibility with existing DINOv3 and I3D model outputs"""
    print("\n🔍 Testing Compatibility with Existing Models")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Simulate DINOv3 output: [batch_size, num_patches, feature_dim]
    batch_size = 4
    dinov3_output = torch.randn(batch_size, 196, 768)  # Real DINOv3 output shape
    spatial_positions = generator.generate_spatial_positions()
    
    print(f"DINOv3 output shape: {dinov3_output.shape}")
    print(f"Spatial positions shape: {spatial_positions.shape}")
    
    # Verify we can broadcast/combine them
    assert dinov3_output.shape[1] == spatial_positions.shape[0], \
        f"Number of patches mismatch: {dinov3_output.shape[1]} vs {spatial_positions.shape[0]}"
    print(f"✅ DINOv3 compatibility verified")
    
    # Simulate I3D output: [batch_size, num_segments, feature_dim]
    i3d_output = torch.randn(batch_size, 8, 512)  # Real I3D output shape
    temporal_positions = generator.generate_temporal_positions()
    
    print(f"I3D output shape: {i3d_output.shape}")
    print(f"Temporal positions shape: {temporal_positions.shape}")
    
    # Verify we can broadcast/combine them
    assert i3d_output.shape[1] == temporal_positions.shape[0], \
        f"Number of segments mismatch: {i3d_output.shape[1]} vs {temporal_positions.shape[0]}"
    print(f"✅ I3D compatibility verified")


def test_metadata_summary():
    """Test metadata summary generation"""
    print("\n🔍 Testing Metadata Summary")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Generate some metadata first
    generator.generate_spatial_positions()
    generator.generate_temporal_positions()
    generator.compute_spatial_distances()
    generator.compute_temporal_distances()
    
    # Get summary
    summary = generator.get_metadata_summary()
    
    # Validate structure
    assert 'spatial' in summary, "Summary should contain spatial information"
    assert 'temporal' in summary, "Summary should contain temporal information"
    assert 'device' in summary, "Summary should contain device information"
    print(f"✅ Summary structure validated")
    
    # Validate spatial information
    spatial_info = summary['spatial']
    assert spatial_info['num_patches'] == 196, f"Expected 196 patches, got {spatial_info['num_patches']}"
    assert spatial_info['grid_size'] == "14×14", f"Expected 14×14 grid, got {spatial_info['grid_size']}"
    print(f"✅ Spatial information validated")
    
    # Validate temporal information
    temporal_info = summary['temporal']
    assert temporal_info['input_frames'] == 16, f"Expected 16 frames, got {temporal_info['input_frames']}"
    assert temporal_info['output_segments'] == 8, f"Expected 8 segments, got {temporal_info['output_segments']}"
    print(f"✅ Temporal information validated")
    
    print(f"\n📊 Complete Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def create_test_visualizations():
    """Create test visualizations to manually verify correctness"""
    print("\n🎨 Creating Test Visualizations")
    print("-" * 40)
    
    generator = create_metadata_generator()
    
    # Create output directory
    output_dir = Path("experiments/metadata_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and visualize spatial positions
    spatial_pos = generator.generate_spatial_positions()
    generator.visualize_spatial_positions(str(output_dir / "spatial_positions_test.png"))
    
    # Generate and visualize temporal positions
    temporal_pos = generator.generate_temporal_positions()
    generator.visualize_temporal_positions(str(output_dir / "temporal_positions_test.png"))
    
    # Create neighbor graph visualization
    spatial_neighbors = generator.get_spatial_neighbors(k=8, include_self=False)
    
    # Plot spatial neighbor connections for a few sample patches
    plt.figure(figsize=(10, 10))
    positions = spatial_pos.cpu().numpy()
    
    # Plot all patches
    plt.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=30, alpha=0.6, label='All Patches')
    
    # Highlight a few patches and their neighbors
    sample_patches = [97, 98, 99]  # Center patches
    colors = ['red', 'green', 'purple']
    
    for i, (patch_idx, color) in enumerate(zip(sample_patches, colors)):
        # Highlight the patch
        plt.scatter(positions[patch_idx, 0], positions[patch_idx, 1], 
                   c=color, s=100, marker='o', label=f'Patch {patch_idx}')
        
        # Draw connections to neighbors
        neighbors = spatial_neighbors[patch_idx].cpu().numpy()
        for neighbor_idx in neighbors:
            plt.plot([positions[patch_idx, 0], positions[neighbor_idx, 0]], 
                    [positions[patch_idx, 1], positions[neighbor_idx, 1]], 
                    color=color, alpha=0.4, linewidth=1)
    
    plt.title('Spatial K-Nearest Neighbor Connections (k=8)', fontsize=14)
    plt.xlabel('X Coordinate (Normalized)', fontsize=12)
    plt.ylabel('Y Coordinate (Normalized)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(output_dir / "spatial_neighbors_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizations saved to {output_dir}/")


def run_all_tests():
    """Run all metadata generator tests"""
    print("🚀 Running GASM-CAST Metadata Generator Tests")
    print("=" * 60)
    
    try:
        test_spatial_positions()
        test_temporal_positions() 
        test_distance_matrices()
        test_neighbor_graphs()
        test_compatibility_with_existing_models()
        test_metadata_summary()
        create_test_visualizations()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Metadata generator is working correctly.")
        print("✅ Ready for graph attention implementation.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
