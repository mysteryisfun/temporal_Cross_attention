"""
Metadata Generator for GASM-CAST Architecture

This module generates universal spatial and temporal position arrays that are used
across all videos in the dataset. These positions are architecture-dependent
(based on DINOv3 and I3D structures) rather than content-dependent.

Key Components:
- Spatial positions for DINOv3 patches (14×14 grid = 196 patches)
- Temporal positions for I3D segments (16 frames → 8 segments)
- Graph construction utilities for attention mechanisms
"""

import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class MetadataGenerator:
    """Generates spatial and temporal metadata for GASM-CAST architecture"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # DINOv3 specifications
        self.dinov3_patch_size = 16  # ViT-B/16
        self.dinov3_image_size = 224
        self.dinov3_grid_size = 14   # 224 / 16 = 14
        self.dinov3_num_patches = 196  # 14 * 14
        
        # I3D specifications  
        self.i3d_input_frames = 16
        self.i3d_output_segments = 8
        
        # Pre-compute universal metadata
        self._spatial_positions = None
        self._temporal_positions = None
        self._spatial_distances = None
        self._temporal_distances = None
    
    def generate_spatial_positions(self, normalize: bool = True) -> torch.Tensor:
        """
        Generate spatial positions for DINOv3 patches in a 14×14 grid.
        
        Args:
            normalize: Whether to normalize coordinates to [0, 1] range
            
        Returns:
            torch.Tensor: [196, 2] tensor with (x, y) coordinates for each patch
        """
        if self._spatial_positions is not None:
            return self._spatial_positions
            
        # Create 14×14 grid coordinates
        x_coords = torch.arange(self.dinov3_grid_size, dtype=torch.float32)
        y_coords = torch.arange(self.dinov3_grid_size, dtype=torch.float32)
        
        # Create meshgrid
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Flatten to get [196, 2] shape
        spatial_positions = torch.stack([
            x_grid.flatten(),
            y_grid.flatten()
        ], dim=1)
        
        # Normalize coordinates to [0, 1] range if requested
        if normalize:
            spatial_positions = spatial_positions / (self.dinov3_grid_size - 1)
        
        # Move to device and cache
        self._spatial_positions = spatial_positions.to(self.device)
        
        print(f"✅ Generated spatial positions: {self._spatial_positions.shape}")
        print(f"   Coordinate range: [{self._spatial_positions.min():.3f}, {self._spatial_positions.max():.3f}]")
        
        return self._spatial_positions
    
    def generate_temporal_positions(self, normalize: bool = True) -> torch.Tensor:
        """
        Generate temporal positions for I3D segments mapping from 16 frames to 8 segments.
        
        Args:
            normalize: Whether to normalize temporal indices to [0, 1] range
            
        Returns:
            torch.Tensor: [8, 1] tensor with temporal indices for each segment
        """
        if self._temporal_positions is not None:
            return self._temporal_positions
            
        # Map 16 frames to 8 segments (each segment covers 2 frames)
        # Segment i covers frames [2*i, 2*i+1] and is centered at frame (2*i + 0.5)
        segment_centers = torch.arange(self.i3d_output_segments, dtype=torch.float32) * 2 + 0.5
        
        # Reshape to [8, 1] for consistency with spatial positions
        temporal_positions = segment_centers.unsqueeze(1)
        
        # Normalize to [0, 1] range if requested
        if normalize:
            temporal_positions = temporal_positions / (self.i3d_input_frames - 1)
        
        # Move to device and cache
        self._temporal_positions = temporal_positions.to(self.device)
        
        print(f"✅ Generated temporal positions: {self._temporal_positions.shape}")
        print(f"   Temporal range: [{self._temporal_positions.min():.3f}, {self._temporal_positions.max():.3f}]")
        
        return self._temporal_positions
    
    def compute_spatial_distances(self) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between all spatial positions.
        
        Returns:
            torch.Tensor: [196, 196] distance matrix
        """
        if self._spatial_distances is not None:
            return self._spatial_distances
            
        if self._spatial_positions is None:
            self.generate_spatial_positions()
        
        # Compute pairwise distances using broadcasting
        # positions: [196, 2]
        positions = self._spatial_positions.unsqueeze(0)  # [1, 196, 2]
        positions_T = self._spatial_positions.unsqueeze(1)  # [196, 1, 2]
        
        # Compute squared differences and sum over coordinate dimension
        diff_squared = (positions - positions_T) ** 2  # [196, 196, 2]
        distances = torch.sqrt(diff_squared.sum(dim=2))  # [196, 196]
        
        self._spatial_distances = distances
        
        print(f"✅ Computed spatial distance matrix: {distances.shape}")
        print(f"   Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
        
        return distances
    
    def compute_temporal_distances(self) -> torch.Tensor:
        """
        Compute pairwise temporal distances between all temporal positions.
        
        Returns:
            torch.Tensor: [8, 8] distance matrix
        """
        if self._temporal_distances is not None:
            return self._temporal_distances
            
        if self._temporal_positions is None:
            self.generate_temporal_positions()
        
        # Compute pairwise distances
        positions = self._temporal_positions.squeeze(1)  # [8]
        positions_expanded = positions.unsqueeze(0)  # [1, 8]
        positions_T_expanded = positions.unsqueeze(1)  # [8, 1]
        
        distances = torch.abs(positions_expanded - positions_T_expanded)  # [8, 8]
        
        self._temporal_distances = distances
        
        print(f"✅ Computed temporal distance matrix: {distances.shape}")
        print(f"   Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
        
        return distances
    
    def get_spatial_neighbors(self, k: int = 8, include_self: bool = False) -> torch.Tensor:
        """
        Get k-nearest spatial neighbors for each patch.
        
        Args:
            k: Number of neighbors to return
            include_self: Whether to include self in neighbors
            
        Returns:
            torch.Tensor: [196, k] indices of nearest neighbors
        """
        distances = self.compute_spatial_distances()
        
        # Get k+1 nearest neighbors (including self)
        _, neighbor_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        
        if not include_self:
            # Remove self (first column, distance = 0)
            neighbor_indices = neighbor_indices[:, 1:]
        else:
            neighbor_indices = neighbor_indices[:, :k]
            
        print(f"✅ Generated spatial k-NN graph: k={k}, include_self={include_self}")
        print(f"   Output shape: {neighbor_indices.shape}")
        
        return neighbor_indices
    
    def get_temporal_neighbors(self, k: int = 3, include_self: bool = False) -> torch.Tensor:
        """
        Get k-nearest temporal neighbors for each segment.
        
        Args:
            k: Number of neighbors to return
            include_self: Whether to include self in neighbors
            
        Returns:
            torch.Tensor: [8, k] indices of nearest neighbors
        """
        distances = self.compute_temporal_distances()
        
        # Get k+1 nearest neighbors (including self)
        _, neighbor_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        
        if not include_self:
            # Remove self (first column, distance = 0)
            neighbor_indices = neighbor_indices[:, 1:]
        else:
            neighbor_indices = neighbor_indices[:, :k]
            
        print(f"✅ Generated temporal k-NN graph: k={k}, include_self={include_self}")
        print(f"   Output shape: {neighbor_indices.shape}")
        
        return neighbor_indices
    
    def visualize_spatial_positions(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the spatial positions as a 2D grid.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if self._spatial_positions is None:
            self.generate_spatial_positions()
        
        positions = self._spatial_positions.cpu().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.7)
        
        # Add grid lines
        for i in range(self.dinov3_grid_size):
            plt.axhline(y=i/(self.dinov3_grid_size-1), color='gray', linestyle='--', alpha=0.3)
            plt.axvline(x=i/(self.dinov3_grid_size-1), color='gray', linestyle='--', alpha=0.3)
        
        plt.title('DINOv3 Spatial Positions (14×14 Patches)', fontsize=14)
        plt.xlabel('X Coordinate (Normalized)', fontsize=12)
        plt.ylabel('Y Coordinate (Normalized)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved spatial positions visualization to {save_path}")
        
        plt.show()
    
    def visualize_temporal_positions(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the temporal positions as a timeline.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if self._temporal_positions is None:
            self.generate_temporal_positions()
        
        positions = self._temporal_positions.cpu().numpy().flatten()
        
        plt.figure(figsize=(12, 4))
        
        # Plot segment positions
        plt.scatter(positions, [0.5] * len(positions), c='red', s=100, alpha=0.8, label='I3D Segments')
        
        # Add frame positions for reference
        frame_positions = np.arange(16) / 15  # Normalized frame positions
        plt.scatter(frame_positions, [0.3] * len(frame_positions), c='blue', s=30, alpha=0.5, label='Input Frames')
        
        # Connect segments to their contributing frames
        for i, seg_pos in enumerate(positions):
            frame_start = i * 2
            frame_end = i * 2 + 1
            plt.plot([frame_positions[frame_start], seg_pos], [0.3, 0.5], 'gray', alpha=0.3, linestyle=':')
            plt.plot([frame_positions[frame_end], seg_pos], [0.3, 0.5], 'gray', alpha=0.3, linestyle=':')
        
        plt.title('I3D Temporal Positions (16 Frames → 8 Segments)', fontsize=14)
        plt.xlabel('Temporal Position (Normalized)', fontsize=12)
        plt.ylabel('Level', fontsize=12)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved temporal positions visualization to {save_path}")
        
        plt.show()
    
    def get_metadata_summary(self) -> dict:
        """
        Get a comprehensive summary of all generated metadata.
        
        Returns:
            dict: Summary of metadata properties
        """
        summary = {
            'spatial': {
                'num_patches': self.dinov3_num_patches,
                'grid_size': f"{self.dinov3_grid_size}×{self.dinov3_grid_size}",
                'patch_size': self.dinov3_patch_size,
                'image_size': self.dinov3_image_size,
                'positions_shape': None,
                'distance_matrix_shape': None
            },
            'temporal': {
                'input_frames': self.i3d_input_frames,
                'output_segments': self.i3d_output_segments,
                'frames_per_segment': self.i3d_input_frames // self.i3d_output_segments,
                'positions_shape': None,
                'distance_matrix_shape': None
            },
            'device': str(self.device)
        }
        
        if self._spatial_positions is not None:
            summary['spatial']['positions_shape'] = list(self._spatial_positions.shape)
        if self._spatial_distances is not None:
            summary['spatial']['distance_matrix_shape'] = list(self._spatial_distances.shape)
        if self._temporal_positions is not None:
            summary['temporal']['positions_shape'] = list(self._temporal_positions.shape)
        if self._temporal_distances is not None:
            summary['temporal']['distance_matrix_shape'] = list(self._temporal_distances.shape)
        
        return summary


def create_metadata_generator(device: str = 'cuda') -> MetadataGenerator:
    """
    Factory function to create a MetadataGenerator instance.
    
    Args:
        device: Device to use for computations ('cuda' or 'cpu')
        
    Returns:
        MetadataGenerator: Configured metadata generator
    """
    # Auto-detect device if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print(f"⚠️  CUDA not available, using CPU instead")
    
    return MetadataGenerator(device=device)


if __name__ == "__main__":
    # Demo usage
    print("🚀 GASM-CAST Metadata Generator Demo")
    print("=" * 50)
    
    # Create generator
    generator = create_metadata_generator()
    
    # Generate all metadata
    spatial_pos = generator.generate_spatial_positions()
    temporal_pos = generator.generate_temporal_positions()
    
    # Compute distance matrices
    spatial_dist = generator.compute_spatial_distances()
    temporal_dist = generator.compute_temporal_distances()
    
    # Generate neighbor graphs
    spatial_neighbors = generator.get_spatial_neighbors(k=8)
    temporal_neighbors = generator.get_temporal_neighbors(k=3)
    
    # Print summary
    summary = generator.get_metadata_summary()
    print("\n📊 Metadata Summary:")
    print(f"Spatial: {summary['spatial']}")
    print(f"Temporal: {summary['temporal']}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    generator.visualize_spatial_positions("spatial_positions.png")
    generator.visualize_temporal_positions("temporal_positions.png")
    
    print("\n✅ Metadata generation complete!")
