"""
ASCII Visualization of GASM-CAST Metadata

Creates simple ASCII art visualizations to show the spatial and temporal structure.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metadata_generator import create_metadata_generator


def print_spatial_grid():
    """Print ASCII representation of spatial 14x14 grid"""
    print("🧩 SPATIAL GRID VISUALIZATION (14×14 DINOv3 Patches)")
    print("=" * 60)
    
    generator = create_metadata_generator()
    positions = generator.generate_spatial_positions()
    neighbors = generator.get_spatial_neighbors(k=8)
    
    # Create grid representation
    print("Grid Layout (Patch IDs):")
    print("    " + "".join([f"{i:3d} " for i in range(14)]))
    print("  +" + "-" * 56 + "+")
    
    for row in range(14):
        row_str = f"{row:2d}| "
        for col in range(14):
            patch_id = row * 14 + col
            row_str += f"{patch_id:3d} "
        row_str += "|"
        print(row_str)
    
    print("  +" + "-" * 56 + "+")
    
    # Show neighbor connections for center patch
    center_patch = 97  # Approximately center
    center_neighbors = neighbors[center_patch].cpu().numpy()
    
    print(f"\nCenter Patch Connectivity (Patch {center_patch}):")
    print("    " + "".join([f"{i:3d} " for i in range(14)]))
    print("  +" + "-" * 56 + "+")
    
    for row in range(14):
        row_str = f"{row:2d}| "
        for col in range(14):
            patch_id = row * 14 + col
            if patch_id == center_patch:
                row_str += " ★  "  # Center patch
            elif patch_id in center_neighbors:
                row_str += " ●  "  # Neighbor
            else:
                row_str += " ·  "  # Other patch
        row_str += "|"
        print(row_str)
    
    print("  +" + "-" * 56 + "+")
    print("Legend: ★ = Center patch, ● = k=8 neighbors, · = other patches")


def print_temporal_timeline():
    """Print ASCII representation of temporal segments"""
    print("\n🕐 TEMPORAL TIMELINE VISUALIZATION (16 Frames → 8 Segments)")
    print("=" * 70)
    
    generator = create_metadata_generator()
    positions = generator.generate_temporal_positions()
    neighbors = generator.get_temporal_neighbors(k=3)
    
    print("Frame Timeline:")
    print("Frames: " + "".join([f"F{i:2d}" for i in range(16)]))
    print("        " + "".join(["───" for i in range(16)]))
    
    print("\nSegment Mapping:")
    segment_line = "        "
    for seg in range(8):
        segment_line += f"S{seg}─"
        if seg < 7:
            segment_line += "─"
    print(segment_line)
    
    # Show which frames contribute to which segments
    print("\nFrame → Segment Mapping:")
    for seg in range(8):
        frame_start = seg * 2
        frame_end = seg * 2 + 1
        neighbor_list = neighbors[seg].cpu().numpy()
        print(f"  Segment {seg}: Frames {frame_start:2d}-{frame_end:2d} | Neighbors: {neighbor_list}")
    
    # ASCII art of temporal graph
    print(f"\nTemporal Graph Connectivity:")
    print("Segments: S0   S1   S2   S3   S4   S5   S6   S7")
    print("          │    │    │    │    │    │    │    │")
    
    # Show connections
    for seg in range(8):
        connection_line = "          "
        neighbor_list = neighbors[seg].cpu().numpy()
        for i in range(8):
            if i == seg:
                connection_line += "●────"
            elif i in neighbor_list:
                connection_line += "●────"
            else:
                connection_line += "     "
        print(f"S{seg} ──────{connection_line}")


def print_distance_heatmap():
    """Print ASCII heatmap of distance matrices"""
    print("\n📏 DISTANCE MATRICES (ASCII Heatmap)")
    print("=" * 50)
    
    generator = create_metadata_generator()
    spatial_distances = generator.compute_spatial_distances()
    temporal_distances = generator.compute_temporal_distances()
    
    # Temporal distance heatmap (8x8)
    print("🕐 Temporal Distance Matrix (8×8):")
    temp_dist = temporal_distances.cpu().numpy()
    
    print("     S0   S1   S2   S3   S4   S5   S6   S7")
    for i in range(8):
        row_str = f"S{i}  "
        for j in range(8):
            distance = temp_dist[i, j]
            if distance == 0:
                symbol = "██"  # Same segment
            elif distance < 0.2:
                symbol = "▓▓"  # Very close
            elif distance < 0.4:
                symbol = "▒▒"  # Close
            elif distance < 0.6:
                symbol = "░░"  # Medium
            else:
                symbol = "  "  # Far
            row_str += f" {symbol} "
        print(row_str)
    
    print("\nLegend: ██ = Same (0.0), ▓▓ = Very close (<0.2), ▒▒ = Close (<0.4), ░░ = Medium (<0.6),    = Far (≥0.6)")
    
    # Spatial distance sample (too big for full matrix, show 8x8 sample)
    print(f"\n🧩 Spatial Distance Sample (Center 8×8 region):")
    spatial_dist = spatial_distances.cpu().numpy()
    center_start = 94  # Around center
    
    print("     " + "".join([f"P{center_start+j:2d} " for j in range(8)]))
    for i in range(8):
        row_str = f"P{center_start+i:2d} "
        for j in range(8):
            distance = spatial_dist[center_start+i, center_start+j]
            if distance == 0:
                symbol = "██"
            elif distance < 0.1:
                symbol = "▓▓"
            elif distance < 0.2:
                symbol = "▒▒"
            elif distance < 0.4:
                symbol = "░░"
            else:
                symbol = "  "
            row_str += f" {symbol} "
        print(row_str)


def print_gasm_cast_architecture():
    """Print ASCII representation of GASM-CAST architecture"""
    print("\n🏗️  GASM-CAST ARCHITECTURE OVERVIEW")
    print("=" * 70)
    
    architecture = """
    INPUT VIDEO (16 frames, 224×224)
            │
    ┌───────┴──────┐
    │              │
    ▼              ▼
DINOv3          I3D/R3D-18
(Frame 0)       (All 16 frames)
    │              │
    ▼              ▼
[196, 768]      [512] → [8, 64]
Semantic        Motion
Features        Features
    │              │
    ▼              ▼
┌─────────┐  ┌─────────┐
│ SPATIAL │  │TEMPORAL │
│ GRAPH   │  │ GRAPH   │  ← Universal Metadata
│ k=8     │  │ k=3     │
└─────────┘  └─────────┘
    │              │
    ▼              ▼
[196, 768]      [8, 64]
Enhanced        Enhanced
Semantic        Motion
    │              │
    └──────┬───────┘
           ▼
    ┌─────────────┐
    │   B-CAST    │  ← Bottleneck Cross-Attention
    │ FUSION      │    (768D/64D → 256D → Cross-Attn)
    │ 3 Layers    │
    └─────────────┘
           │
           ▼
    [768+64] → [174]
    Classification
           │
           ▼
    Action Prediction
    """
    
    print(architecture)
    
    print("\n📊 Key Statistics:")
    print(f"  • Spatial patches: 196 (14×14 grid)")
    print(f"  • Temporal segments: 8 (from 16 frames)")
    print(f"  • Spatial neighbors per patch: k=8 (4% connectivity)")
    print(f"  • Temporal neighbors per segment: k=3 (37.5% connectivity)")
    print(f"  • B-CAST compression: 768D/64D → 256D (50% computation reduction)")
    print(f"  • Total metadata memory: ~0.15 MB")
    print(f"  • Target accuracy: 76-80% on Something-Something-V2")


def main():
    """Print all ASCII visualizations"""
    print("🎨 GASM-CAST ASCII VISUALIZATION SUITE")
    print("═" * 70)
    
    print_spatial_grid()
    print_temporal_timeline()
    print_distance_heatmap()
    print_gasm_cast_architecture()
    
    print(f"\n🎉 ASCII VISUALIZATION COMPLETE")
    print(f"✅ Visual structure of GASM-CAST metadata displayed")
    print(f"📈 Ready to proceed with graph attention implementation!")


if __name__ == "__main__":
    main()
