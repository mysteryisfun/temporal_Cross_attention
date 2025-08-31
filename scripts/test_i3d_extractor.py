"""
Test I3D Motion Feature Extractor with Real Video Data
=====================================================

This script tests the I3D motion extractor with a random video from the dataset.
It will:
1. Pick a random video from data/raw/videos
2. Extract 8 frames uniformly
3. Save the frames as images
4. Extract motion features
5. Save features to file
"""

import os
import torch
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from src.models.motion_extractor import create_motion_extractor
from PIL import Image


def find_random_video(data_dir: str) -> str:
    """Find a random video file in the dataset."""
    video_dir = Path(data_dir) / "raw" / "videos"
    
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.webm', '.mkv', '.mov']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f"**/*{ext}")))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    
    # Pick random video
    random_video = random.choice(video_files)
    print(f"Selected random video: {random_video}")
    return str(random_video)


def save_frames(frames, output_dir: str, video_name: str):
    """Save extracted frames as images."""
    frames_dir = Path(output_dir) / "frames" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i:02d}.jpg"
        frame.save(frame_path)
        saved_paths.append(frame_path)
        print(f"  Saved frame {i+1}: {frame_path}")
    
    return saved_paths


def save_features(features, output_dir: str, video_name: str):
    """Save extracted features."""
    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    features_path = features_dir / f"{video_name}_motion_features.npy"
    np.save(features_path, features.cpu().numpy())
    
    # Save as text for inspection
    features_txt = features_dir / f"{video_name}_motion_features.txt"
    with open(features_txt, 'w') as f:
        f.write(f"Motion Features for {video_name}\n")
        f.write(f"Shape: {features.shape}\n")
        f.write(f"Min: {features.min().item():.6f}\n")
        f.write(f"Max: {features.max().item():.6f}\n")
        f.write(f"Mean: {features.mean().item():.6f}\n")
        f.write(f"Std: {features.std().item():.6f}\n")
        f.write(f"\nFirst 10 feature values:\n")
        f.write(str(features[0, :10].cpu().numpy()))
    
    print(f"  Features saved to: {features_path}")
    print(f"  Feature stats saved to: {features_txt}")
    
    return features_path, features_txt


def create_visualization(frames, features, output_dir: str, video_name: str):
    """Create visualization of frames and feature statistics."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'I3D Motion Feature Extraction - {video_name}', fontsize=16)
    
    # Plot frames
    for i, frame in enumerate(frames):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(frame)
        axes[row, col].set_title(f'Frame {i+1}')
        axes[row, col].axis('off')
    
    # Save visualization
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / f"{video_name}_frames.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature histogram
    plt.figure(figsize=(10, 6))
    plt.hist(features.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Motion Feature Distribution - {video_name}')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    hist_path = viz_dir / f"{video_name}_feature_hist.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to: {viz_path}")
    print(f"  Feature histogram saved to: {hist_path}")
    
    return viz_path, hist_path


def test_i3d_extractor():
    """Main test function."""
    print("=== I3D Motion Feature Extractor Test ===\n")
    
    # Setup
    data_dir = "data"
    output_dir = "experiments/i3d_test"
    
    # Create motion extractor
    print("1. Creating I3D motion extractor...")
    extractor = create_motion_extractor()
    print(f"   ✓ Model: {extractor.model_type}")
    print(f"   ✓ Feature dimension: {extractor.get_feature_dim()}")
    print(f"   ✓ Device: {extractor.device}")
    print()
    
    # Find random video
    print("2. Finding random video from dataset...")
    try:
        video_path = find_random_video(data_dir)
        video_name = Path(video_path).stem
        print(f"   ✓ Video: {video_name}")
        print()
    except FileNotFoundError as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Extract features
    print("3. Extracting motion features...")
    try:
        features, frames = extractor(video_path)
        print(f"   ✓ Features shape: {features.shape}")
        print(f"   ✓ Number of frames: {len(frames)}")
        print(f"   ✓ Feature stats:")
        print(f"     - Min: {features.min().item():.6f}")
        print(f"     - Max: {features.max().item():.6f}")
        print(f"     - Mean: {features.mean().item():.6f}")
        print(f"     - Std: {features.std().item():.6f}")
        print()
    except Exception as e:
        print(f"   ✗ Error extracting features: {e}")
        return
    
    # Save frames
    print("4. Saving extracted frames...")
    try:
        frame_paths = save_frames(frames, output_dir, video_name)
        print(f"   ✓ Saved {len(frame_paths)} frames")
        print()
    except Exception as e:
        print(f"   ✗ Error saving frames: {e}")
        return
    
    # Save features
    print("5. Saving motion features...")
    try:
        feature_paths = save_features(features, output_dir, video_name)
        print(f"   ✓ Features saved")
        print()
    except Exception as e:
        print(f"   ✗ Error saving features: {e}")
        return
    
    # Create visualizations
    print("6. Creating visualizations...")
    try:
        viz_paths = create_visualization(frames, features, output_dir, video_name)
        print(f"   ✓ Visualizations created")
        print()
    except Exception as e:
        print(f"   ✗ Error creating visualizations: {e}")
        return
    
    print("=== Test completed successfully! ===")
    print(f"\nResults saved to: {output_dir}/")
    print(f"- Frames: {output_dir}/frames/{video_name}/")
    print(f"- Features: {output_dir}/features/")
    print(f"- Visualizations: {output_dir}/visualizations/")


if __name__ == "__main__":
    test_i3d_extractor()
