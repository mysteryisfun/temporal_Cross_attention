#!/usr/bin/env python3
"""
Temporal Motion Visualization Script

This script visualizes the temporal dynamics of motion features extracted by I3D.
It shows how motion patterns change over time in the video sequence.

Usage:
    python scripts/visualize_temporal_motion.py --video path/to/video.webm --features path/to/features.npy
    python scripts/visualize_temporal_motion.py --directory experiments/i3d_test/
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path
import glob
from typing import List, Optional, Tuple
import seaborn as sns

class TemporalMotionVisualizer:
    """Visualizer for temporal motion dynamics"""
    
    def __init__(self):
        """Initialize the temporal visualizer"""
        plt.style.use('dark_background')
        self.colors = plt.cm.plasma(np.linspace(0, 1, 8))  # 8 frames
    
    def load_video_frames(self, video_path: str, max_frames: int = 8) -> List[np.ndarray]:
        """
        Load frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def load_saved_frames(self, frames_dir: str) -> List[np.ndarray]:
        """
        Load frames from saved frame directory
        
        Args:
            frames_dir: Directory containing saved frames
            
        Returns:
            List of frame arrays
        """
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        frames = []
        
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames
    
    def create_motion_flow_visualization(self, frames: List[np.ndarray], 
                                       features: np.ndarray, 
                                       video_name: str) -> None:
        """
        Create visualization showing temporal motion flow
        
        Args:
            frames: List of video frames
            features: Motion features array (1, 512) or (512,)
            video_name: Name of the video for title
        """
        if features.ndim == 2:
            features = features.flatten()
        
        # Reshape features to simulate temporal dynamics (8 frames x 64 motion channels)
        temporal_features = features.reshape(8, 64)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 8, height_ratios=[2, 1, 2], hspace=0.3, wspace=0.1)
        
        # Title
        fig.suptitle(f'Temporal Motion Dynamics - {video_name}', 
                    fontsize=20, fontweight='bold', color='white')
        
        # Top row: Original frames
        frame_axes = []
        for i in range(8):
            ax = fig.add_subplot(gs[0, i])
            if i < len(frames):
                ax.imshow(frames[i])
                ax.set_title(f'Frame {i+1}', color='white', fontsize=10)
            ax.axis('off')
            frame_axes.append(ax)
        
        # Middle row: Motion intensity timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        
        # Calculate motion intensity for each temporal segment
        motion_intensity = np.mean(temporal_features, axis=1)  # Average across channels
        frame_numbers = np.arange(1, 9)
        
        # Create motion intensity plot with gradient
        bars = ax_timeline.bar(frame_numbers, motion_intensity, 
                              color=self.colors, alpha=0.8, width=0.8)
        
        # Add motion flow arrows
        for i in range(7):
            arrow_height = (motion_intensity[i] + motion_intensity[i+1]) / 2
            ax_timeline.annotate('', xy=(i+2, arrow_height), xytext=(i+1, arrow_height),
                               arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
        
        ax_timeline.set_xlabel('Frame Number', color='white', fontsize=12)
        ax_timeline.set_ylabel('Motion Intensity', color='white', fontsize=12)
        ax_timeline.set_title('Temporal Motion Flow', color='white', fontsize=14)
        ax_timeline.tick_params(colors='white')
        ax_timeline.grid(True, alpha=0.3)
        ax_timeline.set_facecolor('black')
        
        # Bottom row: Motion feature heatmaps for each frame
        for i in range(8):
            ax = fig.add_subplot(gs[2, i])
            
            # Reshape frame features to 8x8 grid for visualization
            frame_features = temporal_features[i].reshape(8, 8)
            
            im = ax.imshow(frame_features, cmap='hot', aspect='auto')
            ax.set_title(f'Motion\nFeatures {i+1}', color='white', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add border with frame color
            border = Rectangle((0, 0), 7, 7, linewidth=3, 
                             edgecolor=self.colors[i], facecolor='none')
            ax.add_patch(border)
        
        plt.tight_layout()
        plt.show()
    
    def create_animated_motion_evolution(self, frames: List[np.ndarray], 
                                       features: np.ndarray, 
                                       video_name: str) -> None:
        """
        Create animated visualization of motion evolution
        
        Args:
            frames: List of video frames
            features: Motion features array
            video_name: Name of the video
        """
        if features.ndim == 2:
            features = features.flatten()
        
        temporal_features = features.reshape(8, 64)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Animated Motion Evolution - {video_name}', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Initialize plots
        ax_frame = axes[0, 0]
        ax_features = axes[0, 1]
        ax_timeline = axes[1, 0]
        ax_vector = axes[1, 1]
        
        # Frame display
        frame_img = ax_frame.imshow(frames[0] if frames else np.zeros((224, 224, 3)))
        ax_frame.set_title('Current Frame', color='white')
        ax_frame.axis('off')
        
        # Feature heatmap
        feature_2d = temporal_features[0].reshape(8, 8)
        feature_img = ax_features.imshow(feature_2d, cmap='hot', aspect='auto')
        ax_features.set_title('Motion Features', color='white')
        ax_features.set_xticks([])
        ax_features.set_yticks([])
        
        # Timeline with current position
        motion_intensity = np.mean(temporal_features, axis=1)
        bars = ax_timeline.bar(range(8), motion_intensity, color='gray', alpha=0.5)
        current_bar = ax_timeline.bar([0], [motion_intensity[0]], color='red')
        ax_timeline.set_title('Motion Timeline', color='white')
        ax_timeline.set_xlabel('Frame', color='white')
        ax_timeline.set_ylabel('Intensity', color='white')
        ax_timeline.tick_params(colors='white')
        
        # Motion vector visualization
        angles = np.linspace(0, 2*np.pi, 64)
        radii = temporal_features[0]
        ax_vector.scatter(angles, radii, c=radii, cmap='plasma', s=20)
        ax_vector.set_title('Motion Vectors', color='white')
        ax_vector.tick_params(colors='white')
        
        def animate(frame_idx):
            """Animation function"""
            # Update frame
            if frame_idx < len(frames):
                frame_img.set_array(frames[frame_idx])
            
            # Update features
            feature_2d = temporal_features[frame_idx].reshape(8, 8)
            feature_img.set_array(feature_2d)
            
            # Update timeline
            current_bar[0].set_height(motion_intensity[frame_idx])
            current_bar[0].set_x(frame_idx - 0.4)
            
            # Update motion vectors
            ax_vector.clear()
            radii = temporal_features[frame_idx]
            ax_vector.scatter(angles, radii, c=radii, cmap='plasma', s=20)
            ax_vector.set_title(f'Motion Vectors - Frame {frame_idx+1}', color='white')
            ax_vector.tick_params(colors='white')
            
            return [frame_img, feature_img]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=8, 
                                     interval=800, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_optical_flow_style_viz(self, frames: List[np.ndarray], 
                                    features: np.ndarray, 
                                    video_name: str) -> None:
        """
        Create optical flow style visualization
        
        Args:
            frames: List of video frames
            features: Motion features array
            video_name: Name of the video
        """
        if features.ndim == 2:
            features = features.flatten()
        
        temporal_features = features.reshape(8, 64)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Motion Flow Analysis - {video_name}', 
                    fontsize=16, fontweight='bold', color='white')
        
        for i in range(8):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            if i < len(frames):
                # Show frame with motion overlay
                ax.imshow(frames[i], alpha=0.7)
                
                # Create motion field visualization
                frame_features = temporal_features[i].reshape(8, 8)
                
                # Create coordinate grids
                y, x = np.mgrid[0:8, 0:8]
                x_scaled = x * (frames[i].shape[1] / 8)
                y_scaled = y * (frames[i].shape[0] / 8)
                
                # Use features as motion vectors
                u = frame_features * np.cos(np.linspace(0, 2*np.pi, 8*8).reshape(8, 8))
                v = frame_features * np.sin(np.linspace(0, 2*np.pi, 8*8).reshape(8, 8))
                
                # Scale vectors
                u = u * 20
                v = v * 20
                
                # Draw motion vectors
                ax.quiver(x_scaled, y_scaled, u, v, 
                         frame_features, cmap='hot', alpha=0.8,
                         scale=200, width=0.003)
            
            ax.set_title(f'Frame {i+1} Motion', color='white', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize temporal motion dynamics')
    
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--features', type=str, help='Path to motion features .npy file')
    parser.add_argument('--directory', type=str, help='Path to experiment directory (contains both video and features)')
    parser.add_argument('--frames-dir', type=str, help='Path to saved frames directory')
    
    args = parser.parse_args()
    
    visualizer = TemporalMotionVisualizer()
    
    try:
        if args.directory:
            # Auto-find files in experiment directory
            video_files = glob.glob(os.path.join(args.directory, "**/*.webm"), recursive=True)
            feature_files = glob.glob(os.path.join(args.directory, "**/*_motion_features.npy"), recursive=True)
            frame_dirs = glob.glob(os.path.join(args.directory, "**/frames/*"), recursive=True)
            
            if not feature_files:
                print(f"No motion feature files found in {args.directory}")
                return 1
            
            features = np.load(feature_files[0])
            video_name = Path(feature_files[0]).stem.replace('_motion_features', '')
            
            # Try to load frames
            frames = []
            if frame_dirs:
                frames_dir = os.path.dirname(frame_dirs[0])
                video_frames_dir = os.path.join(frames_dir, video_name)
                if os.path.exists(video_frames_dir):
                    frames = visualizer.load_saved_frames(video_frames_dir)
            elif video_files:
                frames = visualizer.load_video_frames(video_files[0])
            
        elif args.features:
            features = np.load(args.features)
            video_name = Path(args.features).stem.replace('_motion_features', '')
            
            frames = []
            if args.video:
                frames = visualizer.load_video_frames(args.video)
            elif args.frames_dir:
                frames = visualizer.load_saved_frames(args.frames_dir)
        
        else:
            print("Please provide --directory or --features argument")
            return 1
        
        print(f"Loaded features: {features.shape}")
        print(f"Loaded frames: {len(frames)}")
        
        # Create visualizations
        print("\n=== Creating Motion Flow Visualization ===")
        visualizer.create_motion_flow_visualization(frames, features, video_name)
        
        print("\n=== Creating Optical Flow Style Visualization ===")
        visualizer.create_optical_flow_style_viz(frames, features, video_name)
        
        print("\n=== Creating Animated Motion Evolution ===")
        anim = visualizer.create_animated_motion_evolution(frames, features, video_name)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
