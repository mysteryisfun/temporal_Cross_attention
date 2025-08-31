#!/usr/bin/env python3
"""
Motion Feature Visualization Script

This script provides comprehensive visualization tools for motion features 
extracted by the I3D motion extractor. It can visualize individual feature 
files or compare multiple motion feature files.

Usage:
    python scripts/visualize_motion_features.py --file path/to/features.npy
    python scripts/visualize_motion_features.py --directory experiments/i3d_test/features/
    python scripts/visualize_motion_features.py --compare file1.npy file2.npy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import List, Optional, Tuple, Dict

class MotionFeatureVisualizer:
    """Visualizer for I3D motion features"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_features(self, file_path: str) -> np.ndarray:
        """
        Load motion features from .npy file
        
        Args:
            file_path: Path to .npy file
            
        Returns:
            Loaded features array
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        features = np.load(file_path)
        print(f"Loaded features from: {file_path}")
        print(f"Shape: {features.shape}")
        print(f"Range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        return features
    
    def visualize_single_feature(self, features: np.ndarray, title: str = "Motion Features") -> None:
        """
        Create comprehensive visualization for a single feature vector
        
        Args:
            features: Feature array of shape (1, 512) or (512,)
            title: Plot title
        """
        # Flatten if needed
        if features.ndim == 2:
            features = features.flatten()
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(f'{title} - I3D Motion Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature distribution histogram
        axes[0, 0].hist(features, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(features.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {features.mean():.3f}')
        axes[0, 0].axvline(features.mean() + features.std(), color='orange', linestyle='--', linewidth=2, label=f'+1σ: {features.mean() + features.std():.3f}')
        axes[0, 0].axvline(features.mean() - features.std(), color='orange', linestyle='--', linewidth=2, label=f'-1σ: {features.mean() - features.std():.3f}')
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature sequence plot
        axes[0, 1].plot(features, linewidth=1, color='darkblue', alpha=0.8)
        axes[0, 1].set_title('Feature Sequence (Index vs Value)')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature heatmap (reshaped as 16x32 for visualization)
        feature_2d = features.reshape(16, 32)
        im1 = axes[0, 2].imshow(feature_2d, cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Feature Heatmap (16x32 reshape)')
        axes[0, 2].set_xlabel('Feature Dimension')
        axes[0, 2].set_ylabel('Feature Group')
        plt.colorbar(im1, ax=axes[0, 2], shrink=0.8)
        
        # 4. Top activated features
        top_indices = np.argsort(features)[-20:]
        top_values = features[top_indices]
        axes[1, 0].barh(range(len(top_indices)), top_values, color='lightcoral')
        axes[1, 0].set_yticks(range(len(top_indices)))
        axes[1, 0].set_yticklabels(top_indices)
        axes[1, 0].set_title('Top 20 Activated Features')
        axes[1, 0].set_xlabel('Feature Value')
        axes[1, 0].set_ylabel('Feature Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature statistics box plot
        quartiles = [np.percentile(features, q) for q in [25, 50, 75]]
        stats_data = [features]
        axes[1, 1].boxplot(stats_data, labels=['Motion Features'], patch_artist=True, 
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1, 1].set_title('Feature Statistics Box Plot')
        axes[1, 1].set_ylabel('Feature Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add text annotations
        stats_text = f"""Statistics:
Min: {features.min():.4f}
Max: {features.max():.4f}
Mean: {features.mean():.4f}
Std: {features.std():.4f}
Q25: {quartiles[0]:.4f}
Q50: {quartiles[1]:.4f}
Q75: {quartiles[2]:.4f}
Active (>0): {np.sum(features > 0)}/512
Sparse (≈0): {np.sum(np.abs(features) < 0.01)}/512"""
        
        axes[1, 1].text(1.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        # 6. Cumulative distribution
        sorted_features = np.sort(features)
        cumulative = np.arange(1, len(sorted_features) + 1) / len(sorted_features)
        axes[1, 2].plot(sorted_features, cumulative, linewidth=2, color='purple')
        axes[1, 2].set_title('Cumulative Distribution Function')
        axes[1, 2].set_xlabel('Feature Value')
        axes[1, 2].set_ylabel('Cumulative Probability')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_features(self, feature_files: List[str]) -> None:
        """
        Compare multiple motion feature files
        
        Args:
            feature_files: List of paths to .npy files
        """
        if len(feature_files) < 2:
            raise ValueError("Need at least 2 feature files for comparison")
        
        features_list = []
        labels = []
        
        for file_path in feature_files:
            features = self.load_features(file_path)
            if features.ndim == 2:
                features = features.flatten()
            features_list.append(features)
            labels.append(Path(file_path).stem)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Motion Features Comparison', fontsize=16, fontweight='bold')
        
        # 1. Distribution comparison
        for i, (features, label) in enumerate(zip(features_list, labels)):
            axes[0, 0].hist(features, bins=30, alpha=0.6, label=label)
        axes[0, 0].set_title('Feature Distributions')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        axes[0, 1].boxplot(features_list, labels=labels, patch_artist=True)
        axes[0, 1].set_title('Feature Statistics Comparison')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature correlation heatmap (if 2 features)
        if len(features_list) == 2:
            correlation = np.corrcoef(features_list[0], features_list[1])[0, 1]
            axes[1, 0].scatter(features_list[0], features_list[1], alpha=0.5, s=1)
            axes[1, 0].set_title(f'Feature Correlation (r={correlation:.3f})')
            axes[1, 0].set_xlabel(labels[0])
            axes[1, 0].set_ylabel(labels[1])
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Correlation matrix for multiple features
            corr_matrix = np.corrcoef(features_list)
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Feature Correlation Matrix')
            axes[1, 0].set_xticks(range(len(labels)))
            axes[1, 0].set_yticks(range(len(labels)))
            axes[1, 0].set_xticklabels(labels, rotation=45)
            axes[1, 0].set_yticklabels(labels)
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Statistics table
        stats_data = []
        for features, label in zip(features_list, labels):
            stats_data.append([
                label,
                f"{features.mean():.4f}",
                f"{features.std():.4f}",
                f"{features.min():.4f}",
                f"{features.max():.4f}",
                f"{np.sum(features > 0)}/512"
            ])
        
        table = axes[1, 1].table(cellText=stats_data,
                                colLabels=['File', 'Mean', 'Std', 'Min', 'Max', 'Active'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Feature Statistics Summary')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_directory(self, directory_path: str) -> None:
        """
        Visualize all .npy files in a directory
        
        Args:
            directory_path: Path to directory containing .npy files
        """
        npy_files = glob.glob(os.path.join(directory_path, "*.npy"))
        
        if not npy_files:
            print(f"No .npy files found in directory: {directory_path}")
            return
        
        print(f"Found {len(npy_files)} .npy files in {directory_path}")
        
        if len(npy_files) == 1:
            # Single file visualization
            features = self.load_features(npy_files[0])
            self.visualize_single_feature(features, title=Path(npy_files[0]).stem)
        else:
            # Multiple file comparison
            self.compare_features(npy_files[:5])  # Limit to first 5 files


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize I3D motion features')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to single .npy feature file')
    group.add_argument('--directory', type=str, help='Path to directory containing .npy files')
    group.add_argument('--compare', nargs='+', help='Paths to multiple .npy files to compare')
    
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 10], 
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MotionFeatureVisualizer(figsize=tuple(args.figsize))
    
    try:
        if args.file:
            # Single file visualization
            features = visualizer.load_features(args.file)
            title = Path(args.file).stem
            visualizer.visualize_single_feature(features, title=title)
            
        elif args.directory:
            # Directory visualization
            visualizer.visualize_directory(args.directory)
            
        elif args.compare:
            # Multiple file comparison
            visualizer.compare_features(args.compare)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
