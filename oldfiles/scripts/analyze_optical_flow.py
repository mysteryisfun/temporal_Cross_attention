#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: analyze_optical_flow.py
# Description: Script to analyze optical flow results and generate visualizations

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path

def analyze_flow_magnitude_by_video(flow_dir, output_dir):
    """
    Analyze optical flow magnitude distribution by video.
    
    Args:
        flow_dir (str): Directory containing optical flow results.
        output_dir (str): Directory to save analysis results.
        
    Returns:
        dict: Flow magnitude statistics by video.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video directories
    video_dirs = []
    for item in os.listdir(flow_dir):
        item_path = os.path.join(flow_dir, item)
        if os.path.isdir(item_path) and not item_path.endswith('visualizations'):
            video_dirs.append(item_path)
    
    print(f"Found {len(video_dirs)} video directories in {flow_dir}")
    
    # Analyze each video
    video_stats = {}
    
    for video_dir in tqdm(video_dirs, desc="Analyzing videos"):
        video_id = os.path.basename(video_dir)
        
        # Get all raw flow files for this video
        flow_files = []
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.endswith('_flow_raw.npy'):
                    flow_files.append(os.path.join(root, file))
        
        if not flow_files:
            continue
        
        # Calculate flow magnitudes
        magnitudes = []
        for flow_file in flow_files:
            try:
                flow = np.load(flow_file)
                u, v = flow[..., 0], flow[..., 1]
                magnitude = np.sqrt(u**2 + v**2)
                
                # Get statistics for this flow
                magnitudes.append({
                    'file': os.path.basename(flow_file),
                    'mean': float(np.mean(magnitude)),
                    'std': float(np.std(magnitude)),
                    'max': float(np.max(magnitude)),
                    'min': float(np.min(magnitude)),
                    'median': float(np.median(magnitude))
                })
            except Exception as e:
                print(f"Error processing {flow_file}: {str(e)}")
        
        if not magnitudes:
            continue
        
        # Calculate video-level statistics
        mean_values = [m['mean'] for m in magnitudes]
        max_values = [m['max'] for m in magnitudes]
        
        video_stats[video_id] = {
            'num_flows': len(magnitudes),
            'mean_magnitude': float(np.mean(mean_values)),
            'std_magnitude': float(np.std(mean_values)),
            'max_magnitude': float(np.max(max_values)),
            'flow_stats': magnitudes
        }
        
        # Generate visualization for this video
        plt.figure(figsize=(10, 6))
        
        # Plot flow magnitudes over time
        plt.subplot(2, 1, 1)
        plt.plot(mean_values, 'b-', label='Mean Magnitude')
        plt.plot(max_values, 'r-', label='Max Magnitude')
        plt.title(f"Flow Magnitude for {video_id}")
        plt.xlabel('Frame Index')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        # Plot histogram of mean magnitudes
        plt.subplot(2, 1, 2)
        plt.hist(mean_values, bins=max(5, len(mean_values)//2), alpha=0.7)
        plt.axvline(np.mean(mean_values), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(mean_values):.2f}')
        plt.title(f"Distribution of Mean Flow Magnitude")
        plt.xlabel('Mean Flow Magnitude')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_flow_analysis.png"), dpi=300)
        plt.close()
    
    # Save overall statistics
    with open(os.path.join(output_dir, "video_flow_stats.json"), 'w') as f:
        json.dump(video_stats, f, indent=2)
    
    # Generate overall visualization
    if video_stats:
        # Prepare data for visualization
        video_ids = list(video_stats.keys())
        mean_magnitudes = [video_stats[vid]['mean_magnitude'] for vid in video_ids]
        max_magnitudes = [video_stats[vid]['max_magnitude'] for vid in video_ids]
        num_flows = [video_stats[vid]['num_flows'] for vid in video_ids]
        
        # Sort by mean magnitude for better visualization
        sorted_indices = np.argsort(mean_magnitudes)
        video_ids = [video_ids[i] for i in sorted_indices]
        mean_magnitudes = [mean_magnitudes[i] for i in sorted_indices]
        max_magnitudes = [max_magnitudes[i] for i in sorted_indices]
        num_flows = [num_flows[i] for i in sorted_indices]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean flow magnitude by video
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(video_ids)), mean_magnitudes, alpha=0.7)
        # Add max magnitude as error bars
        plt.errorbar(range(len(video_ids)), mean_magnitudes, 
                    yerr=[[0]*len(video_ids), [max_m - mean_m for max_m, mean_m in zip(max_magnitudes, mean_magnitudes)]], 
                    fmt='none', ecolor='red', capsize=3)
        
        plt.title('Mean Flow Magnitude by Video')
        plt.xlabel('Video ID')
        plt.ylabel('Mean Flow Magnitude')
        plt.xticks(range(len(video_ids)), [vid[:8] + '...' for vid in video_ids], rotation=90)
        plt.grid(True, axis='y')
        
        # Plot 2: Histogram of mean magnitudes
        plt.subplot(2, 2, 3)
        plt.hist(mean_magnitudes, bins=20, alpha=0.7, color='#2196F3')
        plt.axvline(np.mean(mean_magnitudes), color='red', linestyle='dashed', 
                   label=f'Overall Mean: {np.mean(mean_magnitudes):.2f}')
        plt.title('Distribution of Mean Flow Magnitudes')
        plt.xlabel('Mean Flow Magnitude')
        plt.ylabel('Number of Videos')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Flow count by video
        plt.subplot(2, 2, 4)
        plt.scatter(mean_magnitudes, num_flows, alpha=0.7)
        plt.title('Number of Flows vs. Mean Magnitude')
        plt.xlabel('Mean Flow Magnitude')
        plt.ylabel('Number of Flows')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_flow_analysis.png"), dpi=300)
        plt.close()
    
    return video_stats

def create_flow_visualizations(flow_dir, output_dir, num_samples=10):
    """
    Create visualizations of optical flow for sample videos.
    
    Args:
        flow_dir (str): Directory containing optical flow results.
        output_dir (str): Directory to save visualizations.
        num_samples (int): Number of videos to visualize.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find videos with RGB flow visualizations
    video_flows = {}
    for root, dirs, files in os.walk(flow_dir):
        for file in files:
            if file.endswith('_flow_rgb.jpg'):
                flow_path = os.path.join(root, file)
                video_id = flow_path.split(os.sep)[-2]
                
                if video_id not in video_flows:
                    video_flows[video_id] = []
                    
                video_flows[video_id].append(flow_path)
    
    # Select top videos with most flows
    top_videos = sorted(video_flows.items(), key=lambda x: len(x[1]), reverse=True)[:num_samples]
    
    for video_id, flow_paths in tqdm(top_videos, desc="Creating flow visualizations"):
        # Sort flow paths to ensure temporal order
        flow_paths.sort()
        
        # Create a grid of flow visualizations
        rows = min(2, len(flow_paths))
        cols = min(5, len(flow_paths))
        total_cells = rows * cols
        
        plt.figure(figsize=(4 * cols, 4 * rows))
        
        for i, flow_path in enumerate(flow_paths[:total_cells]):
            ax = plt.subplot(rows, cols, i + 1)
            img = cv2.imread(flow_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            
            frame_name = os.path.basename(flow_path).split('_flow_rgb')[0]
            ax.set_title(f"{frame_name}")
            ax.axis('off')
        
        plt.suptitle(f"Optical Flow for {video_id}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{video_id}_flow_grid.png"), dpi=300)
        plt.close()
    
    print(f"Created flow visualizations for {len(top_videos)} videos in {output_dir}")

def compare_flows_with_faces(flow_dir, faces_dir, output_dir, num_samples=5):
    """
    Create visualizations comparing face images with their optical flow.
    
    Args:
        flow_dir (str): Directory containing optical flow results.
        faces_dir (str): Directory containing face images.
        output_dir (str): Directory to save visualizations.
        num_samples (int): Number of videos to visualize.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find videos with both face images and flow visualizations
    common_videos = []
    
    # Get all video IDs in the flow directory
    flow_videos = [d for d in os.listdir(flow_dir) if os.path.isdir(os.path.join(flow_dir, d))]
    
    # Check which ones also exist in the faces directory
    for video_id in flow_videos:
        face_dir = os.path.join(faces_dir, video_id)
        if os.path.isdir(face_dir):
            common_videos.append(video_id)
    
    print(f"Found {len(common_videos)} videos with both faces and optical flow")
    
    # Select random samples
    if len(common_videos) > num_samples:
        import random
        random.shuffle(common_videos)
        selected_videos = common_videos[:num_samples]
    else:
        selected_videos = common_videos
    
    for video_id in tqdm(selected_videos, desc="Creating comparison visualizations"):
        # Get face images
        face_dir = os.path.join(faces_dir, video_id)
        face_files = [f for f in os.listdir(face_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        face_files.sort()
        
        # Get flow images
        flow_dir_vid = os.path.join(flow_dir, video_id)
        if not os.path.exists(flow_dir_vid):
            flow_dir_vid = os.path.join(flow_dir, video_id, video_id)  # Handle double-nested structure
            
        flow_files = []
        if os.path.exists(flow_dir_vid):
            flow_files = [f for f in os.listdir(flow_dir_vid) if f.endswith('_flow_rgb.jpg')]
            flow_files.sort()
        
        if not face_files or not flow_files:
            continue
        
        # Create visualization grid
        # Each row will have: Face 1, Face 2, Flow 1->2
        num_flows = len(flow_files)
        num_faces = len(face_files)
        
        # We need at least 2 faces to show a flow
        if num_faces < 2:
            continue
        
        # Determine how many rows we can create
        max_rows = 3  # Limit to 3 rows for readability
        num_rows = min(max_rows, num_flows)
        
        plt.figure(figsize=(12, 4 * num_rows))
        
        for i in range(num_rows):
            # Load face images
            face1_path = os.path.join(face_dir, face_files[i])
            face2_path = os.path.join(face_dir, face_files[i+1])
            
            face1 = cv2.imread(face1_path)
            face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
            
            face2 = cv2.imread(face2_path)
            face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
            
            # Load flow image
            base_name = os.path.splitext(face_files[i])[0]
            flow_path = None
            
            # Find matching flow file
            for flow_file in flow_files:
                if flow_file.startswith(base_name):
                    flow_path = os.path.join(flow_dir_vid, flow_file)
                    break
            
            if not flow_path:
                continue
                
            flow = cv2.imread(flow_path)
            flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
            
            # Plot the images
            # Face 1
            ax1 = plt.subplot(num_rows, 3, i*3 + 1)
            ax1.imshow(face1)
            ax1.set_title(f"Face {i+1}")
            ax1.axis('off')
            
            # Face 2
            ax2 = plt.subplot(num_rows, 3, i*3 + 2)
            ax2.imshow(face2)
            ax2.set_title(f"Face {i+2}")
            ax2.axis('off')
            
            # Flow
            ax3 = plt.subplot(num_rows, 3, i*3 + 3)
            ax3.imshow(flow)
            ax3.set_title(f"Flow {i+1}->{i+2}")
            ax3.axis('off')
        
        plt.suptitle(f"Faces and Optical Flow for {video_id}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{video_id}_faces_flow_comparison.png"), dpi=300)
        plt.close()
    
    print(f"Created comparison visualizations for {len(selected_videos)} videos in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze optical flow results")
    
    # Input/output options
    parser.add_argument('--flow-dir', type=str, default='data/optical_flow',
                        help="Directory containing optical flow results")
    parser.add_argument('--faces-dir', type=str, default='data/faces',
                        help="Directory containing face images")
    parser.add_argument('--output-dir', type=str, default='results/visualizations/optical_flow',
                        help="Directory to save analysis results")
    
    # Analysis options
    parser.add_argument('--analyze-magnitude', action='store_true', default=True,
                        help="Analyze flow magnitude by video")
    parser.add_argument('--create-visualizations', action='store_true', default=True,
                        help="Create flow visualizations")
    parser.add_argument('--compare-with-faces', action='store_true', default=True,
                        help="Compare flows with face images")
    parser.add_argument('--num-samples', type=int, default=10,
                        help="Number of sample videos to visualize")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing optical flow results in {args.flow_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Analyze flow magnitude by video
    if args.analyze_magnitude:
        print("Analyzing flow magnitude by video...")
        magnitude_dir = os.path.join(args.output_dir, "magnitude_analysis")
        analyze_flow_magnitude_by_video(args.flow_dir, magnitude_dir)
    
    # Create flow visualizations
    if args.create_visualizations:
        print("Creating flow visualizations...")
        visualizations_dir = os.path.join(args.output_dir, "flow_visualizations")
        create_flow_visualizations(args.flow_dir, visualizations_dir, args.num_samples)
    
    # Compare flows with faces
    if args.compare_with_faces:
        print("Creating face-flow comparison visualizations...")
        comparison_dir = os.path.join(args.output_dir, "face_flow_comparison")
        compare_flows_with_faces(args.flow_dir, args.faces_dir, comparison_dir, args.num_samples)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
