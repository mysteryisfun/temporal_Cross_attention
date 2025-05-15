#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Visualization Tool

This script generates visualizations for the preprocessed data:
- Face detection results
- Optical flow visualizations
- Dataset sample montages
- Preprocessing pipeline status visualizations

The visualizations are saved to the output directory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import logging
import random
from datetime import datetime
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline_logger import DataPipelineLogger

def visualize_face_samples(faces_dir, output_path, n_videos=5, n_faces_per_video=5):
    """
    Visualize sample faces from the dataset
    
    Args:
        faces_dir (str): Directory containing detected faces
        output_path (str): Path to save output visualization
        n_videos (int): Number of videos to sample
        n_faces_per_video (int): Number of faces to sample per video
        
    Returns:
        bool: True if visualization was created successfully
    """
    # Get list of video directories
    video_dirs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    
    if not video_dirs:
        logging.warning(f"No video directories found in {faces_dir}")
        return False
    
    # Sample videos
    if len(video_dirs) > n_videos:
        video_dirs = random.sample(video_dirs, n_videos)
    
    # Create figure
    fig = plt.figure(figsize=(15, 3 * len(video_dirs)))
    gs = GridSpec(len(video_dirs), n_faces_per_video, figure=fig)
    
    # For each video, sample faces
    for i, video_id in enumerate(video_dirs):
        video_path = os.path.join(faces_dir, video_id)
        face_files = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')]
        
        if not face_files:
            continue
            
        # Sort files to get sequential order
        face_files.sort()
        
        # Sample evenly spaced faces
        indices = np.linspace(0, len(face_files) - 1, n_faces_per_video, dtype=int)
        sampled_faces = [face_files[idx] for idx in indices]
        
        # Display faces
        for j, face_file in enumerate(sampled_faces):
            ax = fig.add_subplot(gs[i, j])
            
            face_path = os.path.join(video_path, face_file)
            img = cv2.imread(face_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            ax.set_title(f"Frame {j+1}")
            ax.axis('off')
            
        # Add video ID as row label
        row_title = fig.text(0.01, 0.5 - i * (1/len(video_dirs)), video_id, 
                            ha='right', va='center', rotation='vertical', 
                            transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return True

def visualize_optical_flow_samples(flow_dir, output_path, n_videos=5, n_flows_per_video=5):
    """
    Visualize sample optical flows from the dataset
    
    Args:
        flow_dir (str): Directory containing optical flow data
        output_path (str): Path to save output visualization
        n_videos (int): Number of videos to sample
        n_flows_per_video (int): Number of flows to sample per video
        
    Returns:
        bool: True if visualization was created successfully
    """
    # Get list of video directories
    video_dirs = [d for d in os.listdir(flow_dir) if os.path.isdir(os.path.join(flow_dir, d))]
    
    if not video_dirs:
        logging.warning(f"No video directories found in {flow_dir}")
        return False
    
    # Sample videos
    if len(video_dirs) > n_videos:
        video_dirs = random.sample(video_dirs, n_videos)
    
    # Create figure
    fig = plt.figure(figsize=(15, 3 * len(video_dirs)))
    gs = GridSpec(len(video_dirs), n_flows_per_video, figure=fig)
    
    # For each video, sample flows
    for i, video_id in enumerate(video_dirs):
        video_path = os.path.join(flow_dir, video_id)
        flow_files = [f for f in os.listdir(video_path) if f.endswith('_flow_rgb.jpg')]
        
        if not flow_files:
            continue
            
        # Sort files to get sequential order
        flow_files.sort()
        
        # Sample evenly spaced flows
        indices = np.linspace(0, len(flow_files) - 1, n_flows_per_video, dtype=int)
        sampled_flows = [flow_files[idx] for idx in indices]
        
        # Display flows
        for j, flow_file in enumerate(sampled_flows):
            ax = fig.add_subplot(gs[i, j])
            
            flow_path = os.path.join(video_path, flow_file)
            img = cv2.imread(flow_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            ax.set_title(f"Flow {j+1}")
            ax.axis('off')
            
        # Add video ID as row label
        row_title = fig.text(0.01, 0.5 - i * (1/len(video_dirs)), video_id, 
                            ha='right', va='center', rotation='vertical', 
                            transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return True

def visualize_preprocessing_pipeline_status(metrics_dir, output_path):
    """
    Visualize the status of the preprocessing pipeline
    
    Args:
        metrics_dir (str): Directory containing preprocessing metrics
        output_path (str): Path to save output visualization
        
    Returns:
        bool: True if visualization was created successfully
    """
    # Look for metrics files
    frame_metrics_file = None
    face_metrics_file = None
    flow_metrics_file = None
    
    for file in os.listdir(metrics_dir):
        if file.startswith('frame_extraction_metrics_') and file.endswith('.csv'):
            frame_metrics_file = os.path.join(metrics_dir, file)
        elif file.startswith('face_detection_metrics_') and file.endswith('.csv'):
            face_metrics_file = os.path.join(metrics_dir, file)
        elif file.startswith('optical_flow_metrics_') and file.endswith('.csv'):
            flow_metrics_file = os.path.join(metrics_dir, file)
    
    # Check if we have all metrics files
    if not (frame_metrics_file and face_metrics_file and flow_metrics_file):
        # Try to load summary file instead
        summary_file = os.path.join(os.path.dirname(metrics_dir), "preprocessing_summary.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            # Create visualization from summary
            fig, axs = plt.subplots(3, 1, figsize=(12, 12))
            
            # Frame extraction metrics
            frame_metrics = summary.get('preprocessing_metrics', {}).get('frame_extraction', {})
            if frame_metrics:
                axs[0].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                           [frame_metrics.get('total_videos', 0), 
                            frame_metrics.get('processed_videos', 0), 
                            frame_metrics.get('failed_videos', 0)])
                axs[0].set_title('Frame Extraction Metrics')
                
                # Add text with additional metrics
                axs[0].text(0.5, 0.8, 
                           f"Total Frames: {frame_metrics.get('total_frames', 0)}\n"
                           f"Avg Frames/Video: {frame_metrics.get('average_frames_per_video', 0):.2f}",
                           transform=axs[0].transAxes,
                           bbox=dict(facecolor='white', alpha=0.7))
            
            # Face detection metrics
            face_metrics = summary.get('preprocessing_metrics', {}).get('face_detection', {})
            if face_metrics:
                axs[1].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                           [face_metrics.get('total_videos', 0), 
                            face_metrics.get('processed_videos', 0), 
                            face_metrics.get('failed_videos', 0)])
                axs[1].set_title('Face Detection Metrics')
                
                # Add text with additional metrics
                detection_rate = face_metrics.get('face_detection_rate', 0) * 100
                axs[1].text(0.5, 0.8, 
                           f"Total Faces: {face_metrics.get('total_faces', 0)}\n"
                           f"Detection Rate: {detection_rate:.2f}%",
                           transform=axs[1].transAxes,
                           bbox=dict(facecolor='white', alpha=0.7))
            
            # Optical flow metrics
            flow_metrics = summary.get('preprocessing_metrics', {}).get('optical_flow', {})
            if flow_metrics:
                axs[2].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                           [flow_metrics.get('total_videos', 0), 
                            flow_metrics.get('processed_videos', 0), 
                            flow_metrics.get('failed_videos', 0)])
                axs[2].set_title('Optical Flow Metrics')
                
                # Add text with additional metrics
                computation_rate = flow_metrics.get('flow_computation_rate', 0) * 100
                axs[2].text(0.5, 0.8, 
                           f"Total Flows: {flow_metrics.get('total_flows', 0)}\n"
                           f"Computation Rate: {computation_rate:.2f}%",
                           transform=axs[2].transAxes,
                           bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        else:
            logging.warning("Preprocessing metrics files not found")
            return False
    
    # Load metrics
    frame_metrics = pd.read_csv(frame_metrics_file)
    face_metrics = pd.read_csv(face_metrics_file)
    flow_metrics = pd.read_csv(flow_metrics_file)
    
    # Create visualization
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Frame extraction metrics
    if not frame_metrics.empty:
        latest_metrics = frame_metrics.iloc[-1]
        axs[0].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                   [latest_metrics.get('total_videos', 0), 
                    latest_metrics.get('processed_videos', 0), 
                    latest_metrics.get('failed_videos', 0)])
        axs[0].set_title('Frame Extraction Metrics')
        
        # Add text with additional metrics
        axs[0].text(0.5, 0.8, 
                   f"Total Frames: {latest_metrics.get('total_frames', 0)}\n"
                   f"Avg Frames/Video: {latest_metrics.get('average_frames_per_video', 0):.2f}",
                   transform=axs[0].transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Face detection metrics
    if not face_metrics.empty:
        latest_metrics = face_metrics.iloc[-1]
        axs[1].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                   [latest_metrics.get('total_videos', 0), 
                    latest_metrics.get('processed_videos', 0), 
                    latest_metrics.get('failed_videos', 0)])
        axs[1].set_title('Face Detection Metrics')
        
        # Add text with additional metrics
        detection_rate = latest_metrics.get('face_detection_rate', 0) * 100
        axs[1].text(0.5, 0.8, 
                   f"Total Faces: {latest_metrics.get('total_faces', 0)}\n"
                   f"Detection Rate: {detection_rate:.2f}%",
                   transform=axs[1].transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Optical flow metrics
    if not flow_metrics.empty:
        latest_metrics = flow_metrics.iloc[-1]
        axs[2].bar(['Total Videos', 'Processed Videos', 'Failed Videos'], 
                   [latest_metrics.get('total_videos', 0), 
                    latest_metrics.get('processed_videos', 0), 
                    latest_metrics.get('failed_videos', 0)])
        axs[2].set_title('Optical Flow Metrics')
        
        # Add text with additional metrics
        computation_rate = latest_metrics.get('flow_computation_rate', 0) * 100
        axs[2].text(0.5, 0.8, 
                   f"Total Flows: {latest_metrics.get('total_flows', 0)}\n"
                   f"Computation Rate: {computation_rate:.2f}%",
                   transform=axs[2].transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return True

def create_data_pipeline_visualization(faces_dir, flow_dir, metrics_dir, output_dir):
    """
    Create visualizations for the data pipeline
    
    Args:
        faces_dir (str): Directory containing detected faces
        flow_dir (str): Directory containing optical flow data
        metrics_dir (str): Directory containing preprocessing metrics
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to created visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_paths = []
    
    # Face samples visualization
    face_vis_path = os.path.join(output_dir, "face_samples.png")
    if visualize_face_samples(faces_dir, face_vis_path):
        vis_paths.append(face_vis_path)
    
    # Optical flow samples visualization
    flow_vis_path = os.path.join(output_dir, "optical_flow_samples.png")
    if visualize_optical_flow_samples(flow_dir, flow_vis_path):
        vis_paths.append(flow_vis_path)
    
    # Preprocessing pipeline status visualization
    pipeline_vis_path = os.path.join(output_dir, "preprocessing_pipeline_status.png")
    if visualize_preprocessing_pipeline_status(metrics_dir, pipeline_vis_path):
        vis_paths.append(pipeline_vis_path)
    
    return vis_paths

def create_side_by_side_visualization(faces_dir, flow_dir, output_dir):
    """
    Create side-by-side visualization of faces and their corresponding optical flow
    
    Args:
        faces_dir (str): Directory containing detected faces
        flow_dir (str): Directory containing optical flow data
        output_dir (str): Directory to save visualizations
        
    Returns:
        list: Paths to created visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_paths = []
    
    # Find videos with both faces and flow
    common_videos = []
    for video_id in os.listdir(faces_dir):
        video_face_dir = os.path.join(faces_dir, video_id)
        video_flow_dir = os.path.join(flow_dir, video_id)
        
        if os.path.isdir(video_face_dir) and os.path.isdir(video_flow_dir):
            # Check if there are face and flow files
            face_files = [f for f in os.listdir(video_face_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')]
            flow_files = [f for f in os.listdir(video_flow_dir) if f.endswith('_flow_rgb.jpg')]
            
            if face_files and flow_files:
                common_videos.append(video_id)
    
    if not common_videos:
        logging.warning("No videos with both faces and optical flow found")
        return vis_paths
    
    # Sample up to 5 videos
    if len(common_videos) > 5:
        common_videos = random.sample(common_videos, 5)
    
    # For each video, create a side-by-side visualization
    for video_id in common_videos:
        video_face_dir = os.path.join(faces_dir, video_id)
        video_flow_dir = os.path.join(flow_dir, video_id)
        
        face_files = sorted([f for f in os.listdir(video_face_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')])
        flow_files = sorted([f for f in os.listdir(video_flow_dir) if f.endswith('_flow_rgb.jpg')])
        
        # Match face and flow files
        # Optical flow is computed between consecutive frames, so we need to match accordingly
        pairs = []
        for i in range(min(5, len(face_files) - 1, len(flow_files))):
            face_file = face_files[i]
            # Find the corresponding flow file (may have different naming conventions)
            flow_file = None
            for f in flow_files:
                if face_file.replace('.jpg', '') in f:
                    flow_file = f
                    break
            
            if flow_file:
                pairs.append((face_file, flow_file))
        
        if not pairs:
            # Try a simpler matching approach
            pairs = [(face_files[i], flow_files[i]) for i in range(min(5, len(face_files), len(flow_files)))]
        
        if pairs:
            # Create visualization
            fig, axs = plt.subplots(len(pairs), 2, figsize=(10, 5 * len(pairs)))
            
            for i, (face_file, flow_file) in enumerate(pairs):
                # Load face image
                face_path = os.path.join(video_face_dir, face_file)
                face_img = cv2.imread(face_path)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Load flow image
                flow_path = os.path.join(video_flow_dir, flow_file)
                flow_img = cv2.imread(flow_path)
                flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
                
                # Display images
                if len(pairs) > 1:
                    axs[i, 0].imshow(face_img)
                    axs[i, 0].set_title(f"Face {i+1}")
                    axs[i, 0].axis('off')
                    
                    axs[i, 1].imshow(flow_img)
                    axs[i, 1].set_title(f"Optical Flow {i+1}")
                    axs[i, 1].axis('off')
                else:
                    axs[0].imshow(face_img)
                    axs[0].set_title(f"Face {i+1}")
                    axs[0].axis('off')
                    
                    axs[1].imshow(flow_img)
                    axs[1].set_title(f"Optical Flow {i+1}")
                    axs[1].axis('off')
            
            plt.suptitle(f"Video ID: {video_id}")
            plt.tight_layout()
            
            # Save visualization
            output_path = os.path.join(output_dir, f"side_by_side_{video_id}.png")
            plt.savefig(output_path)
            plt.close()
            
            vis_paths.append(output_path)
    
    # Create a montage of all side-by-side visualizations
    if vis_paths:
        montage_path = os.path.join(output_dir, "side_by_side_montage.png")
        
        # Load all visualizations
        images = []
        for path in vis_paths:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to a consistent height
                h, w = img.shape[:2]
                new_h = 800
                new_w = int(w * new_h / h)
                img = cv2.resize(img, (new_w, new_h))
                images.append(img)
        
        if images:
            # Create a vertical montage
            montage = np.vstack(images)
            
            # Save montage
            plt.figure(figsize=(15, 5 * len(images)))
            plt.imshow(montage)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(montage_path)
            plt.close()
            
            vis_paths.append(montage_path)
    
    return vis_paths

def main():
    parser = argparse.ArgumentParser(description="Create data pipeline visualizations")
    
    parser.add_argument('--faces-dir', type=str, required=True,
                        help="Directory containing detected faces")
    parser.add_argument('--flow-dir', type=str, required=True,
                        help="Directory containing optical flow data")
    parser.add_argument('--metrics-dir', type=str, required=True,
                        help="Directory containing preprocessing metrics")
    parser.add_argument('--output-dir', type=str, default="results/visualizations/data_pipeline",
                        help="Directory to save visualizations")
    parser.add_argument('--log-config', type=str,
                        help="Path to logging configuration YAML")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = DataPipelineLogger(
        config_path=args.log_config,
        experiment_name=f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir=os.path.dirname(os.path.dirname(args.output_dir))  # Set to results dir
    )
    
    logger.logger.info("Starting data pipeline visualization...")
    
    # Create data pipeline visualizations
    vis_paths = create_data_pipeline_visualization(
        args.faces_dir,
        args.flow_dir,
        args.metrics_dir,
        args.output_dir
    )
    
    # Create side-by-side visualizations
    side_by_side_paths = create_side_by_side_visualization(
        args.faces_dir,
        args.flow_dir,
        args.output_dir
    )
    
    # Report created visualizations
    all_vis_paths = vis_paths + side_by_side_paths
    
    if all_vis_paths:
        logger.logger.info(f"Created {len(all_vis_paths)} visualizations:")
        for path in all_vis_paths:
            logger.logger.info(f"  - {path}")
    else:
        logger.logger.warning("No visualizations created")
    
    logger.logger.info("Data pipeline visualization completed.")

if __name__ == "__main__":
    main()
