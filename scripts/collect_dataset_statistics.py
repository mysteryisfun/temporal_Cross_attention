#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Statistics Collection Script

This script analyzes the dataset and collects comprehensive statistics including:
- Class distribution (personality trait scores)
- Video characteristics (duration, frame count)
- Face detection metrics
- Optical flow metrics

The statistics are logged both to file and visualized.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import logging
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline_logger import DataPipelineLogger

def load_annotations(annotation_file):
    """Load annotations from pickle file"""
    try:
        # Try different encoding options
        encodings = ['ascii', 'latin1', 'utf-8', None]
        
        for encoding in encodings:
            try:
                with open(annotation_file, 'rb') as f:
                    if encoding:
                        return pickle.load(f, encoding=encoding)
                    else:
                        return pickle.load(f)
            except Exception as specific_error:
                last_error = specific_error
                continue
                
        # If we get here, all encoding attempts failed
        logging.error(f"Error loading annotations with all encoding attempts: {last_error}")
        return None
    except Exception as e:
        logging.error(f"Error accessing annotation file: {e}")
        return None

def count_faces_dir(faces_dir):
    """Count the number of face images in a directory"""
    if not os.path.exists(faces_dir):
        return 0
    
    face_files = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return len(face_files)

def count_optical_flow_dir(flow_dir):
    """Count the number of optical flow images in a directory"""
    if not os.path.exists(flow_dir):
        return 0
    
    flow_files = [f for f in os.listdir(flow_dir) if f.endswith('_flow_rgb.jpg')]
    return len(flow_files)

def analyze_trait_distributions(annotations):
    """Analyze distributions of personality traits"""
    trait_stats = {}
    
    # Process each trait
    for trait in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]:
        if trait not in annotations:
            continue
            
        trait_values = list(annotations[trait].values())
        
        trait_stats[trait] = {
            "mean": float(np.mean(trait_values)),
            "std": float(np.std(trait_values)),
            "min": float(np.min(trait_values)),
            "max": float(np.max(trait_values)),
            "median": float(np.median(trait_values)),
            "count": len(trait_values)
        }
    
    return trait_stats

def analyze_face_detection(faces_root, video_ids):
    """Analyze face detection results"""
    face_detection_stats = {}
    
    # Process each video directory
    for video_id in tqdm(video_ids, desc="Analyzing face detection"):
        video_dir = os.path.join(faces_root, video_id)
        
        if not os.path.exists(video_dir):
            continue
            
        # Count total face images
        face_count = count_faces_dir(video_dir)
        
        face_detection_stats[video_id] = {
            "face_count": face_count
        }
    
    # Calculate aggregate statistics
    if face_detection_stats:
        face_counts = [stats["face_count"] for stats in face_detection_stats.values()]
        
        aggregate_stats = {
            "total_faces": sum(face_counts),
            "mean_faces_per_video": np.mean(face_counts),
            "std_faces_per_video": np.std(face_counts),
            "min_faces_per_video": np.min(face_counts),
            "max_faces_per_video": np.max(face_counts),
            "videos_with_faces": len([c for c in face_counts if c > 0]),
            "videos_without_faces": len([c for c in face_counts if c == 0])
        }
        
        return {
            "per_video": face_detection_stats,
            "aggregate": aggregate_stats
        }
    
    return {"per_video": {}, "aggregate": {}}

def analyze_optical_flow(flow_root, video_ids):
    """Analyze optical flow quality"""
    flow_stats = {}
    
    # Process each video directory
    for video_id in tqdm(video_ids, desc="Analyzing optical flow"):
        video_dir = os.path.join(flow_root, video_id)
        
        if not os.path.exists(video_dir):
            continue
            
        # Count optical flow images
        flow_count = count_optical_flow_dir(video_dir)
        
        # If there are flow images, analyze a sample for quality metrics
        if flow_count > 0:
            flow_magnitudes = []
            
            # Get all flow_rgb files
            flow_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                         if f.endswith('_flow_rgb.jpg')]
            
            # Sample up to 10 files
            sample_files = flow_files[:min(10, len(flow_files))]
            
            for flow_file in sample_files:
                # Check if there's a corresponding raw flow file
                raw_flow_file = flow_file.replace('_flow_rgb.jpg', '_flow_raw.npy')
                
                if os.path.exists(raw_flow_file):
                    try:
                        # Load raw flow data
                        flow = np.load(raw_flow_file)
                        
                        # Calculate magnitude
                        u, v = flow[..., 0], flow[..., 1]
                        magnitude = np.sqrt(u**2 + v**2)
                        
                        flow_magnitudes.append({
                            "mean": float(np.mean(magnitude)),
                            "max": float(np.max(magnitude)),
                            "std": float(np.std(magnitude))
                        })
                    except Exception as e:
                        logging.warning(f"Error analyzing flow file {raw_flow_file}: {e}")
            
            # Aggregate flow statistics
            if flow_magnitudes:
                flow_stats[video_id] = {
                    "flow_count": flow_count,
                    "average_magnitude": np.mean([fm["mean"] for fm in flow_magnitudes]),
                    "max_magnitude": np.max([fm["max"] for fm in flow_magnitudes]),
                    "std_magnitude": np.mean([fm["std"] for fm in flow_magnitudes])
                }
            else:
                flow_stats[video_id] = {
                    "flow_count": flow_count,
                    "status": "no_raw_data"
                }
    
    # Calculate aggregate statistics
    if flow_stats:
        videos_with_magnitude = [vid for vid in flow_stats if "average_magnitude" in flow_stats[vid]]
        
        if videos_with_magnitude:
            flow_counts = [stats["flow_count"] for stats in flow_stats.values()]
            avg_magnitudes = [stats["average_magnitude"] for stats in flow_stats.values() 
                            if "average_magnitude" in stats]
            
            aggregate_stats = {
                "total_flows": sum(flow_counts),
                "mean_flows_per_video": np.mean(flow_counts),
                "std_flows_per_video": np.std(flow_counts),
                "mean_magnitude": np.mean(avg_magnitudes),
                "std_magnitude": np.std(avg_magnitudes),
                "videos_with_flow": len([c for c in flow_counts if c > 0]),
                "videos_without_flow": len([c for c in flow_counts if c == 0])
            }
            
            return {
                "per_video": flow_stats,
                "aggregate": aggregate_stats
            }
    
    return {"per_video": {}, "aggregate": {}}

def visualize_trait_distributions(trait_stats, output_dir):
    """Visualize personality trait distributions"""
    traits = list(trait_stats.keys())
    trait_means = [trait_stats[t]["mean"] for t in traits]
    trait_stds = [trait_stats[t]["std"] for t in traits]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot means with error bars
    plt.bar(traits, trait_means, yerr=trait_stds, capsize=10)
    plt.ylim(0, 1)
    plt.ylabel("Score (0-1)")
    plt.title("Personality Trait Distributions")
    
    # Add value labels
    for i, v in enumerate(trait_means):
        plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "trait_distributions.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def collect_dataset_statistics(annotation_file, faces_root, flow_root, output_dir):
    """
    Collect comprehensive dataset statistics
    
    Args:
        annotation_file (str): Path to annotation file
        faces_root (str): Directory containing face images
        flow_root (str): Directory containing optical flow data
        output_dir (str): Directory to save output visualizations
        
    Returns:
        dict: Comprehensive statistics
    """
    # Load annotations
    annotations = load_annotations(annotation_file)
    if not annotations:
        return None
    
    # Extract video IDs from annotations
    video_ids = set()
    for trait in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]:
        if trait in annotations:
            # Add video IDs without extension
            for video_file in annotations[trait].keys():
                video_id = os.path.splitext(video_file)[0]
                video_ids.add(video_id)
    
    # Analyze trait distributions
    trait_stats = analyze_trait_distributions(annotations)
    
    # Analyze face detection results
    face_stats = analyze_face_detection(faces_root, video_ids)
    
    # Analyze optical flow quality
    flow_stats = analyze_optical_flow(flow_root, video_ids)
    
    # Combine all statistics
    dataset_stats = {
        "trait_statistics": trait_stats,
        "face_detection_statistics": face_stats["aggregate"],
        "optical_flow_statistics": flow_stats["aggregate"],
        "video_count": len(video_ids)
    }
    
    return {
        "dataset_stats": dataset_stats,
        "face_per_video_stats": face_stats["per_video"],
        "flow_per_video_stats": flow_stats["per_video"]
    }

def main():
    parser = argparse.ArgumentParser(description="Collect and log dataset statistics")
    
    parser.add_argument('--annotation-file', type=str, required=True,
                        help="Path to annotation pickle file")
    parser.add_argument('--faces-dir', type=str, required=True,
                        help="Directory containing face images")
    parser.add_argument('--flow-dir', type=str, required=True,
                        help="Directory containing optical flow data")
    parser.add_argument('--output-dir', type=str, default="results",
                        help="Directory to save statistics and visualizations")
    parser.add_argument('--log-config', type=str,
                        help="Path to logging configuration YAML")
    
    args = parser.parse_args()
    
    # Initialize the data pipeline logger
    logger = DataPipelineLogger(
        config_path=args.log_config,
        experiment_name=f"dataset_stats_{os.path.basename(args.annotation_file).split('.')[0]}",
        output_dir=args.output_dir
    )
    
    logger.logger.info("Starting dataset statistics collection...")
    
    # Collect statistics
    stats = collect_dataset_statistics(
        args.annotation_file,
        args.faces_dir,
        args.flow_dir,
        os.path.join(args.output_dir, "visualizations")
    )
    
    if not stats:
        logger.logger.error("Failed to collect dataset statistics.")
        return
    
    # Log dataset statistics
    dataset_name = os.path.basename(args.annotation_file).split('.')[0]
    logger.log_dataset_statistics(dataset_name, stats["dataset_stats"])
    
    # Visualize trait distributions
    trait_vis_path = visualize_trait_distributions(
        stats["dataset_stats"]["trait_statistics"],
        os.path.join(args.output_dir, "visualizations")
    )
    logger.logger.info(f"Trait distribution visualization saved to {trait_vis_path}")
    
    # Visualize face detection results
    if stats["face_per_video_stats"]:
        logger.visualize_face_detection_results(stats["face_per_video_stats"])
    
    # Visualize optical flow quality
    if stats["flow_per_video_stats"]:
        logger.visualize_optical_flow_quality(stats["flow_per_video_stats"])
    
    # Sample visualization
    # Try to find paired samples
    paired_samples = []
    for video_id in stats["face_per_video_stats"]:
        face_dir = os.path.join(args.faces_dir, video_id)
        flow_dir = os.path.join(args.flow_dir, video_id)
        
        if os.path.exists(face_dir) and os.path.exists(flow_dir):
            face_files = [os.path.join(face_dir, f) for f in os.listdir(face_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            flow_files = [os.path.join(flow_dir, f) for f in os.listdir(flow_dir) 
                         if f.endswith('_flow_rgb.jpg')]
            
            if face_files and flow_files:
                # For simplicity, just pair the first few
                for i in range(min(5, len(face_files)-1, len(flow_files))):
                    paired_samples.append((face_files[i], flow_files[i]))
                    
                # If we have enough samples, stop
                if len(paired_samples) >= 5:
                    break
    
    # Visualize sample pairs
    if paired_samples:
        faces = [pair[0] for pair in paired_samples]
        flows = [pair[1] for pair in paired_samples]
        logger.visualize_dataset_samples(faces, flows)
    
    logger.logger.info("Dataset statistics collection completed.")

if __name__ == "__main__":
    main()
