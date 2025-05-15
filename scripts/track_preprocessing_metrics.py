#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing Metrics Tracking Script

This script analyzes the preprocessing steps and logs metrics including:
- Frame extraction metrics (success rate, time)
- Face detection metrics (detection rate, confidence)
- Optical flow metrics (quality, computation time)

The metrics are logged both to file and visualization systems.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline_logger import DataPipelineLogger

def collect_frame_extraction_metrics(frames_dir, raw_dir):
    """
    Collect metrics about frame extraction process
    
    Args:
        frames_dir (str): Directory containing extracted frames
        raw_dir (str): Directory containing raw videos
        
    Returns:
        dict: Frame extraction metrics
    """
    metrics = {
        "total_videos": 0,
        "processed_videos": 0,
        "failed_videos": 0,
        "total_frames": 0,
        "average_frames_per_video": 0,
        "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Count videos in raw directory
    video_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    metrics["total_videos"] = len(video_files)
    
    # Count processed videos and frames
    processed_videos = set()
    frame_counts = []
    
    for root, dirs, files in os.walk(frames_dir):
        # Check if this directory contains frames
        frame_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if frame_files:
            # This is a video directory with frames
            video_id = os.path.basename(root)
            processed_videos.add(video_id)
            frame_counts.append(len(frame_files))
    
    metrics["processed_videos"] = len(processed_videos)
    metrics["failed_videos"] = metrics["total_videos"] - metrics["processed_videos"]
    metrics["total_frames"] = sum(frame_counts)
    
    if frame_counts:
        metrics["average_frames_per_video"] = np.mean(frame_counts)
        metrics["min_frames"] = int(np.min(frame_counts))
        metrics["max_frames"] = int(np.max(frame_counts))
    
    return metrics

def collect_face_detection_metrics(faces_dir, frames_dir):
    """
    Collect metrics about face detection process
    
    Args:
        faces_dir (str): Directory containing detected faces
        frames_dir (str): Directory containing extracted frames
        
    Returns:
        dict: Face detection metrics
    """
    metrics = {
        "total_videos": 0,
        "processed_videos": 0,
        "failed_videos": 0,
        "total_faces": 0,
        "total_frames": 0,
        "face_detection_rate": 0,
        "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Count videos with frames
    video_dirs = set()
    frame_counts = {}
    
    for root, dirs, files in os.walk(frames_dir):
        frame_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if frame_files:
            video_id = os.path.basename(root)
            video_dirs.add(video_id)
            frame_counts[video_id] = len(frame_files)
    
    metrics["total_videos"] = len(video_dirs)
    
    # Count videos with detected faces
    processed_videos = set()
    face_counts = {}
    
    for root, dirs, files in os.walk(faces_dir):
        face_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')]
        
        if face_files:
            video_id = os.path.basename(root)
            processed_videos.add(video_id)
            face_counts[video_id] = len(face_files)
    
    metrics["processed_videos"] = len(processed_videos)
    metrics["failed_videos"] = len(video_dirs) - len(processed_videos)
    metrics["total_faces"] = sum(face_counts.values()) if face_counts else 0
    metrics["total_frames"] = sum(frame_counts.values()) if frame_counts else 0
    
    # Calculate detection rate
    if metrics["total_frames"] > 0:
        metrics["face_detection_rate"] = metrics["total_faces"] / metrics["total_frames"]
    
    # Calculate per-video detection rates
    detection_rates = []
    for video_id in video_dirs:
        if video_id in face_counts and video_id in frame_counts and frame_counts[video_id] > 0:
            rate = face_counts[video_id] / frame_counts[video_id]
            detection_rates.append(rate)
    
    if detection_rates:
        metrics["average_detection_rate"] = np.mean(detection_rates)
        metrics["min_detection_rate"] = np.min(detection_rates)
        metrics["max_detection_rate"] = np.max(detection_rates)
    
    return metrics

def collect_optical_flow_metrics(flow_dir, faces_dir):
    """
    Collect metrics about optical flow computation
    
    Args:
        flow_dir (str): Directory containing optical flow data
        faces_dir (str): Directory containing detected faces
        
    Returns:
        dict: Optical flow metrics
    """
    metrics = {
        "total_videos": 0,
        "processed_videos": 0,
        "failed_videos": 0,
        "total_flows": 0,
        "total_faces": 0,
        "flow_computation_rate": 0,
        "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Count videos with faces
    video_dirs = set()
    face_counts = {}
    
    for root, dirs, files in os.walk(faces_dir):
        face_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')]
        
        if face_files:
            video_id = os.path.basename(root)
            video_dirs.add(video_id)
            face_counts[video_id] = len(face_files)
    
    metrics["total_videos"] = len(video_dirs)
    
    # Count videos with optical flow
    processed_videos = set()
    flow_counts = {}
    
    for root, dirs, files in os.walk(flow_dir):
        flow_files = [f for f in files if f.endswith('_flow_rgb.jpg')]
        
        if flow_files:
            video_id = os.path.basename(root)
            processed_videos.add(video_id)
            flow_counts[video_id] = len(flow_files)
    
    metrics["processed_videos"] = len(processed_videos)
    metrics["failed_videos"] = len(video_dirs) - len(processed_videos)
    metrics["total_flows"] = sum(flow_counts.values()) if flow_counts else 0
    metrics["total_faces"] = sum(face_counts.values()) if face_counts else 0
    
    # For optical flow, we expect one less flow than faces (flow between consecutive faces)
    expected_flows = 0
    for video_id, face_count in face_counts.items():
        # We can compute flow between consecutive faces, so expected flows = faces - 1
        if face_count > 1:
            expected_flows += (face_count - 1)
    
    metrics["expected_flows"] = expected_flows
    
    # Calculate flow computation rate
    if expected_flows > 0:
        metrics["flow_computation_rate"] = metrics["total_flows"] / expected_flows
    
    # Calculate average flow quality if raw flow data is available
    flow_magnitudes = []
    
    # Sample up to 100 flow files to assess quality
    sample_count = 0
    for video_id in processed_videos:
        video_flow_dir = os.path.join(flow_dir, video_id)
        if not os.path.exists(video_flow_dir):
            continue
            
        flow_files = [f for f in os.listdir(video_flow_dir) if f.endswith('_flow_raw.npy')]
        for flow_file in flow_files[:10]:  # Sample up to 10 per video
            if sample_count >= 100:
                break
                
            flow_path = os.path.join(video_flow_dir, flow_file)
            try:
                flow = np.load(flow_path)
                u, v = flow[..., 0], flow[..., 1]
                magnitude = np.sqrt(u**2 + v**2)
                flow_magnitudes.append(np.mean(magnitude))
                sample_count += 1
            except Exception as e:
                logging.warning(f"Error loading flow file {flow_path}: {e}")
    
    if flow_magnitudes:
        metrics["average_flow_magnitude"] = np.mean(flow_magnitudes)
        metrics["std_flow_magnitude"] = np.std(flow_magnitudes)
        metrics["min_flow_magnitude"] = np.min(flow_magnitudes)
        metrics["max_flow_magnitude"] = np.max(flow_magnitudes)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Track preprocessing metrics")
    
    parser.add_argument('--frames-dir', type=str, required=True,
                        help="Directory containing extracted frames")
    parser.add_argument('--faces-dir', type=str, required=True,
                        help="Directory containing detected faces")
    parser.add_argument('--flow-dir', type=str, required=True,
                        help="Directory containing optical flow data")
    parser.add_argument('--raw-dir', type=str, required=True,
                        help="Directory containing raw videos")
    parser.add_argument('--output-dir', type=str, default="results",
                        help="Directory to save metrics and visualizations")
    parser.add_argument('--log-config', type=str,
                        help="Path to logging configuration YAML")
    
    args = parser.parse_args()
    
    # Initialize the data pipeline logger
    logger = DataPipelineLogger(
        config_path=args.log_config,
        experiment_name=f"preprocessing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir=args.output_dir
    )
    
    logger.logger.info("Starting preprocessing metrics collection...")
    
    # Collect frame extraction metrics
    logger.logger.info("Collecting frame extraction metrics...")
    frame_metrics = collect_frame_extraction_metrics(args.frames_dir, args.raw_dir)
    logger.log_preprocessing_metrics("frame_extraction", frame_metrics)
    
    # Collect face detection metrics
    logger.logger.info("Collecting face detection metrics...")
    face_metrics = collect_face_detection_metrics(args.faces_dir, args.frames_dir)
    logger.log_preprocessing_metrics("face_detection", face_metrics)
    
    # Collect optical flow metrics
    logger.logger.info("Collecting optical flow metrics...")
    flow_metrics = collect_optical_flow_metrics(args.flow_dir, args.faces_dir)
    logger.log_preprocessing_metrics("optical_flow", flow_metrics)
    
    # Create summary visualization
    logger.logger.info("Creating summary visualization...")
    
    # Save preprocessing summary
    logger.save_preprocessing_summary()
    
    logger.logger.info("Preprocessing metrics collection completed.")

if __name__ == "__main__":
    main()
