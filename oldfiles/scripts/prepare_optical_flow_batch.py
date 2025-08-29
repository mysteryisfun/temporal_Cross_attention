#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: prepare_optical_flow_batch.py
# Description: Process multiple videos to create optical flow sequences for batch evaluation

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from scripts.compute_optical_flow import compute_optical_flow_for_video
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor

def process_videos_to_flow(video_dir, output_dir, num_videos=10, frames_per_video=16):
    """
    Process multiple videos to create optical flow sequences
    
    Args:
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save optical flow sequences
        num_videos (int): Number of videos to process
        frames_per_video (int): Number of frames to extract per video
        
    Returns:
        list: Paths to created optical flow sequences
    """
    logger = get_logger(experiment_name="batch_flow_preparation")
    logger.logger.info(f"Processing up to {num_videos} videos from {video_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        logger.logger.error(f"No video files found in {video_dir}")
        return []
    
    # Sort and limit to specified number
    video_files.sort()
    if num_videos > 0:
        video_files = video_files[:num_videos]
    
    logger.logger.info(f"Found {len(video_files)} video files")
    
    # Process each video
    flow_sequences = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            video_name = os.path.basename(video_file).split('.')[0]
            flow_dir = os.path.join(output_dir, video_name)
            os.makedirs(flow_dir, exist_ok=True)
            
            # Compute optical flow
            flow_sequence = compute_optical_flow_for_video(
                video_path=video_file,
                output_dir=flow_dir,
                num_frames=frames_per_video,
                save_flow=True,
                flow_filename=f"{video_name}_flow_raw.npy"
            )
            
            if flow_sequence is not None:
                flow_path = os.path.join(flow_dir, f"{video_name}_flow_raw.npy")
                flow_sequences.append(flow_path)
                logger.logger.info(f"Created flow sequence: {flow_path}")
        except Exception as e:
            logger.logger.error(f"Error processing {video_file}: {str(e)}")
    
    logger.logger.info(f"Processed {len(flow_sequences)} videos successfully")
    return flow_sequences

def main():
    """
    Main function to prepare optical flow sequences for batch evaluation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare optical flow sequences for batch evaluation")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="data/optical_flow/batch_test",
                       help="Directory to save optical flow sequences")
    parser.add_argument("--num_videos", type=int, default=10,
                       help="Number of videos to process")
    parser.add_argument("--frames_per_video", type=int, default=16,
                       help="Number of frames to extract per video")
    args = parser.parse_args()
    
    # Process videos
    flow_sequences = process_videos_to_flow(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video
    )
    
    # If flow sequences were created, inform the user how to run evaluation
    if flow_sequences:
        print("\nOptical flow sequences created successfully.")
        print(f"To evaluate the dynamic feature extractor with these sequences, run:")
        print(f"python -m scripts.evaluate_dynamic_feature_extractor --flow_dir \"{args.output_dir}\" --output_dir \"results/dynamic_feature_extractor/batch_evaluation\"")

if __name__ == "__main__":
    main()
