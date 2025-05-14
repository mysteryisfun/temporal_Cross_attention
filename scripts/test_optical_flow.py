#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_optical_flow.py
# Description: Test script for optical flow computation module

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.optical_flow_computer import setup_logger, process_all_videos, visualize_flow_quality, analyze_flow_magnitude

def main():
    parser = argparse.ArgumentParser(description="Test optical flow computation on faces")
    
    # Input/output options
    parser.add_argument('--faces-dir', type=str, default='data/faces',
                        help="Directory containing face images (default: data/faces)")
    parser.add_argument('--output-dir', type=str, default='data/optical_flow',
                        help="Directory to save optical flow results (default: data/optical_flow)")
    parser.add_argument('--num-samples', type=int, default=5,
                        help="Number of video samples to process (default: 5, use -1 for all)")
    
    # Method options
    parser.add_argument('--method', type=str, default='farneback',
                        choices=['farneback', 'tvl1'],
                        help="Optical flow method to use (default: farneback)")
    
    # Output options
    parser.add_argument('--save-rgb', action='store_true', default=True,
                        help="Save RGB visualization of optical flow")
    parser.add_argument('--save-raw', action='store_true', default=True,
                        help="Save raw optical flow data as .npy files")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true', default=True,
                        help="Generate visualizations of optical flow results")
    parser.add_argument('--analyze', action='store_true', default=True,
                        help="Analyze optical flow magnitude distribution")
    
    # Logging options
    parser.add_argument('--log-file', type=str, default='logs/optical_flow_test.log',
                        help="Path to log file")
    
    args = parser.parse_args()
    
    # Set up logger
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(args.log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of all video directories
    all_videos = []
    for item in os.listdir(args.faces_dir):
        item_path = os.path.join(args.faces_dir, item)
        if os.path.isdir(item_path):
            all_videos.append(item_path)
    
    logger.info(f"Found {len(all_videos)} video directories in {args.faces_dir}")
    
    # Select specified number of samples
    if args.num_samples > 0 and args.num_samples < len(all_videos):
        import random
        random.shuffle(all_videos)
        selected_videos = all_videos[:args.num_samples]
        logger.info(f"Selected {len(selected_videos)} sample videos for processing")
    else:
        selected_videos = all_videos
        logger.info(f"Processing all {len(selected_videos)} videos")
    
    # Process each video individually to ensure success
    successful_videos = 0
    failed_videos = 0
    total_flows = 0
    
    for video_path in selected_videos:
        video_id = os.path.basename(video_path)
        video_output_dir = os.path.join(args.output_dir, video_id)
        
        logger.info(f"Processing video {video_id}")
        
        # Process the video
        result = process_all_videos(
            video_path,
            video_output_dir,
            args.method,
            None,  # default parameters
            args.save_rgb,
            args.save_raw,
            1,  # Single worker since we're processing one video at a time
            logger
        )
        
        if result['successful_videos'] > 0:
            successful_videos += 1
            total_flows += result['total_flows']
            logger.info(f"Successfully processed {video_id}")
        else:
            failed_videos += 1
            logger.warning(f"Failed to process {video_id}")
    
    # Log summary
    logger.info(f"Summary:")
    logger.info(f"Total videos processed: {len(selected_videos)}")
    logger.info(f"Successfully processed: {successful_videos}")
    logger.info(f"Failed to process: {failed_videos}")
    logger.info(f"Total flows computed: {total_flows}")
    
    # Visualize results
    if args.visualize:
        vis_path = os.path.join(args.output_dir, "flow_samples.png")
        logger.info(f"Generating flow visualizations at {vis_path}")
        visualize_flow_quality(args.output_dir, vis_path)
    
    # Analyze flow magnitude
    if args.analyze and args.save_raw:
        analysis_path = os.path.join(args.output_dir, "flow_magnitude_analysis.png")
        logger.info(f"Analyzing flow magnitude at {analysis_path}")
        stats = analyze_flow_magnitude(args.output_dir, analysis_path)
        if stats:
            logger.info(f"Average flow magnitude: {stats['mean']:.2f}")
            logger.info(f"Maximum flow magnitude: {stats['max']:.2f}")
    
    logger.info("Optical flow test completed.")

if __name__ == "__main__":
    main()
