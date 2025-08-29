#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: process_all_optical_flow.py
# Description: Script to process all face data and compute optical flow

import os
import sys
import argparse
import logging
import time
from tqdm import tqdm
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.optical_flow_computer import setup_logger, process_video_faces, visualize_flow_quality, analyze_flow_magnitude

def main():
    parser = argparse.ArgumentParser(description="Process all face data and compute optical flow")
    
    # Input/output options
    parser.add_argument('--faces-dir', type=str, default='data/faces',
                        help="Directory containing face images (default: data/faces)")
    parser.add_argument('--output-dir', type=str, default='data/optical_flow',
                        help="Directory to save optical flow results (default: data/optical_flow)")
    
    # Method options
    parser.add_argument('--method', type=str, default='farneback',
                        choices=['farneback', 'tvl1'],
                        help="Optical flow method to use (default: farneback)")
    
    # Output options
    parser.add_argument('--save-rgb', action='store_true', default=True,
                        help="Save RGB visualization of optical flow")
    parser.add_argument('--save-raw', action='store_true', default=True,
                        help="Save raw optical flow data as .npy files")
    parser.add_argument('--no-save-rgb', action='store_false', dest='save_rgb',
                        help="Don't save RGB visualization of optical flow")
    parser.add_argument('--no-save-raw', action='store_false', dest='save_raw',
                        help="Don't save raw optical flow data")
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel workers for each video batch")
    parser.add_argument('--batch-size', type=int, default=10,
                        help="Number of videos to process in each batch")
    parser.add_argument('--max-videos', type=int, default=-1,
                        help="Maximum number of videos to process (-1 for all)")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true', default=True,
                        help="Generate visualizations of optical flow results")
    parser.add_argument('--analyze', action='store_true', default=True,
                        help="Analyze optical flow magnitude distribution")
    
    # Logging options
    parser.add_argument('--log-file', type=str, default='logs/optical_flow_processing.log',
                        help="Path to log file")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set up logger
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(args.log_file)
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of all video directories
    video_dirs = []
    if os.path.exists(args.faces_dir):
        for item in os.listdir(args.faces_dir):
            item_path = os.path.join(args.faces_dir, item)
            if os.path.isdir(item_path):
                video_dirs.append(item_path)
    
    if not video_dirs:
        logger.error(f"No video directories found in {args.faces_dir}")
        return
    
    # Apply maximum limit if specified
    if args.max_videos > 0 and args.max_videos < len(video_dirs):
        logger.info(f"Limiting to {args.max_videos} videos (out of {len(video_dirs)} available)")
        video_dirs = video_dirs[:args.max_videos]
    
    logger.info(f"Found {len(video_dirs)} video directories to process")
    
    # Process videos in batches to manage memory
    total_videos = len(video_dirs)
    batch_size = min(args.batch_size, total_videos)
    num_batches = (total_videos + batch_size - 1) // batch_size  # Ceiling division
    
    start_time = time.time()
    successful_videos = 0
    failed_videos = 0
    total_flows = 0
    
    # Configure method parameters based on selected algorithm
    if args.method == 'farneback':
        params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
    else:  # TVL1
        params = {
            'tau': 0.25,
            'lambda_': 0.15,
            'theta': 0.3,
            'nscales': 5,
            'warps': 5,
            'epsilon': 0.01,
            'iterations': 300
        }
    
    logger.info(f"Starting optical flow computation using {args.method} method")
    logger.info(f"Processing {total_videos} videos in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_videos)
        batch_videos = video_dirs[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_videos)} videos)")
        
        for video_path in tqdm(batch_videos, desc=f"Batch {batch_idx + 1}/{num_batches}", unit="video"):
            video_id = os.path.basename(video_path)
            video_output_dir = os.path.join(args.output_dir, video_id)
            
            try:
                result = process_video_faces(
                    video_path,
                    args.output_dir,
                    args.method,
                    params,
                    args.save_rgb,
                    args.save_raw,
                    logger
                )
                
                if result['status'] == 'success':
                    successful_videos += 1
                    total_flows += result['num_flows']
                    logger.info(f"Successfully processed {video_id} with {result['num_flows']} flows")
                else:
                    failed_videos += 1
                    logger.warning(f"Failed to process {video_id}: {result['error']}")
            except Exception as e:
                failed_videos += 1
                logger.error(f"Error processing {video_id}: {str(e)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Compute summary statistics
    success_rate = successful_videos / total_videos * 100 if total_videos > 0 else 0
    
    # Log summary
    logger.info(f"Optical flow computation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"Successfully processed: {successful_videos} ({success_rate:.2f}%)")
    logger.info(f"Failed to process: {failed_videos}")
    logger.info(f"Total flows computed: {total_flows}")
    
    # Generate visualizations if requested
    if args.visualize:
        vis_path = os.path.join(args.output_dir, "flow_samples.png")
        logger.info(f"Generating flow visualizations at {vis_path}")
        visualize_flow_quality(args.output_dir, vis_path, num_samples=10)
    
    # Analyze flow magnitude if requested
    if args.analyze and args.save_raw:
        analysis_path = os.path.join(args.output_dir, "flow_magnitude_analysis.png")
        logger.info(f"Analyzing flow magnitude at {analysis_path}")
        stats = analyze_flow_magnitude(args.output_dir, analysis_path)
        if stats:
            logger.info(f"Average flow magnitude: {stats['mean']:.2f}")
            logger.info(f"Maximum flow magnitude: {stats['max']:.2f}")
    
    logger.info("Optical flow processing completed.")

if __name__ == "__main__":
    main()
