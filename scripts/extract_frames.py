#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: extract_frames.py
# Author: [Your Name]
# Description: Utility script to extract frames from videos using frame_extractor module

import os
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.frame_extractor import setup_logger, process_video_directory, process_annotation_videos

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos for Cross-Attention CNN Research")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video-dir', type=str, help="Directory containing video files")
    input_group.add_argument('--annotation-file', type=str, help="Path to annotation file")
    
    # Output options
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save extracted frames")
    
    # Frame extraction options
    parser.add_argument('--sampling-rate', type=int, default=1, help="Extract every nth frame")
    parser.add_argument('--max-frames', type=int, help="Maximum number of frames to extract per video")
    parser.add_argument('--resize', type=str, help="Resize frames to WxH (e.g., 224x224)")
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--video-base-dir', type=str, help="Base directory containing video files (required with --annotation-file)")
    
    # Logging options
    parser.add_argument('--log-file', type=str, help="Path to log file")
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set up logger
    log_file = args.log_file
    if not log_file and not os.path.exists('./logs'):
        os.makedirs('./logs', exist_ok=True)
        log_file = './logs/frame_extraction.log'
    
    logger = setup_logger(log_file)
    logger.setLevel(getattr(logging, args.log_level))
    
    # Parse frame size
    frame_size = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            frame_size = (width, height)
            logger.info(f"Will resize frames to {width}x{height}")
        except ValueError:
            logger.error("Invalid resize format. Use WxH (e.g., 224x224)")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Process videos
    if args.video_dir:
        logger.info(f"Processing videos from directory: {args.video_dir}")
        stats = process_video_directory(
            args.video_dir,
            args.output_dir,
            args.sampling_rate,
            args.max_frames,
            frame_size,
            args.workers,
            logger
        )
    else:  # Using annotation file
        if not args.video_base_dir:
            logger.error("--video-base-dir is required when using --annotation-file")
            return
        
        logger.info(f"Processing videos from annotation file: {args.annotation_file}")
        stats = process_annotation_videos(
            args.annotation_file,
            args.video_base_dir,
            args.output_dir,
            args.sampling_rate,
            args.max_frames,
            frame_size,
            args.workers,
            logger
        )
    
    # Print summary statistics
    if stats:
        logger.info("Frame Extraction Summary:")
        logger.info(f"Total videos processed: {stats['total_videos']}")
        logger.info(f"Successfully processed: {stats['success_count']}")
        logger.info(f"Failed to process: {stats['failure_count']}")
        logger.info(f"Total frames extracted: {stats['total_extracted_frames']}")
        logger.info(f"Total processing time: {stats['processing_time']:.2f} seconds")
    
if __name__ == "__main__":
    main()
