#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: process_faces.py
# Description: Utility script to detect and align faces in extracted frames

import os
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.face_detector import setup_logger, process_frames_directory, visualize_detection_results

def main():
    parser = argparse.ArgumentParser(description="Detect and align faces in video frames")
    
    # Input/output options
    parser.add_argument('--input-dir', type=str, required=True,
                        help="Directory containing input frames")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save processed faces")
    
    # Processing options
    parser.add_argument('--confidence', type=float, default=0.9,
                        help="Minimum face detection confidence threshold (0-1)")
    parser.add_argument('--min-size', type=int, default=80,
                        help="Minimum face size to process (pixels)")
    parser.add_argument('--target-size', type=str, default="224x224",
                        help="Target size for face images (WxH)")
    parser.add_argument('--all-faces', action='store_true',
                        help="Process all detected faces instead of just the largest")
    parser.add_argument('--recursive', action='store_true',
                        help="Process subdirectories recursively")
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel workers")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize detection results")
    parser.add_argument('--vis-output', type=str,
                        help="Path to save visualization")
    
    # Logging options
    parser.add_argument('--log-file', type=str,
                        help="Path to log file")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set up logger
    log_file = args.log_file
    if not log_file and not os.path.exists('./logs'):
        os.makedirs('./logs', exist_ok=True)
        log_file = './logs/face_detection.log'
    
    logger = setup_logger(log_file)
    logger.setLevel(getattr(logging, args.log_level))
    
    # Parse target size
    try:
        target_w, target_h = map(int, args.target_size.split('x'))
        target_size = (target_w, target_h)
    except ValueError:
        logger.error("Invalid target size format. Use WxH (e.g., 224x224)")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process frames
    logger.info(f"Starting face detection and alignment on frames in {args.input_dir}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Minimum face size: {args.min_size}px")
    logger.info(f"Target size: {target_size[0]}x{target_size[1]}px")
    logger.info(f"Processing {'all faces' if args.all_faces else 'largest face only'}")
    logger.info(f"Output directory: {args.output_dir}")
    
    results = process_frames_directory(
        args.input_dir,
        args.output_dir,
        args.confidence,
        args.min_size,
        target_size,
        not args.all_faces,
        args.workers,
        args.recursive,
        logger
    )
    
    # Visualize results if requested
    if args.visualize:
        vis_output = args.vis_output or os.path.join(args.output_dir, "detection_results.png")
        logger.info(f"Generating visualization at {vis_output}")
        visualize_detection_results(results, vis_output)
    
    logger.info("Face detection and alignment completed.")
    
if __name__ == "__main__":
    main()
