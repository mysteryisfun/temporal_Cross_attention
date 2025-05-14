#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: compute_optical_flow.py
# Description: Command-line script to compute optical flow between consecutive frames

import os
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.optical_flow_computer import setup_logger, process_all_videos, visualize_flow_quality, analyze_flow_magnitude

def main():
    parser = argparse.ArgumentParser(description="Compute optical flow between consecutive frames")
    
    # Input/output options
    parser.add_argument('--faces-dir', type=str, required=True,
                        help="Directory containing face images")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save optical flow results")
    
    # Method options
    parser.add_argument('--method', type=str, default='farneback',
                        choices=['farneback', 'tvl1'],
                        help="Optical flow method to use")
    
    # Farneback parameters
    parser.add_argument('--pyr-scale', type=float, default=0.5,
                        help="Farneback: Scale for pyramid structures")
    parser.add_argument('--levels', type=int, default=3,
                        help="Farneback: Number of pyramid layers")
    parser.add_argument('--winsize', type=int, default=15,
                        help="Farneback: Window size")
    parser.add_argument('--iterations', type=int, default=3,
                        help="Farneback: Number of iterations")
    parser.add_argument('--poly-n', type=int, default=5,
                        help="Farneback: Polynomial degree for pixel neighborhoods")
    parser.add_argument('--poly-sigma', type=float, default=1.2,
                        help="Farneback: Gaussian sigma for smoothing derivatives")
    
    # TVL1 parameters
    parser.add_argument('--tau', type=float, default=0.25,
                        help="TVL1: Time step size")
    parser.add_argument('--lambda', type=float, dest='lambda_', default=0.15,
                        help="TVL1: Weight parameter for data term")
    parser.add_argument('--theta', type=float, default=0.3,
                        help="TVL1: Weight parameter for smoothness term")
    parser.add_argument('--nscales', type=int, default=5,
                        help="TVL1: Number of scales")
    parser.add_argument('--warps', type=int, default=5,
                        help="TVL1: Number of warps per scale")
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help="TVL1: Stopping criterion threshold")
    parser.add_argument('--tvl1-iterations', type=int, default=300,
                        help="TVL1: Maximum number of iterations")
    
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
                        help="Number of parallel workers")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help="Generate visualizations of optical flow results")
    parser.add_argument('--analyze', action='store_true',
                        help="Analyze optical flow magnitude distribution")
    
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
        log_file = './logs/optical_flow.log'
    
    logger = setup_logger(log_file)
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure parameters based on the selected method
    if args.method == 'farneback':
        params = {
            'pyr_scale': args.pyr_scale,
            'levels': args.levels,
            'winsize': args.winsize,
            'iterations': args.iterations,
            'poly_n': args.poly_n,
            'poly_sigma': args.poly_sigma,
            'flags': 0
        }
    else:  # TVL1
        params = {
            'tau': args.tau,
            'lambda_': args.lambda_,
            'theta': args.theta,
            'nscales': args.nscales,
            'warps': args.warps,
            'epsilon': args.epsilon,
            'iterations': args.tvl1_iterations
        }
      # Log parameters
    logger.info(f"Computing optical flow using {args.method} method")
    logger.info(f"Parameters: {params}")
    logger.info(f"Input directory: {args.faces_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Save RGB: {args.save_rgb}")
    logger.info(f"Save raw data: {args.save_raw}")
    
    # Handle two possible input modes:
    # 1. Single video directory with face images
    # 2. Directory containing multiple video directories
    
    # Check if input directory directly contains face images
    face_files = [f for f in os.listdir(args.faces_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if face_files and all(os.path.isfile(os.path.join(args.faces_dir, f)) for f in face_files):
        # Case 1: Single video directory
        video_id = os.path.basename(args.faces_dir)
        logger.info(f"Processing single video directory: {video_id}")
        
        video_output_dir = args.output_dir
        if not video_output_dir.endswith(video_id):
            video_output_dir = os.path.join(args.output_dir, video_id)
            
        results = process_all_videos(
            args.faces_dir,
            video_output_dir,
            args.method,
            params,
            args.save_rgb,
            args.save_raw,
            args.workers,
            logger
        )
    else:
        # Case 2: Directory with multiple video directories
        logger.info(f"Processing multiple video directories")
        
        results = process_all_videos(
            args.faces_dir,
            args.output_dir,
            args.method,
            params,
            args.save_rgb,
            args.save_raw,
            args.workers,
            logger
        )
    
    # Generate visualizations if requested
    if args.visualize:
        vis_path = os.path.join(args.output_dir, "flow_samples.png")
        logger.info(f"Generating flow visualizations at {vis_path}")
        visualize_flow_quality(args.output_dir, vis_path)
    
    # Analyze flow magnitude if requested
    if args.analyze and args.save_raw:
        analysis_path = os.path.join(args.output_dir, "flow_magnitude_analysis.png")
        logger.info(f"Analyzing flow magnitude at {analysis_path}")
        stats = analyze_flow_magnitude(args.output_dir, analysis_path)
        if stats:
            logger.info(f"Average flow magnitude: {stats['mean']:.2f}")
            logger.info(f"Maximum flow magnitude: {stats['max']:.2f}")
    
    logger.info("Optical flow computation completed.")

if __name__ == "__main__":
    main()
