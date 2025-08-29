#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: batch_test_dynamic_extractor.py
# Description: Process multiple videos and test the dynamic feature extractor

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger

def process_videos_batch(video_dir, output_base_dir, num_videos=5):
    """
    Process multiple videos through the dynamic feature extractor test
    
    Args:
        video_dir (str): Directory containing video files
        output_base_dir (str): Base directory to save test results
        num_videos (int): Number of videos to process
        
    Returns:
        dict: Summary of processing results
    """
    logger = get_logger(experiment_name="batch_dynamic_extractor_test")
    logger.logger.info(f"Processing up to {num_videos} videos from {video_dir}")
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find video files
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        logger.logger.error(f"No video files found in {video_dir}")
        return {'status': 'error', 'message': 'No video files found'}
    
    # Sort and limit to specified number
    video_files.sort()
    if num_videos > 0:
        video_files = video_files[:num_videos]
    
    logger.logger.info(f"Found {len(video_files)} video files")
    
    # Process each video
    results = {
        'total_videos': len(video_files),
        'successful': 0,
        'failed': 0,
        'video_results': {}
    }
    
    for i, video_file in enumerate(tqdm(video_files, desc="Processing videos")):
        try:
            video_name = os.path.basename(video_file).split('.')[0]
            test_output_dir = os.path.join(output_base_dir, f"video_{i+1}_{video_name}")
            
            # Run the test script on this video
            logger.logger.info(f"Processing video {i+1}/{len(video_files)}: {video_name}")
            
            # Build the command
            cmd = [
                "python", "-m", "scripts.test_dynamic_extractor_small_sample",
                "--video", video_file,
                "--output_dir", test_output_dir
            ]
            
            # Execute the command
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                status = "success"
                results['successful'] += 1
            else:
                status = "failed"
                results['failed'] += 1
                
            # Store the result
            results['video_results'][video_name] = {
                'status': status,
                'output_dir': test_output_dir,
                'video_path': video_file,
                'exit_code': process.returncode
            }
            
            logger.logger.info(f"Completed processing {video_name} with status: {status}")
            
        except Exception as e:
            logger.logger.error(f"Error processing {video_file}: {str(e)}")
            results['failed'] += 1
            results['video_results'][os.path.basename(video_file)] = {
                'status': 'error',
                'message': str(e),
                'video_path': video_file
            }
    
    # Save the summary
    summary_path = os.path.join(output_base_dir, "batch_processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.logger.info(f"Saved batch processing summary to {summary_path}")
    
    # Print summary
    print(f"\nBatch Processing Summary:")
    print(f"-------------------------")
    print(f"Total videos processed: {results['total_videos']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Details saved to: {summary_path}")
    
    return results

def main():
    """
    Main function to run batch testing of the dynamic feature extractor
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch test dynamic feature extractor")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_feature_extractor/batch_test",
                       help="Directory to save test results")
    parser.add_argument("--num_videos", type=int, default=5,
                       help="Number of videos to process")
    args = parser.parse_args()
    
    # Process videos
    results = process_videos_batch(
        video_dir=args.video_dir,
        output_base_dir=args.output_dir,
        num_videos=args.num_videos
    )
    
    if results['successful'] > 0:
        print("\nTo aggregate and analyze these results further, you can:")
        print(f"1. Examine the individual test outputs in {args.output_dir}")
        print(f"2. Run the comprehensive evaluation script on the generated features")

if __name__ == "__main__":
    main()
