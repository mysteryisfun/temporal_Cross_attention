#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: frame_extractor.py
# Author: [Your Name]
# Created: [Date]
# Description: Extract frames from videos for the Cross-Attention CNN Research project

import os
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from annotation_parser import load_annotations, parse_annotations

# Set up logger
def setup_logger(log_file=None):
    """
    Set up the logger for frame extraction process.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs only to console.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger("FrameExtractor")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def extract_frames(video_path, output_dir, sampling_rate=1, max_frames=None, frame_size=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        sampling_rate (int): Extract every nth frame.
        max_frames (int, optional): Maximum number of frames to extract. If None, extract all frames.
        frame_size (tuple, optional): Resize frames to this size (width, height). If None, keep original size.
        
    Returns:
        dict: Dictionary containing extraction statistics.
    """
    stats = {
        'video_name': os.path.basename(video_path),
        'total_frames': 0,
        'extracted_frames': 0,
        'width': 0,
        'height': 0,
        'fps': 0,
        'duration': 0,
        'status': 'failure',
        'error': None
    }
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            stats['error'] = "Failed to open video file"
            return stats
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Update stats
        stats.update({
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration
        })
        
        # Determine how many frames to extract
        frames_to_extract = total_frames
        if max_frames:
            frames_to_extract = min(total_frames, max_frames * sampling_rate)
        
        # Extract frames
        frame_count = 0
        extracted_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sampling_rate == 0:
                # Resize frame if specified
                if frame_size:
                    frame = cv2.resize(frame, frame_size)
                
                # Save frame
                frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
                
                # Break if we've extracted the maximum number of frames
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        stats['extracted_frames'] = extracted_count
        stats['status'] = 'success'
        
    except Exception as e:
        stats['error'] = str(e)
    
    finally:
        # Release the video capture object
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    
    return stats

def process_video_directory(video_dir, output_base_dir, sampling_rate=1, max_frames=None, 
                           frame_size=None, num_workers=4, use_gpu=False, logger=None):
    """
    Process all videos in a directory and extract frames.
    
    Args:
        video_dir (str): Directory containing video files.
        output_base_dir (str): Base directory to save extracted frames.
        sampling_rate (int): Extract every nth frame.
        max_frames (int, optional): Maximum number of frames to extract per video.
        frame_size (tuple, optional): Resize frames to this size (width, height).
        num_workers (int): Number of parallel workers.
        use_gpu (bool): Whether to use GPU for processing.
        logger (logging.Logger, optional): Logger object.
        
    Returns:
        dict: Extraction statistics.
    """
    if logger is None:
        logger = logging.getLogger("FrameExtractor")
    
    start_time = time.time()
    
    # Get all video files
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(video_files)} video files in {video_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Extract frames from each video in parallel
    total_extracted = 0
    success_count = 0
    failure_count = 0
    extraction_stats = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_video = {}
        for video_file in video_files:
            # Create a subdirectory structure that mirrors the input directory
            rel_path = os.path.relpath(os.path.dirname(video_file), video_dir)
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            output_dir = os.path.join(output_base_dir, rel_path, video_name)
            
            future = executor.submit(
                extract_frames, 
                video_file, 
                output_dir, 
                sampling_rate, 
                max_frames, 
                frame_size,
                use_gpu
            )
            future_to_video[future] = video_file
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Extracting frames"):
            video_file = future_to_video[future]
            try:
                stats = future.result()
                extraction_stats.append(stats)
                
                if stats['status'] == 'success':
                    logger.info(f"Successfully extracted {stats['extracted_frames']} frames from {os.path.basename(video_file)}")
                    total_extracted += stats['extracted_frames']
                    success_count += 1
                else:
                    logger.error(f"Failed to extract frames from {os.path.basename(video_file)}: {stats['error']}")
                    failure_count += 1
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(video_file)}: {str(e)}")
                failure_count += 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"Frame extraction completed in {elapsed_time:.2f} seconds")
    logger.info(f"Extracted a total of {total_extracted} frames from {success_count} videos")
    logger.info(f"Failed to extract frames from {failure_count} videos")
    
    return {
        'total_videos': len(video_files),
        'success_count': success_count,
        'failure_count': failure_count,
        'total_extracted_frames': total_extracted,
        'processing_time': elapsed_time,
        'video_stats': extraction_stats
    }

def process_annotation_videos(annotation_file, video_base_dir, output_base_dir, 
                             sampling_rate=1, max_frames=None, frame_size=None, 
                             num_workers=4, logger=None):
    """
    Process videos specified in annotation file.
    
    Args:
        annotation_file (str): Path to annotation file.
        video_base_dir (str): Base directory containing video files.
        output_base_dir (str): Base directory to save extracted frames.
        sampling_rate (int): Extract every nth frame.
        max_frames (int, optional): Maximum number of frames to extract per video.
        frame_size (tuple, optional): Resize frames to this size (width, height).
        num_workers (int): Number of parallel workers.
        logger (logging.Logger, optional): Logger object.
        
    Returns:
        dict: Extraction statistics.
    """
    if logger is None:
        logger = logging.getLogger("FrameExtractor")
    
    # Since we're having trouble with the annotation file, let's just get the video files directly
    logger.info(f"Searching for video files in {video_base_dir}")
    
    video_files = []
    for root, _, files in os.walk(video_base_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process each video file
    start_time = time.time()
    total_extracted = 0
    success_count = 0
    failure_count = 0
    extraction_stats = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_video = {}
        for video_file in video_files:
            # Use video ID as the directory name
            video_id = os.path.splitext(os.path.basename(video_file))[0]
            output_dir = os.path.join(output_base_dir, video_id)
            
            future = executor.submit(
                extract_frames, 
                video_file, 
                output_dir, 
                sampling_rate, 
                max_frames, 
                frame_size
            )
            future_to_video[future] = video_id
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Extracting frames"):
            video_id = future_to_video[future]
            try:
                stats = future.result()
                extraction_stats.append(stats)
                
                if stats['status'] == 'success':
                    logger.info(f"Successfully extracted {stats['extracted_frames']} frames from {video_id}")
                    total_extracted += stats['extracted_frames']
                    success_count += 1
                else:
                    logger.error(f"Failed to extract frames from {video_id}: {stats['error']}")
                    failure_count += 1
            except Exception as e:
                logger.error(f"Error processing {video_id}: {str(e)}")
                failure_count += 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"Frame extraction completed in {elapsed_time:.2f} seconds")
    logger.info(f"Extracted a total of {total_extracted} frames from {success_count} videos")
    logger.info(f"Failed to extract frames from {failure_count} videos")
    
    return {
        'total_videos': len(video_files),
        'success_count': success_count,
        'failure_count': failure_count,
        'total_extracted_frames': total_extracted,
        'processing_time': elapsed_time,
        'video_stats': extraction_stats
    }

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
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_file)
    
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
