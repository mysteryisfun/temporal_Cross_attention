#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Data Pipeline Logging with Sample Data

This script tests the data pipeline logging implementation with a small subset of data.
It runs the dataset statistics collection, preprocessing metrics tracking, and
visualization generation on a limited number of videos to verify functionality.
"""

import os
import sys
import argparse
import shutil
import random
import pickle
from pathlib import Path
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline_logger import DataPipelineLogger
from utils.json_utils import convert_numpy_types

def create_sample_dataset(data_dir, sample_dir, num_videos=5):
    """
    Create a sample dataset with a limited number of videos
    
    Args:
        data_dir (str): Original data directory
        sample_dir (str): Sample data directory to create
        num_videos (int): Number of videos to include in sample
        
    Returns:
        dict: Paths to sample directories
    """
    # Create sample directories
    sample_paths = {
        'root': sample_dir,
        'raw': os.path.join(sample_dir, 'raw'),
        'processed': os.path.join(sample_dir, 'processed'),
        'faces': os.path.join(sample_dir, 'faces'),
        'flow': os.path.join(sample_dir, 'optical_flow')
    }
    
    # Create directories
    for path in sample_paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Find all video directories in faces directory
    faces_dir = os.path.join(data_dir, 'faces')
    video_dirs = [d for d in os.listdir(faces_dir) 
                 if os.path.isdir(os.path.join(faces_dir, d))]
    
    if not video_dirs:
        logging.error(f"No video directories found in {faces_dir}")
        return sample_paths
    
    # Select random sample of videos
    if len(video_dirs) > num_videos:
        selected_videos = random.sample(video_dirs, num_videos)
    else:
        selected_videos = video_dirs
    
    logging.info(f"Selected {len(selected_videos)} videos for sample dataset")
    
    # Copy sample videos to sample directories
    for video_id in selected_videos:
        # Copy faces
        src_face_dir = os.path.join(faces_dir, video_id)
        dst_face_dir = os.path.join(sample_paths['faces'], video_id)
        
        if os.path.exists(src_face_dir):
            shutil.copytree(src_face_dir, dst_face_dir, dirs_exist_ok=True)
        
        # Copy optical flow
        src_flow_dir = os.path.join(data_dir, 'optical_flow', video_id)
        dst_flow_dir = os.path.join(sample_paths['flow'], video_id)
        
        if os.path.exists(src_flow_dir):
            shutil.copytree(src_flow_dir, dst_flow_dir, dirs_exist_ok=True)
        
        # Copy processed frames
        src_proc_dir = os.path.join(data_dir, 'processed', video_id)
        dst_proc_dir = os.path.join(sample_paths['processed'], video_id)
        
        if os.path.exists(src_proc_dir):
            shutil.copytree(src_proc_dir, dst_proc_dir, dirs_exist_ok=True)
      # Copy annotation file
    # Check in both raw and train-annotation folders
    for anno_file in ['annotation_training.pkl', 'annotation_validation.pkl']:
        # Try in raw directory first
        src_anno_path = os.path.join(data_dir, 'raw', anno_file)
        if os.path.exists(src_anno_path):
            shutil.copy2(src_anno_path, os.path.join(sample_paths['raw'], anno_file))
            continue
            
        # Try in train-annotation directory
        src_anno_path = os.path.join(data_dir, 'raw', 'train-annotation', anno_file)
        if os.path.exists(src_anno_path):
            shutil.copy2(src_anno_path, os.path.join(sample_paths['raw'], anno_file))
    
    return sample_paths

def run_test_analysis(sample_paths, annotation_file, output_dir):
    """
    Run test analysis on sample dataset
    
    Args:
        sample_paths (dict): Paths to sample directories
        annotation_file (str): Path to annotation file
        output_dir (str): Directory for output results
    """
    # Initialize logger
    logger = DataPipelineLogger(
        experiment_name="test_data_pipeline",
        output_dir=output_dir
    )
    
    logger.logger.info("Starting test analysis on sample dataset...")
    
    try:        # Test dataset statistics collection
        logger.logger.info("Testing dataset statistics collection...")
        
        # Fix annotation loading with different encodings
        def load_annotations_with_encoding(annotation_file):
            """Load annotations with multiple encoding attempts"""
            encodings = ['ascii', 'latin1', 'utf-8', None]
            
            for encoding in encodings:
                try:
                    with open(annotation_file, 'rb') as f:
                        if encoding:
                            return pickle.load(f, encoding=encoding)
                        else:
                            return pickle.load(f)
                except Exception:
                    continue
            
            return None
          # Monkey patch the load_annotations function in collect_dataset_statistics
        import scripts.collect_dataset_statistics
        original_load = scripts.collect_dataset_statistics.load_annotations
        scripts.collect_dataset_statistics.load_annotations = load_annotations_with_encoding
        
        from scripts.collect_dataset_statistics import collect_dataset_statistics
        
        stats = collect_dataset_statistics(
            annotation_file,
            sample_paths['faces'],
            sample_paths['flow'],
            os.path.join(output_dir, "visualizations")
        )
        
        if stats:
            # Convert numpy types before logging
            processed_stats = {"dataset_stats": convert_numpy_types(stats["dataset_stats"])}
            logger.log_dataset_statistics("test_dataset", processed_stats["dataset_stats"])
            logger.logger.info("Dataset statistics collection test passed.")
        else:
            logger.logger.error("Dataset statistics collection test failed.")
        
        # Test preprocessing metrics tracking
        logger.logger.info("Testing preprocessing metrics tracking...")
        
        from scripts.track_preprocessing_metrics import (
            collect_frame_extraction_metrics,
            collect_face_detection_metrics,
            collect_optical_flow_metrics
        )
        
        # Frame extraction metrics
        frame_metrics = collect_frame_extraction_metrics(
            sample_paths['processed'], 
            sample_paths['raw']
        )
        logger.log_preprocessing_metrics("frame_extraction", frame_metrics)
        
        # Face detection metrics
        face_metrics = collect_face_detection_metrics(
            sample_paths['faces'], 
            sample_paths['processed']
        )
        logger.log_preprocessing_metrics("face_detection", face_metrics)
        
        # Optical flow metrics
        flow_metrics = collect_optical_flow_metrics(
            sample_paths['flow'], 
            sample_paths['faces']
        )
        logger.log_preprocessing_metrics("optical_flow", flow_metrics)
        
        logger.logger.info("Preprocessing metrics tracking test passed.")
        
        # Test visualization generation
        logger.logger.info("Testing visualization generation...")
        
        from scripts.visualize_data_pipeline import (
            create_data_pipeline_visualization,
            create_side_by_side_visualization
        )
        
        vis_dir = os.path.join(output_dir, "visualizations", "data_pipeline")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create visualizations
        vis_paths = create_data_pipeline_visualization(
            sample_paths['faces'],
            sample_paths['flow'],
            os.path.join(output_dir, "preprocessing_metrics"),
            vis_dir
        )
        
        # Create side-by-side visualizations
        side_by_side_paths = create_side_by_side_visualization(
            sample_paths['faces'],
            sample_paths['flow'],
            vis_dir
        )
        
        if vis_paths or side_by_side_paths:
            logger.logger.info("Visualization generation test passed.")
        else:
            logger.logger.warning("Visualization generation test completed but no visualizations were created.")
        
        # Save summary
        logger.save_preprocessing_summary()
        
        logger.logger.info("Test analysis completed successfully.")
        
    except Exception as e:
        logger.logger.error(f"Test analysis failed: {str(e)}")
        import traceback
        logger.logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Test data pipeline logging with sample data")
    
    parser.add_argument('--data-dir', type=str, default="data",
                        help="Original data directory")
    parser.add_argument('--sample-dir', type=str, default="data_sample",
                        help="Directory for sample dataset")
    parser.add_argument('--output-dir', type=str, default="results_test",
                        help="Directory for test results")
    parser.add_argument('--num-videos', type=int, default=5,
                        help="Number of videos to include in sample")
    parser.add_argument('--no-create-sample', action='store_true',
                        help="Skip sample dataset creation (use existing)")
    parser.add_argument('--annotation-file', type=str,
                        help="Path to annotation file (will be copied to sample dataset)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create sample dataset if needed
    if not args.no_create_sample:
        logging.info(f"Creating sample dataset in {args.sample_dir} with {args.num_videos} videos")
        sample_paths = create_sample_dataset(args.data_dir, args.sample_dir, args.num_videos)
    else:
        logging.info(f"Using existing sample dataset in {args.sample_dir}")
        sample_paths = {
            'root': args.sample_dir,
            'raw': os.path.join(args.sample_dir, 'raw'),
            'processed': os.path.join(args.sample_dir, 'processed'),
            'faces': os.path.join(args.sample_dir, 'faces'),
            'flow': os.path.join(args.sample_dir, 'optical_flow')
        }
    
    # Find annotation file
    if args.annotation_file:
        # Use provided annotation file and copy to sample dataset
        if os.path.exists(args.annotation_file):
            os.makedirs(sample_paths['raw'], exist_ok=True)
            dst_path = os.path.join(sample_paths['raw'], os.path.basename(args.annotation_file))
            shutil.copy2(args.annotation_file, dst_path)
            annotation_file = dst_path
            logging.info(f"Copied annotation file to {dst_path}")
        else:
            logging.error(f"Provided annotation file does not exist: {args.annotation_file}")
            return
    else:
        # Look for annotation files in sample dataset
        annotation_file = os.path.join(sample_paths['raw'], 'annotation_validation.pkl')
        if not os.path.exists(annotation_file):
            annotation_file = os.path.join(sample_paths['raw'], 'annotation_training.pkl')
          # If still not found, look in original data directory
        if not os.path.exists(annotation_file):
            for anno_name in ['annotation_validation.pkl', 'annotation_training.pkl']:
                # Check in raw directory
                orig_anno = os.path.join(args.data_dir, 'raw', anno_name)
                if os.path.exists(orig_anno):
                    os.makedirs(sample_paths['raw'], exist_ok=True)
                    dst_path = os.path.join(sample_paths['raw'], anno_name)
                    shutil.copy2(orig_anno, dst_path)
                    annotation_file = dst_path
                    logging.info(f"Copied annotation file from original data: {dst_path}")
                    break
                    
                # Check in train-annotation directory
                orig_anno = os.path.join(args.data_dir, 'raw', 'train-annotation', anno_name)
                if os.path.exists(orig_anno):
                    os.makedirs(sample_paths['raw'], exist_ok=True)
                    dst_path = os.path.join(sample_paths['raw'], anno_name)
                    shutil.copy2(orig_anno, dst_path)
                    annotation_file = dst_path
                    logging.info(f"Copied annotation file from train-annotation: {dst_path}")
                    break
    
    if not os.path.exists(annotation_file):
        logging.error(f"No annotation file found. Please specify with --annotation-file")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test analysis
    logging.info(f"Running test analysis with results in {args.output_dir}")
    run_test_analysis(sample_paths, annotation_file, args.output_dir)
    
    logging.info("Test completed.")
    logging.info(f"Sample dataset: {args.sample_dir}")
    logging.info(f"Test results: {args.output_dir}")

if __name__ == "__main__":
    main()
