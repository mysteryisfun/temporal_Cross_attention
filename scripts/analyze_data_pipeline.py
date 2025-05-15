#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Pipeline Analysis Script

This script runs a comprehensive analysis of the data pipeline, including:
1. Dataset statistics collection
2. Preprocessing metrics tracking
3. Data visualization generation

It combines all the data pipeline logging features into a single, easy-to-use script.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_pipeline_logger import DataPipelineLogger

def run_analysis_script(script_path, args):
    """Run a Python script with the given arguments"""
    cmd = [sys.executable, script_path] + args
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive data pipeline analysis")
    
    # Data directories
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Root directory containing dataset")
    parser.add_argument('--annotation-file', type=str, required=True,
                        help="Path to annotation pickle file")
    parser.add_argument('--output-dir', type=str, default="results",
                        help="Directory to save results")
    
    # Logging options
    parser.add_argument('--log-config', type=str,
                        help="Path to logging configuration YAML")
    
    # Control options
    parser.add_argument('--skip-statistics', action='store_true',
                        help="Skip dataset statistics collection")
    parser.add_argument('--skip-metrics', action='store_true',
                        help="Skip preprocessing metrics tracking")
    parser.add_argument('--skip-visualization', action='store_true',
                        help="Skip data visualization generation")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = args.data_dir
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    faces_dir = os.path.join(data_dir, "faces")
    flow_dir = os.path.join(data_dir, "optical_flow")
    
    output_dir = args.output_dir
    metrics_dir = os.path.join(output_dir, "preprocessing_metrics")
    vis_dir = os.path.join(output_dir, "visualizations", "data_pipeline")
    
    # Ensure directories exist
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize the data pipeline logger
    logger = DataPipelineLogger(
        config_path=args.log_config,
        experiment_name=f"data_pipeline_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir=output_dir
    )
    
    logger.logger.info("Starting data pipeline analysis...")
    
    # Step 1: Dataset statistics collection
    if not args.skip_statistics:
        logger.logger.info("Running dataset statistics collection...")
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_dataset_statistics.py")
        script_args = [
            "--annotation-file", args.annotation_file,
            "--faces-dir", faces_dir,
            "--flow-dir", flow_dir,
            "--output-dir", output_dir
        ]
        
        if args.log_config:
            script_args.extend(["--log-config", args.log_config])
        
        success = run_analysis_script(script_path, script_args)
        
        if success:
            logger.logger.info("Dataset statistics collection completed successfully.")
        else:
            logger.logger.error("Dataset statistics collection failed.")
    
    # Step 2: Preprocessing metrics tracking
    if not args.skip_metrics:
        logger.logger.info("Running preprocessing metrics tracking...")
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_preprocessing_metrics.py")
        script_args = [
            "--frames-dir", processed_dir,
            "--faces-dir", faces_dir,
            "--flow-dir", flow_dir,
            "--raw-dir", raw_dir,
            "--output-dir", output_dir
        ]
        
        if args.log_config:
            script_args.extend(["--log-config", args.log_config])
        
        success = run_analysis_script(script_path, script_args)
        
        if success:
            logger.logger.info("Preprocessing metrics tracking completed successfully.")
        else:
            logger.logger.error("Preprocessing metrics tracking failed.")
    
    # Step 3: Data visualization generation
    if not args.skip_visualization:
        logger.logger.info("Running data visualization generation...")
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_data_pipeline.py")
        script_args = [
            "--faces-dir", faces_dir,
            "--flow-dir", flow_dir,
            "--metrics-dir", metrics_dir,
            "--output-dir", vis_dir
        ]
        
        if args.log_config:
            script_args.extend(["--log-config", args.log_config])
        
        success = run_analysis_script(script_path, script_args)
        
        if success:
            logger.logger.info("Data visualization generation completed successfully.")
        else:
            logger.logger.error("Data visualization generation failed.")
    
    # Summarize analysis
    logger.logger.info("Data pipeline analysis completed.")
    logger.logger.info(f"Results saved to {output_dir}")
    
    # Generate summary output locations
    summary_locations = {
        "Dataset Statistics": os.path.join(output_dir, "dataset_statistics.json"),
        "Preprocessing Summary": os.path.join(output_dir, "preprocessing_summary.json"),
        "Visualizations Directory": vis_dir
    }
    
    logger.logger.info("Summary of output locations:")
    for name, location in summary_locations.items():
        if os.path.exists(location):
            logger.logger.info(f"  - {name}: {location}")

if __name__ == "__main__":
    main()
