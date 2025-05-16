#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: optical_flow_computer.py
# Description: Optical flow computation module for Cross-Attention CNN Research

import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import json

# Set up logger
def setup_logger(log_file=None):
    """
    Set up the logger for optical flow computation process.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs only to console.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger("OpticalFlowComputer")
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


class OpticalFlowComputer:
    """
    A class for computing optical flow between consecutive frames.
    """
    
    def __init__(self, method='farneback', params=None):
        """
        Initialize the OpticalFlowComputer with specified method and parameters.
        
        Args:
            method (str): Optical flow method to use ('farneback' or 'tvl1').
            params (dict, optional): Parameters for the optical flow algorithm.
        """
        self.method = method.lower()
        
        # Set default parameters based on method
        if self.method == 'farneback':
            self.default_params = {
                'pyr_scale': 0.5,      # Scale for pyramid structures
                'levels': 3,            # Number of pyramid layers
                'winsize': 15,          # Window size
                'iterations': 3,        # Number of iterations
                'poly_n': 5,            # Polynomial degree used to approximate pixel neighborhoods
                'poly_sigma': 1.2,      # Standard deviation for Gaussian used to smooth derivatives
                'flags': 0              # Additional flags
            }
        elif self.method == 'tvl1':
            # Check if optflow module is available
            if not hasattr(cv2, 'optflow'):
                raise ImportError("The cv2.optflow module is not available. "
                                 "Please install OpenCV with extra modules to use TVL1 method.")
            
            # Create DualTVL1 optical flow object with default parameters
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            self.default_params = {
                'tau': 0.25,            # Time step size
                'lambda_': 0.15,        # Weight parameter for data term
                'theta': 0.3,           # Weight parameter for smoothness term
                'nscales': 5,           # Number of scales
                'warps': 5,             # Number of warps per scale
                'epsilon': 0.01,        # Stopping criterion threshold
                'iterations': 300       # Maximum number of iterations
            }
        else:
            raise ValueError(f"Unsupported optical flow method: {method}. Use 'farneback' or 'tvl1'.")
        
        # Update default parameters with provided parameters
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
        
        # If TVL1 method, update the object parameters
        if self.method == 'tvl1':
            for key, value in self.params.items():
                if hasattr(self.tvl1, key):
                    setattr(self.tvl1, key, value)
    
    def compute_flow(self, prev_frame, curr_frame):
        """
        Compute optical flow between two consecutive frames.
        
        Args:
            prev_frame (numpy.ndarray): Previous frame (grayscale).
            curr_frame (numpy.ndarray): Current frame (grayscale).
            
        Returns:
            numpy.ndarray: Optical flow field (u, v).
        """
        # Check if frames are grayscale, convert if not
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Compute optical flow
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, 
                curr_gray, 
                None,
                self.params['pyr_scale'],
                self.params['levels'],
                self.params['winsize'],
                self.params['iterations'],
                self.params['poly_n'],
                self.params['poly_sigma'],
                self.params['flags']
            )
        elif self.method == 'tvl1':
            flow = self.tvl1.calc(prev_gray, curr_gray, None)
        
        return flow
    
    def flow_to_rgb(self, flow, max_flow=None):
        """
        Convert optical flow field to RGB image for visualization.
        
        Args:
            flow (numpy.ndarray): Optical flow field (u, v).
            max_flow (float, optional): Maximum flow magnitude for normalization.
                If None, it will be calculated from the flow field.
                
        Returns:
            numpy.ndarray: RGB representation of optical flow.
        """
        # Calculate flow magnitude and angle
        u, v = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) + np.pi
        
        # Normalize magnitude
        if max_flow is None:
            max_flow = np.max(magnitude)
        
        if max_flow > 0:
            magnitude = np.clip(magnitude / max_flow, 0, 1)
        
        # Convert to HSV
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / (2 * np.pi)  # Hue based on flow direction
        hsv[..., 1] = 255                        # Full saturation
        hsv[..., 2] = np.clip(magnitude * 255, 0, 255).astype(np.uint8)  # Value based on flow magnitude
        
        # Convert HSV to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    
    def compute_sequence_flow(self, frames):
        """
        Compute optical flow for a sequence of frames.
        
        Args:
            frames (list): List of frames (numpy.ndarray).
            
        Returns:
            list: List of optical flow fields between consecutive frames.
        """
        if len(frames) < 2:
            return []
        
        flows = []
        for i in range(len(frames) - 1):
            flow = self.compute_flow(frames[i], frames[i+1])
            flows.append(flow)
        
        return flows
    
    def save_flow(self, flow, output_path, save_rgb=True, save_raw=False, max_flow=None):
        """
        Save optical flow to disk.
        
        Args:
            flow (numpy.ndarray): Optical flow field (u, v).
            output_path (str): Base path for saving the flow.
            save_rgb (bool): Whether to save RGB visualization.
            save_raw (bool): Whether to save raw flow data as .npy file.
            max_flow (float, optional): Maximum flow magnitude for normalization.
                
        Returns:
            dict: Paths where the flow was saved.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        saved_paths = {}
        
        # Save RGB visualization
        if save_rgb:
            rgb_path = f"{output_path}_flow_rgb.jpg"
            rgb = self.flow_to_rgb(flow, max_flow)
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved_paths['rgb'] = rgb_path
        
        # Save raw flow data
        if save_raw:
            npy_path = f"{output_path}_flow_raw.npy"
            np.save(npy_path, flow)
            saved_paths['raw'] = npy_path
        
        return saved_paths
    
    def extract_frames_from_video(self, video_path, num_frames=16):
        """
        Extract a fixed number of evenly spaced frames from a video file.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to extract.
        
        Returns:
            list: List of frames as numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            raise ValueError(f"Video has only {total_frames} frames, but {num_frames} requested.")
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frames.append(frame)
        cap.release()
        if len(frames) != num_frames:
            raise ValueError(f"Extracted {len(frames)} frames, expected {num_frames}.")
        return frames


def process_video_faces(faces_dir, output_dir, method='farneback', params=None, 
                      save_rgb=True, save_raw=False, logger=None):
    """
    Process face images from a video to compute optical flow.
    
    Args:
        faces_dir (str): Directory containing face images for a single video.
        output_dir (str): Directory to save optical flow results.
        method (str): Optical flow method to use ('farneback' or 'tvl1').
        params (dict, optional): Parameters for the optical flow algorithm.
        save_rgb (bool): Whether to save RGB visualization.
        save_raw (bool): Whether to save raw flow data.
        logger (logging.Logger, optional): Logger object.
        
    Returns:
        dict: Processing statistics.
    """
    if logger is None:
        logger = logging.getLogger("OpticalFlowComputer")
    
    stats = {
        'video_id': os.path.basename(faces_dir),
        'status': 'failure',
        'error': None,
        'num_frames': 0,
        'num_flows': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Get all face images
        face_files = sorted([
            os.path.join(faces_dir, f) for f in os.listdir(faces_dir)
            if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')
        ])
        
        if len(face_files) < 2:
            stats['error'] = f"Not enough faces to compute flow (found {len(face_files)})"
            stats['num_frames'] = len(face_files)
            return stats
        
        # Create output directory
        video_id = os.path.basename(faces_dir)
        video_output_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Initialize optical flow computer
        flow_computer = OpticalFlowComputer(method=method, params=params)
        
        # Load frames
        frames = []
        for face_file in face_files:
            frame = cv2.imread(face_file)
            if frame is not None:
                frames.append(frame)
        
        # Compute optical flow
        flows = []
        for i in range(len(frames) - 1):
            try:
                flow = flow_computer.compute_flow(frames[i], frames[i+1])
                flows.append(flow)
                
                # Save optical flow
                base_name = os.path.basename(face_files[i])
                base_path = os.path.join(video_output_dir, os.path.splitext(base_name)[0])
                flow_computer.save_flow(flow, base_path, save_rgb, save_raw)
            except Exception as e:
                logger.warning(f"Error computing flow for {face_files[i]}: {str(e)}")
        
        # Update statistics
        stats['status'] = 'success'
        stats['num_frames'] = len(frames)
        stats['num_flows'] = len(flows)
        
    except Exception as e:
        stats['error'] = str(e)
    
    stats['processing_time'] = time.time() - start_time
    
    return stats


def process_all_videos(faces_dir, output_dir, method='farneback', params=None, 
                     save_rgb=True, save_raw=False, num_workers=4, logger=None):
    """
    Process all videos in the faces directory to compute optical flow.
    
    Args:
        faces_dir (str): Directory containing face images for all videos.
        output_dir (str): Directory to save optical flow results.
        method (str): Optical flow method to use ('farneback' or 'tvl1').
        params (dict, optional): Parameters for the optical flow algorithm.
        save_rgb (bool): Whether to save RGB visualization.
        save_raw (bool): Whether to save raw flow data.
        num_workers (int): Number of parallel workers.
        logger (logging.Logger, optional): Logger object.
        
    Returns:
        dict: Processing statistics.
    """
    if logger is None:
        logger = logging.getLogger("OpticalFlowComputer")
    
    start_time = time.time()
    
    # Get all video directories
    video_dirs = []
    
    # Check if faces_dir is already a video directory (contains face images directly)
    face_files = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(face_files) > 0 and all(os.path.isfile(os.path.join(faces_dir, f)) for f in face_files):
        # If it's a video directory, add it directly
        video_dirs.append(faces_dir)
    else:
        # Otherwise, scan for subdirectories
        for item in os.listdir(faces_dir):
            item_path = os.path.join(faces_dir, item)
            if os.path.isdir(item_path):
                video_dirs.append(item_path)
    
    logger.info(f"Found {len(video_dirs)} video directories in {faces_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos in parallel
    successful_videos = 0
    failed_videos = 0
    total_flows = 0
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {}
        for video_dir in video_dirs:
            future = executor.submit(
                process_video_faces,
                video_dir,
                output_dir,
                method,
                params,
                save_rgb,
                save_raw,
                None  # Can't pass logger to subprocess
            )
            future_to_video[future] = video_dir
        
        for future in tqdm(as_completed(future_to_video), total=len(video_dirs), desc="Computing optical flow"):
            video_dir = future_to_video[future]
            video_id = os.path.basename(video_dir)
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    successful_videos += 1
                    total_flows += result['num_flows']
                    logger.info(f"Successfully computed {result['num_flows']} flows for {video_id}")
                else:
                    failed_videos += 1
                    logger.warning(f"Failed to compute flow for {video_id}: {result['error']}")
            except Exception as e:
                failed_videos += 1
                logger.error(f"Error processing {video_id}: {str(e)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Compute summary statistics
    total_videos = len(video_dirs)
    success_rate = successful_videos / total_videos * 100 if total_videos > 0 else 0
    
    # Log summary
    logger.info(f"Optical flow computation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"Successfully processed: {successful_videos} ({success_rate:.2f}%)")
    logger.info(f"Failed to process: {failed_videos}")
    logger.info(f"Total flows computed: {total_flows}")
    
    return {
        'total_videos': total_videos,
        'successful_videos': successful_videos,
        'failed_videos': failed_videos,
        'total_flows': total_flows,
        'success_rate': success_rate,
        'processing_time': elapsed_time,
        'results': results
    }


def visualize_flow_quality(flow_dir, output_file, num_samples=5):
    """
    Generate visualization of optical flow quality.
    
    Args:
        flow_dir (str): Directory containing optical flow results.
        output_file (str): Path to save the visualization.
        num_samples (int): Number of sample videos to visualize.
        
    Returns:
        str: Path to the saved visualization file.
    """
    # Find all RGB flow visualizations
    flow_files = []
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.endswith('_flow_rgb.jpg'):
                flow_files.append(os.path.join(root, file))
    
    if not flow_files:
        return None
    
    # Randomly sample a subset of flows
    if len(flow_files) > num_samples * 3:  # Ensure we have enough samples
        np.random.shuffle(flow_files)
        flow_files = flow_files[:num_samples * 3]
    
    # Organize by video
    video_flows = {}
    for flow_file in flow_files:
        video_id = flow_file.split(os.sep)[-2]
        if video_id not in video_flows:
            video_flows[video_id] = []
        video_flows[video_id].append(flow_file)
    
    # Select top videos with most flows
    top_videos = sorted(video_flows.items(), key=lambda x: len(x[1]), reverse=True)[:num_samples]
    
    # Create visualization grid
    rows = len(top_videos)
    cols = 3  # Show 3 flows per video
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, (video_id, files) in enumerate(top_videos):
        for j in range(min(cols, len(files))):
            ax = plt.subplot(rows, cols, i * cols + j + 1)
            img = cv2.imread(files[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"{video_id} - Flow {j+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def analyze_flow_magnitude(flow_dir, output_file):
    """
    Analyze optical flow magnitude distribution.
    
    Args:
        flow_dir (str): Directory containing optical flow results.
        output_file (str): Path to save the analysis visualization.
        
    Returns:
        dict: Flow magnitude statistics.
    """
    # Find all raw flow files
    flow_files = []
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.endswith('_flow_raw.npy'):
                flow_files.append(os.path.join(root, file))
    
    if not flow_files:
        return None
    
    # Calculate flow magnitudes
    magnitudes = []
    for flow_file in tqdm(flow_files, desc="Analyzing flow magnitude"):
        try:
            flow = np.load(flow_file)
            u, v = flow[..., 0], flow[..., 1]
            magnitude = np.sqrt(u**2 + v**2)
            
            # Get statistics for this flow
            magnitudes.append({
                'mean': float(np.mean(magnitude)),
                'std': float(np.std(magnitude)),
                'max': float(np.max(magnitude)),
                'min': float(np.min(magnitude)),
                'median': float(np.median(magnitude)),
                'file': flow_file
            })
        except Exception as e:
            print(f"Error processing {flow_file}: {str(e)}")
    
    if not magnitudes:
        return None
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Mean flow magnitude histogram
    plt.subplot(2, 2, 1)
    mean_values = [m['mean'] for m in magnitudes]
    plt.hist(mean_values, bins=30, color='#2196F3', alpha=0.7)
    plt.xlabel('Mean Flow Magnitude')
    plt.ylabel('Number of Flows')
    plt.title('Distribution of Mean Flow Magnitude')
    
    # Plot 2: Max flow magnitude histogram
    plt.subplot(2, 2, 2)
    max_values = [m['max'] for m in magnitudes]
    plt.hist(max_values, bins=30, color='#F44336', alpha=0.7)
    plt.xlabel('Max Flow Magnitude')
    plt.ylabel('Number of Flows')
    plt.title('Distribution of Max Flow Magnitude')
    
    # Plot 3: Scatter plot of mean vs. max flow magnitude
    plt.subplot(2, 2, 3)
    plt.scatter(mean_values, max_values, alpha=0.5, c='#4CAF50')
    plt.xlabel('Mean Flow Magnitude')
    plt.ylabel('Max Flow Magnitude')
    plt.title('Mean vs. Max Flow Magnitude')
    
    # Plot 4: Statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats = {
        'mean': np.mean(mean_values),
        'std': np.std(mean_values),
        'min': np.min(mean_values),
        'max': np.max(mean_values),
        'median': np.median(mean_values)
    }
    
    stats_text = (
        f"Flow Magnitude Statistics\n"
        f"-----------------------\n\n"
        f"Number of flows: {len(magnitudes)}\n"
        f"Mean of means: {stats['mean']:.2f}\n"
        f"Std dev of means: {stats['std']:.2f}\n"
        f"Min of means: {stats['min']:.2f}\n"
        f"Max of means: {stats['max']:.2f}\n"
        f"Median of means: {stats['median']:.2f}\n\n"
        f"Average max magnitude: {np.mean(max_values):.2f}\n"
        f"Max recorded magnitude: {np.max(max_values):.2f}"
    )
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics as JSON
    json_file = os.path.splitext(output_file)[0] + '.json'
    with open(json_file, 'w') as f:
        json.dump({
            'overall_stats': stats,
            'flow_stats': magnitudes
        }, f, indent=2)
    
    return stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Optical flow computation")
    parser.add_argument('--faces-dir', type=str, required=True,
                        help="Directory containing face images")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save optical flow results")
    parser.add_argument('--method', type=str, default='farneback',
                        choices=['farneback', 'tvl1'],
                        help="Optical flow method to use")
    parser.add_argument('--save-rgb', action='store_true',
                        help="Save RGB visualization of optical flow")
    parser.add_argument('--save-raw', action='store_true',
                        help="Save raw optical flow data as .npy files")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument('--log-file', type=str, default=None,
                        help="Path to log file")
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_file)
    
    # Process videos
    results = process_all_videos(
        args.faces_dir,
        args.output_dir,
        args.method,
        None,
        args.save_rgb,
        args.save_raw,
        args.workers,
        logger
    )
    
    # Visualize results
    vis_path = os.path.join(args.output_dir, "flow_samples.png")
    visualize_flow_quality(args.output_dir, vis_path)
    
    # Analyze flow magnitude
    if args.save_raw:
        analysis_path = os.path.join(args.output_dir, "flow_magnitude_analysis.png")
        analyze_flow_magnitude(args.output_dir, analysis_path)
