#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: analyze_face_detection.py
# Description: Generate a comprehensive analysis of face detection results

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
from collections import defaultdict
import cv2

# Add the project root to the Python path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.face_detector import setup_logger

def collect_face_data(face_dir):
    """
    Collect face detection data from the processed faces directory.
    
    Args:
        face_dir (str): Directory containing processed faces
        
    Returns:
        dict: Face detection statistics
    """
    # Initialize counters
    stats = {
        'total_videos': 0,
        'total_faces': 0,
        'videos_with_faces': 0,
        'videos_without_faces': 0,
        'faces_per_video': defaultdict(int),
        'face_dimensions': [],
        'face_examples': {}
    }
    
    # Traverse face directory
    for video_id in os.listdir(face_dir):
        video_path = os.path.join(face_dir, video_id)
        if not os.path.isdir(video_path):
            continue
        
        stats['total_videos'] += 1
        
        # Count face images
        face_files = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('detection_results')]
        num_faces = len(face_files)
        stats['total_faces'] += num_faces
        
        if num_faces > 0:
            stats['videos_with_faces'] += 1
            stats['faces_per_video'][video_id] = num_faces
            
            # Sample a face for visualization
            if num_faces > 0 and len(stats['face_examples']) < 10:
                sample_face_path = os.path.join(video_path, face_files[0])
                face_img = cv2.imread(sample_face_path)
                if face_img is not None:
                    stats['face_examples'][video_id] = {
                        'path': sample_face_path,
                        'dimensions': face_img.shape[:2]  # Height, width
                    }
                    stats['face_dimensions'].append(face_img.shape[:2])
        else:
            stats['videos_without_faces'] += 1
    
    # Calculate additional statistics
    if stats['total_videos'] > 0:
        stats['face_detection_rate'] = stats['videos_with_faces'] / stats['total_videos'] * 100
    else:
        stats['face_detection_rate'] = 0
    
    if stats['videos_with_faces'] > 0:
        stats['avg_faces_per_video'] = stats['total_faces'] / stats['videos_with_faces']
    else:
        stats['avg_faces_per_video'] = 0
    
    return stats

def generate_visualization(stats, output_path):
    """
    Generate visualization of face detection statistics.
    
    Args:
        stats (dict): Face detection statistics
        output_path (str): Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # 1. Face detection rate pie chart
    plt.subplot(gs[0, 0])
    labels = ['Videos with faces', 'Videos without faces']
    sizes = [stats['videos_with_faces'], stats['videos_without_faces']]
    colors = ['#4CAF50', '#F44336']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Face Detection Rate')
    
    # 2. Faces per video histogram
    if stats['faces_per_video']:
        plt.subplot(gs[0, 1:])
        faces_per_video = list(stats['faces_per_video'].values())
        plt.hist(faces_per_video, bins=min(20, len(set(faces_per_video))), color='#2196F3', alpha=0.7)
        plt.xlabel('Number of Faces')
        plt.ylabel('Number of Videos')
        plt.title('Faces per Video Distribution')
        plt.grid(alpha=0.3)
    
    # 3. Face dimensions scatter plot
    if stats['face_dimensions']:
        plt.subplot(gs[1, :2])
        face_dims = np.array(stats['face_dimensions'])
        plt.scatter(face_dims[:, 1], face_dims[:, 0], alpha=0.5, c='#673AB7')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Face Dimensions')
        plt.grid(alpha=0.3)
        
        # Add annotations for min/max/avg dimensions
        if len(face_dims) > 0:
            h_min, w_min = face_dims.min(axis=0)
            h_max, w_max = face_dims.max(axis=0)
            h_avg, w_avg = face_dims.mean(axis=0)
            plt.annotate(f'Min: {w_min}x{h_min}', xy=(w_min, h_min), xytext=(w_min+10, h_min+10),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            plt.annotate(f'Max: {w_max}x{h_max}', xy=(w_max, h_max), xytext=(w_max-10, h_max-10),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            plt.annotate(f'Avg: {w_avg:.1f}x{h_avg:.1f}', xy=(w_avg, h_avg), xytext=(w_avg+10, h_avg-10),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 4. Summary statistics
    plt.subplot(gs[1, 2])
    plt.axis('off')
    summary_text = (
        f"Face Detection Summary\n"
        f"----------------------\n\n"
        f"Total videos processed: {stats['total_videos']}\n"
        f"Videos with faces: {stats['videos_with_faces']} ({stats['face_detection_rate']:.1f}%)\n"
        f"Videos without faces: {stats['videos_without_faces']}\n"
        f"Total faces detected: {stats['total_faces']}\n"
        f"Average faces per video: {stats['avg_faces_per_video']:.2f}\n"
    )
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
      # 5. Sample face examples
    if stats['face_examples']:
        plt.subplot(gs[2, :])
        plt.axis('off')
        plt.title('Sample Detected Faces')
        
        num_examples = min(3, len(stats['face_examples']))
        examples = list(stats['face_examples'].items())[:num_examples]
        
        for i, (video_id, example) in enumerate(examples):
            # Read image with OpenCV (BGR) and convert to RGB for matplotlib
            img = cv2.imread(example['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use figure-level add_subplot instead of gridspec indexing for face examples
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"{video_id}\n{img.shape[1]}x{img.shape[0]}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save individual face examples
            example_path = os.path.join(os.path.dirname(output_path), f"face_example_{i+1}_{video_id}.png")
            plt.savefig(example_path)
            plt.close()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Analyze face detection results")
    
    parser.add_argument('--face-dir', type=str, required=True,
                        help="Directory containing processed faces")
    parser.add_argument('--output-dir', type=str, default="./results/visualizations",
                        help="Directory to save analysis results")
    parser.add_argument('--output-name', type=str, default="face_detection_analysis.png",
                        help="Filename for the output visualization")
    parser.add_argument('--log-file', type=str,
                        help="Path to log file")
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    
    # Collect face detection data
    logger.info(f"Analyzing face detection results in {args.face_dir}")
    stats = collect_face_data(args.face_dir)
    
    # Generate visualization
    logger.info(f"Generating visualization at {output_path}")
    generate_visualization(stats, output_path)
    
    # Save statistics as JSON
    json_path = os.path.join(args.output_dir, 'face_detection_stats.json')
    with open(json_path, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        stats['faces_per_video'] = dict(stats['faces_per_video'])
        # Convert face dimensions to list of lists for JSON serialization
        stats['face_dimensions'] = [list(dim) for dim in stats['face_dimensions']]
        # Remove face examples with image data for JSON serialization
        clean_examples = {}
        for video_id, example in stats['face_examples'].items():
            clean_examples[video_id] = {
                'path': example['path'],
                'dimensions': list(example['dimensions'])
            }
        stats['face_examples'] = clean_examples
        
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {json_path}")
    logger.info("Face detection analysis completed.")

if __name__ == "__main__":
    main()
