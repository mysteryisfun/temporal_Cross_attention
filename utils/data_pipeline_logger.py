import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import wandb
from datetime import datetime
import cv2
from .logger import ResearchLogger
from .json_utils import convert_numpy_types

class DataPipelineLogger(ResearchLogger):
    """
    Logger specifically designed for tracking data pipeline metrics and visualizations
    for the Cross-Attention CNN Research project.
    """
    
    def __init__(self, config_path=None, experiment_name=None, output_dir="results"):
        """
        Initialize the data pipeline logger.
        
        Args:
            config_path (str): Path to the logging configuration YAML
            experiment_name (str): Name for this experiment run
            output_dir (str): Base directory for output files
        """
        super().__init__(config_path, experiment_name)
        self.output_dir = output_dir
        
        # Create specific directories for data pipeline logging
        self.stats_dir = os.path.join(self.output_dir, "dataset_statistics")
        self.metrics_dir = os.path.join(self.output_dir, "preprocessing_metrics")
        self.vis_dir = os.path.join(self.output_dir, "visualizations", "data_pipeline")
        
        # Create directories
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
        self.logger.info(f"Data pipeline logger initialized. Output directory: {self.output_dir}")
        self.log_start_time = datetime.now()
        
        # Initialize metrics tracking
        self.preprocessing_metrics = {
            'face_detection': {},
            'frame_extraction': {},
            'optical_flow': {},
            'data_augmentation': {}
        }
    
    def log_dataset_statistics(self, dataset_name, statistics, save_json=True):
        """
        Log comprehensive dataset statistics.
        
        Args:
            dataset_name (str): Name of the dataset
            statistics (dict): Dictionary containing dataset statistics
            save_json (bool): Whether to save statistics to a JSON file
        """
        # Log to console/file
        self.logger.info(f"Dataset Statistics for {dataset_name}:")
        for category, stats in statistics.items():
            self.logger.info(f"  {category}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    self.logger.info(f"    {key}: {value}")
            else:
                self.logger.info(f"    {stats}")
        
        # Log to tracking systems
        if self.tensorboard_writer:
            # Flatten the nested dictionary for TensorBoard
            flat_stats = {}
            for category, stats in statistics.items():
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            flat_stats[f"{category}/{key}"] = value
            
            self.log_metrics(flat_stats)            # Save to JSON file
            if save_json:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{dataset_name}_statistics_{timestamp}.json"
                filepath = os.path.join(self.stats_dir, filename)
                
                # Convert numpy types to Python native types
                processed_stats = convert_numpy_types(statistics)
                
                with open(filepath, 'w') as f:
                    json.dump(processed_stats, f, indent=2)
                
                self.logger.info(f"Dataset statistics saved to {filepath}")
                
                # Save a copy to the main results directory for quick access
                main_filepath = os.path.join(self.output_dir, "dataset_statistics.json")
                with open(main_filepath, 'w') as f:
                    json.dump(processed_stats, f, indent=2)
    
    def log_preprocessing_metrics(self, process_name, metrics, save_csv=True):
        """
        Log metrics from preprocessing steps.
        
        Args:
            process_name (str): Name of the preprocessing step
            metrics (dict): Dictionary containing preprocessing metrics
            save_csv (bool): Whether to save metrics to a CSV file
        """
        # Update internal metrics
        if process_name in self.preprocessing_metrics:
            self.preprocessing_metrics[process_name].update(metrics)
        else:
            self.preprocessing_metrics[process_name] = metrics
        
        # Log to console/file
        self.logger.info(f"Preprocessing metrics for {process_name}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to tracking systems
        if self.tensorboard_writer:
            prefixed_metrics = {f"preprocessing/{process_name}/{k}": v 
                               for k, v in metrics.items() 
                               if isinstance(v, (int, float))}
            self.log_metrics(prefixed_metrics)
        
        # Save to CSV file
        if save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{process_name}_metrics_{timestamp}.csv"
            filepath = os.path.join(self.metrics_dir, filename)
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_df['timestamp'] = timestamp
            metrics_df['process'] = process_name
              # Save to CSV (append if exists)
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            
            metrics_df.to_csv(filepath, index=False)
            self.logger.info(f"Preprocessing metrics saved to {filepath}")
    
    def save_preprocessing_summary(self):
        """Save a summary of all preprocessing metrics"""
        filepath = os.path.join(self.output_dir, "preprocessing_summary.json")
        
        # Add summary information
        summary = {
            'preprocessing_metrics': self.preprocessing_metrics,
            'time_elapsed': str(datetime.now() - self.log_start_time),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Import JSON utility to handle numpy types
        from utils.json_utils import convert_numpy_types
        processed_summary = convert_numpy_types(summary)
        
        with open(filepath, 'w') as f:
            json.dump(processed_summary, f, indent=2)
        
        self.logger.info(f"Preprocessing summary saved to {filepath}")
    
    def visualize_face_detection_results(self, results, output_path=None):
        """
        Visualize face detection results.
        
        Args:
            results (dict): Face detection results
            output_path (str, optional): Output file path
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.vis_dir, f"face_detection_results_{timestamp}.png")
            
        # Extract metrics
        video_ids = list(results.keys())
        detection_rates = [results[vid].get('detection_rate', 0) for vid in video_ids]
        
        # Create bar chart
        plt.figure(figsize=(15, 8))
        
        # If there are many videos, use a subset
        if len(video_ids) > 20:
            sorted_indices = np.argsort(detection_rates)
            # Get 10 best and 10 worst
            indices_to_show = np.concatenate([sorted_indices[:10], sorted_indices[-10:]])
            video_ids = [video_ids[i] for i in indices_to_show]
            detection_rates = [detection_rates[i] for i in indices_to_show]
        
        plt.bar(range(len(video_ids)), detection_rates, align='center')
        plt.xticks(range(len(video_ids)), video_ids, rotation=90)
        plt.xlabel('Video ID')
        plt.ylabel('Face Detection Rate')
        plt.title('Face Detection Rate by Video')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Face detection visualization saved to {output_path}")
        
        # Log to W&B if available
        if self.wandb_run and hasattr(self.wandb_run, 'log'):
            self.wandb_run.log({"face_detection_results": wandb.Image(output_path)})
    
    def visualize_optical_flow_quality(self, flow_stats, output_path=None):
        """
        Visualize optical flow quality metrics.
        
        Args:
            flow_stats (dict): Optical flow statistics
            output_path (str, optional): Output file path
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.vis_dir, f"optical_flow_quality_{timestamp}.png")
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        video_ids = list(flow_stats.keys())
        avg_magnitudes = [flow_stats[vid].get('average_magnitude', 0) for vid in video_ids]
        max_magnitudes = [flow_stats[vid].get('max_magnitude', 0) for vid in video_ids]
        std_magnitudes = [flow_stats[vid].get('std_magnitude', 0) for vid in video_ids]
        
        # If there are many videos, use a subset
        if len(video_ids) > 20:
            # Select 20 random videos
            indices = np.random.choice(len(video_ids), 20, replace=False)
            video_ids = [video_ids[i] for i in indices]
            avg_magnitudes = [avg_magnitudes[i] for i in indices]
            max_magnitudes = [max_magnitudes[i] for i in indices]
            std_magnitudes = [std_magnitudes[i] for i in indices]
        
        # Plot average magnitudes
        axs[0, 0].bar(range(len(video_ids)), avg_magnitudes)
        axs[0, 0].set_title('Average Flow Magnitude')
        axs[0, 0].set_xticks([])
        
        # Plot max magnitudes
        axs[0, 1].bar(range(len(video_ids)), max_magnitudes)
        axs[0, 1].set_title('Maximum Flow Magnitude')
        axs[0, 1].set_xticks([])
        
        # Plot standard deviations
        axs[1, 0].bar(range(len(video_ids)), std_magnitudes)
        axs[1, 0].set_title('Flow Magnitude Std Deviation')
        axs[1, 0].set_xticks([])
        
        # Plot histogram of all average magnitudes
        axs[1, 1].hist(avg_magnitudes, bins=20)
        axs[1, 1].set_title('Distribution of Average Flow Magnitudes')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Optical flow quality visualization saved to {output_path}")
        
        # Log to W&B if available
        if self.wandb_run and hasattr(self.wandb_run, 'log'):
            self.wandb_run.log({"optical_flow_quality": wandb.Image(output_path)})
    
    def visualize_dataset_samples(self, faces, optical_flows, n_samples=5, output_path=None):
        """
        Visualize sample pairs of face images and their corresponding optical flow.
        
        Args:
            faces (list): List of face image file paths
            optical_flows (list): List of optical flow file paths
            n_samples (int): Number of samples to visualize
            output_path (str, optional): Output file path
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.vis_dir, f"dataset_samples_{timestamp}.png")
        
        # Select random samples
        if len(faces) > n_samples:
            indices = np.random.choice(len(faces), n_samples, replace=False)
            face_samples = [faces[i] for i in indices]
            flow_samples = [optical_flows[i] for i in indices]
        else:
            face_samples = faces
            flow_samples = optical_flows
        
        # Create figure
        fig, axs = plt.subplots(n_samples, 2, figsize=(10, 3*n_samples))
        
        for i in range(len(face_samples)):
            # Load face image
            face_img = cv2.imread(face_samples[i])
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Load optical flow image
            flow_img = cv2.imread(flow_samples[i])
            flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
            
            # Plot images
            if n_samples > 1:
                axs[i, 0].imshow(face_img)
                axs[i, 0].set_title(f'Face {i+1}')
                axs[i, 0].axis('off')
                
                axs[i, 1].imshow(flow_img)
                axs[i, 1].set_title(f'Optical Flow {i+1}')
                axs[i, 1].axis('off')
            else:
                axs[0].imshow(face_img)
                axs[0].set_title(f'Face {i+1}')
                axs[0].axis('off')
                
                axs[1].imshow(flow_img)
                axs[1].set_title(f'Optical Flow {i+1}')
                axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Dataset sample visualization saved to {output_path}")
        
        # Log to W&B if available
        if self.wandb_run and hasattr(self.wandb_run, 'log'):
            self.wandb_run.log({"dataset_samples": wandb.Image(output_path)})
