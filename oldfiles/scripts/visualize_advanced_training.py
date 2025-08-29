"""
Advanced training visualization for Cross-Attention Personality Model (Phase 4.3)

This script generates comprehensive visualizations for training metrics,
including:
- Training/validation curves
- Learning rate changes over time
- Per-batch performance analysis
- Training time analysis
- Per-trait performance trends

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/visualize_advanced_training.py --metrics logs/batch_metrics/metrics_history.json --time_logs logs/time_tracking/time_logs.json --lr_logs logs/lr_tracking/lr_logs.json --output results/advanced_visualizations
    }
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from utils.logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Create Advanced Training Visualizations")
    parser.add_argument('--metrics', type=str, required=True, help='Path to metrics_history.json')
    parser.add_argument('--time_logs', type=str, required=True, help='Path to time_logs.json')
    parser.add_argument('--lr_logs', type=str, required=True, help='Path to lr_logs.json')
    parser.add_argument('--output', type=str, default='results/advanced_visualizations', help='Output directory for visualizations')
    return parser.parse_args()

def load_json_file(filepath):
    """
    Load a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_learning_rate_changes(lr_logs, output_dir):
    """
    Plot learning rate changes over epochs.
    
    Args:
        lr_logs (dict): Learning rate tracking logs
        output_dir (str): Directory to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the learning rate
    epochs = lr_logs['epoch']
    lr_values = lr_logs['learning_rate']
    is_reduction = lr_logs['is_reduction']
    
    plt.plot(epochs, lr_values, marker='o', linestyle='-', color='blue', label='Learning Rate')
    
    # Highlight the points where the learning rate was reduced
    reduction_epochs = [epochs[i] for i, reduced in enumerate(is_reduction) if reduced]
    reduction_lr = [lr_values[i] for i, reduced in enumerate(is_reduction) if reduced]
    
    if reduction_epochs:
        plt.scatter(reduction_epochs, reduction_lr, color='red', s=100, zorder=5, 
                   label='Learning Rate Reduction')
    
    # Set plot properties
    plt.title('Learning Rate Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Log scale to better visualize small changes
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotations for initial and final learning rates
    if lr_values:
        plt.annotate(f'Initial LR: {lr_values[0]:.6f}', 
                    xy=(epochs[0], lr_values[0]), 
                    xytext=(epochs[0]+1, lr_values[0]*1.2),
                    arrowprops=dict(arrowstyle="->", color='black', alpha=0.6))
        
        plt.annotate(f'Final LR: {lr_values[-1]:.6f}', 
                    xy=(epochs[-1], lr_values[-1]), 
                    xytext=(epochs[-1]-5, lr_values[-1]*1.2),
                    arrowprops=dict(arrowstyle="->", color='black', alpha=0.6))
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_changes.png'), dpi=300)
    plt.close()

def plot_batch_performance(metrics_history, output_dir):
    """
    Plot per-batch performance metrics.
    
    Args:
        metrics_history (dict): Metrics history dictionary
        output_dir (str): Directory to save the visualization
    """
    batch_data = metrics_history['batch']
    
    if not batch_data['batch_numbers']:
        print("No batch-level data available for visualization")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot batch loss
    ax1.plot(batch_data['batch_numbers'], batch_data['loss'], marker='.', alpha=0.6, label='Loss')
    ax1.set_title('Batch-Level Loss')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot batch learning rate
    ax2.plot(batch_data['batch_numbers'], batch_data['learning_rate'], marker='.', 
             alpha=0.6, color='orange', label='Learning Rate')
    ax2.set_title('Batch-Level Learning Rate')
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_performance.png'), dpi=300)
    plt.close()
    
    # Plot additional batch metrics if available
    if batch_data['metrics']:
        # Determine how many subplots we need
        num_metrics = len(batch_data['metrics'])
        fig = plt.figure(figsize=(14, 4 * num_metrics))
        gs = GridSpec(num_metrics, 1, figure=fig)
        
        for i, (metric_name, metric_values) in enumerate(batch_data['metrics'].items()):
            if metric_values:  # Check if there are values to plot
                ax = fig.add_subplot(gs[i, 0])
                ax.plot(batch_data['batch_numbers'], metric_values, marker='.', alpha=0.6)
                ax.set_title(f'Batch-Level {metric_name.upper()}')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
                
                if i == num_metrics - 1:  # Only add x-label for the last subplot
                    ax.set_xlabel('Batch Number')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_metrics.png'), dpi=300)
        plt.close()

def plot_training_validation_curves(metrics_history, output_dir):
    """
    Plot training and validation curves.
    
    Args:
        metrics_history (dict): Metrics history dictionary
        output_dir (str): Directory to save the visualization
    """
    epoch_data = metrics_history['epoch']
    
    if not epoch_data['epoch_numbers']:
        print("No epoch-level data available for visualization")
        return
    
    # Create figure for loss
    plt.figure(figsize=(12, 6))
    
    # Plot loss curves
    plt.plot(epoch_data['epoch_numbers'], epoch_data['loss'], 
             marker='o', linestyle='-', label='Training Loss')
    
    if epoch_data['val_loss']:
        plt.plot(epoch_data['epoch_numbers'], epoch_data['val_loss'], 
                 marker='o', linestyle='-', label='Validation Loss')
    
    # Set plot properties
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'train_val_loss.png'), dpi=300)
    plt.close()
    
    # Plot metrics
    for metric_name in epoch_data['metrics']:
        if metric_name.startswith('r_squared_') or metric_name.startswith('mae_') or metric_name.startswith('mse_'):
            continue  # Skip per-trait metrics, we'll plot them separately
        
        plt.figure(figsize=(12, 6))
        
        # Plot training metric
        plt.plot(epoch_data['epoch_numbers'], epoch_data['metrics'][metric_name], 
                 marker='o', linestyle='-', label=f'Training {metric_name}')
        
        # Plot validation metric if available
        if metric_name in epoch_data['val_metrics'] and epoch_data['val_metrics'][metric_name]:
            plt.plot(epoch_data['epoch_numbers'], epoch_data['val_metrics'][metric_name], 
                     marker='o', linestyle='-', label=f'Validation {metric_name}')
        
        # Set plot properties
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'train_val_{metric_name}.png'), dpi=300)
        plt.close()

def plot_per_trait_performance(metrics_history, output_dir):
    """
    Plot per-trait performance metrics.
    
    Args:
        metrics_history (dict): Metrics history dictionary
        output_dir (str): Directory to save the visualization
    """
    epoch_data = metrics_history['epoch']
    
    if not epoch_data['epoch_numbers']:
        print("No epoch-level data available for visualization")
        return
    
    # Group metrics by trait
    trait_metrics = {
        'O': {}, 'C': {}, 'E': {}, 'A': {}, 'N': {}
    }
    
    trait_full_names = {
        'O': 'Openness', 
        'C': 'Conscientiousness', 
        'E': 'Extraversion', 
        'A': 'Agreeableness', 
        'N': 'Neuroticism'
    }
    
    # Collect per-trait metrics
    for metric_name in epoch_data['metrics']:
        for trait in trait_metrics:
            if metric_name.endswith(f'_{trait}'):
                base_metric = metric_name.replace(f'_{trait}', '')
                if base_metric not in trait_metrics[trait]:
                    trait_metrics[trait][base_metric] = {}
                
                trait_metrics[trait][base_metric]['train'] = epoch_data['metrics'][metric_name]
                
                # Add validation metric if available
                if metric_name in epoch_data['val_metrics']:
                    trait_metrics[trait][base_metric]['val'] = epoch_data['val_metrics'][metric_name]
    
    # Plot per-trait metrics
    for trait, metrics in trait_metrics.items():
        if not metrics:  # Skip if no metrics for this trait
            continue
        
        # Create figure with multiple subplots
        num_metrics = len(metrics)
        if num_metrics == 0:
            continue
            
        fig = plt.figure(figsize=(14, 5 * num_metrics))
        gs = GridSpec(num_metrics, 1, figure=fig)
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = fig.add_subplot(gs[i, 0])
            
            # Plot training metric
            if 'train' in metric_data:
                ax.plot(epoch_data['epoch_numbers'], metric_data['train'], 
                        marker='o', linestyle='-', label=f'Training {metric_name}')
            
            # Plot validation metric
            if 'val' in metric_data:
                ax.plot(epoch_data['epoch_numbers'], metric_data['val'], 
                        marker='o', linestyle='-', label=f'Validation {metric_name}')
            
            # Set plot properties
            ax.set_title(f'{trait_full_names[trait]} - {metric_name}')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if i == num_metrics - 1:  # Only add x-label for the last subplot
                ax.set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trait_{trait_full_names[trait]}_metrics.png'), dpi=300)
        plt.close()

def plot_training_time_analysis(time_logs, output_dir):
    """
    Visualize training time data.
    
    Args:
        time_logs (dict): Time tracking logs
        output_dir (str): Directory to save the visualization
    """
    if not time_logs['epoch_times']:
        print("No time tracking data available for visualization")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot epoch times
    ax1.bar(time_logs['epoch_numbers'], time_logs['epoch_times'], alpha=0.7)
    ax1.set_title('Training Time per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Add mean time line
    mean_epoch_time = np.mean(time_logs['epoch_times'])
    ax1.axhline(mean_epoch_time, color='r', linestyle='--', 
                label=f'Mean: {mean_epoch_time:.2f}s')
    ax1.legend()
    
    # Plot batch times if available
    if time_logs['batch_times']:
        ax2.scatter(time_logs['batch_numbers'], time_logs['batch_times'], 
                   alpha=0.5, marker='.')
        ax2.set_title('Training Time per Batch')
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Add mean time line
        mean_batch_time = np.mean(time_logs['batch_times'])
        ax2.axhline(mean_batch_time, color='r', linestyle='--', 
                    label=f'Mean: {mean_batch_time:.4f}s')
        ax2.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_analysis.png'), dpi=300)
    plt.close()
    
    # Create a summary plot for total training time
    if 'total_training_time' in time_logs:
        plt.figure(figsize=(8, 6))
        
        # Convert to hours, minutes, seconds
        total_seconds = time_logs['total_training_time']
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Create a horizontal bar for total time
        plt.barh(['Total Training Time'], [total_seconds], alpha=0.7)
        
        # Add text label
        plt.text(total_seconds/2, 0, 
                 f"{int(hours)}h {int(minutes)}m {int(seconds):.1f}s",
                 ha='center', va='center', color='white', fontweight='bold')
        
        # Set plot properties
        plt.title('Total Training Time')
        plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_training_time.png'), dpi=300)
        plt.close()

def create_correlation_matrix(metrics_history, output_dir):
    """
    Create a correlation matrix of different metrics.
    
    Args:
        metrics_history (dict): Metrics history dictionary
        output_dir (str): Directory to save the visualization
    """
    epoch_data = metrics_history['epoch']
    
    if not epoch_data['epoch_numbers']:
        print("No epoch-level data available for correlation matrix")
        return
    
    # Collect metrics for correlation analysis
    corr_data = {}
    
    # Add loss
    corr_data['loss'] = epoch_data['loss']
    
    # Add validation loss if available
    if epoch_data['val_loss']:
        corr_data['val_loss'] = epoch_data['val_loss']
    
    # Add learning rate
    corr_data['learning_rate'] = epoch_data['learning_rate']
    
    # Add metrics
    for metric_name, values in epoch_data['metrics'].items():
        corr_data[metric_name] = values
    
    # Add validation metrics
    for metric_name, values in epoch_data['val_metrics'].items():
        corr_data[f'val_{metric_name}'] = values
    
    # Convert to dataframe for correlation analysis
    import pandas as pd
    metrics_df = pd.DataFrame(corr_data)
    
    # Compute correlation matrix
    corr_matrix = metrics_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Training Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation_matrix.png'), dpi=300)
    plt.close()

def create_combined_training_dashboard(metrics_history, time_logs, lr_logs, output_dir):
    """
    Create a comprehensive training dashboard.
    
    Args:
        metrics_history (dict): Metrics history dictionary
        time_logs (dict): Time tracking logs
        lr_logs (dict): Learning rate logs
        output_dir (str): Directory to save the visualization
    """
    if not metrics_history['epoch']['epoch_numbers']:
        print("No epoch-level data available for dashboard")
        return
    
    # Create a large figure with GridSpec for layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig)
    
    # 1. Training/Validation Loss (top left)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(metrics_history['epoch']['epoch_numbers'], metrics_history['epoch']['loss'], 
                marker='o', linestyle='-', label='Training Loss')
    
    if metrics_history['epoch']['val_loss']:
        ax_loss.plot(metrics_history['epoch']['epoch_numbers'], metrics_history['epoch']['val_loss'], 
                    marker='o', linestyle='-', label='Validation Loss')
    
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    
    # 2. Learning Rate Changes (top middle)
    ax_lr = fig.add_subplot(gs[0, 1])
    ax_lr.plot(lr_logs['epoch'], lr_logs['learning_rate'], marker='o', linestyle='-', color='blue')
    
    # Highlight points where learning rate was reduced
    reduction_epochs = [lr_logs['epoch'][i] for i, reduced in enumerate(lr_logs['is_reduction']) if reduced]
    reduction_lr = [lr_logs['learning_rate'][i] for i, reduced in enumerate(lr_logs['is_reduction']) if reduced]
    
    if reduction_epochs:
        ax_lr.scatter(reduction_epochs, reduction_lr, color='red', s=100, zorder=5)
    
    ax_lr.set_title('Learning Rate Changes')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_yscale('log')
    ax_lr.grid(True, alpha=0.3)
    
    # 3. Training Time per Epoch (top right)
    ax_time = fig.add_subplot(gs[0, 2])
    ax_time.bar(time_logs['epoch_numbers'], time_logs['epoch_times'], alpha=0.7)
    ax_time.set_title('Training Time per Epoch')
    ax_time.set_xlabel('Epoch')
    ax_time.set_ylabel('Time (seconds)')
    ax_time.grid(True, alpha=0.3)
    
    # 4. R-Squared per Trait (middle row)
    ax_r2 = fig.add_subplot(gs[1, :])
    
    # Find all R-squared metrics for each trait
    trait_labels = ['O', 'C', 'E', 'A', 'N']
    trait_colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, trait in enumerate(trait_labels):
        r2_key = f'r_squared_{trait}'
        if r2_key in metrics_history['epoch']['metrics']:
            ax_r2.plot(metrics_history['epoch']['epoch_numbers'], 
                      metrics_history['epoch']['metrics'][r2_key], 
                      marker='o', linestyle='-', color=trait_colors[i], 
                      label=f'{trait} (Training)')
        
        # Add validation if available
        if r2_key in metrics_history['epoch']['val_metrics']:
            ax_r2.plot(metrics_history['epoch']['epoch_numbers'], 
                      metrics_history['epoch']['val_metrics'][r2_key], 
                      marker='s', linestyle='--', color=trait_colors[i], 
                      label=f'{trait} (Validation)')
    
    ax_r2.set_title('R-Squared per Trait')
    ax_r2.set_xlabel('Epoch')
    ax_r2.set_ylabel('R-Squared')
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 5. MAE per Trait (bottom left)
    ax_mae = fig.add_subplot(gs[2, :])
    
    for i, trait in enumerate(trait_labels):
        mae_key = f'mae_{trait}'
        if mae_key in metrics_history['epoch']['metrics']:
            ax_mae.plot(metrics_history['epoch']['epoch_numbers'], 
                       metrics_history['epoch']['metrics'][mae_key], 
                       marker='o', linestyle='-', color=trait_colors[i], 
                       label=f'{trait} (Training)')
        
        # Add validation if available
        if mae_key in metrics_history['epoch']['val_metrics']:
            ax_mae.plot(metrics_history['epoch']['epoch_numbers'], 
                       metrics_history['epoch']['val_metrics'][mae_key], 
                       marker='s', linestyle='--', color=trait_colors[i], 
                       label=f'{trait} (Validation)')
    
    ax_mae.set_title('MAE per Trait')
    ax_mae.set_xlabel('Epoch')
    ax_mae.set_ylabel('MAE')
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 6. MSE per Trait (bottom right)
    ax_mse = fig.add_subplot(gs[3, :])
    
    for i, trait in enumerate(trait_labels):
        mse_key = f'mse_{trait}'
        if mse_key in metrics_history['epoch']['metrics']:
            ax_mse.plot(metrics_history['epoch']['epoch_numbers'], 
                       metrics_history['epoch']['metrics'][mse_key], 
                       marker='o', linestyle='-', color=trait_colors[i], 
                       label=f'{trait} (Training)')
        
        # Add validation if available
        if mse_key in metrics_history['epoch']['val_metrics']:
            ax_mse.plot(metrics_history['epoch']['epoch_numbers'], 
                       metrics_history['epoch']['val_metrics'][mse_key], 
                       marker='s', linestyle='--', color=trait_colors[i], 
                       label=f'{trait} (Validation)')
    
    ax_mse.set_title('MSE per Trait')
    ax_mse.set_xlabel('Epoch')
    ax_mse.set_ylabel('MSE')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Final adjustments and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    logger = get_logger(experiment_name="visualize_advanced_training")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.logger.info("Loading metrics and log data...")
    
    try:
        metrics_history = load_json_file(args.metrics)
        time_logs = load_json_file(args.time_logs)
        lr_logs = load_json_file(args.lr_logs)
        
        logger.logger.info("Data loaded successfully")
    except Exception as e:
        logger.logger.error(f"Error loading data: {e}")
        return
    
    # Generate visualizations
    logger.logger.info("Generating learning rate change visualization...")
    plot_learning_rate_changes(lr_logs, args.output)
    
    logger.logger.info("Generating batch performance visualization...")
    plot_batch_performance(metrics_history, args.output)
    
    logger.logger.info("Generating training/validation curves...")
    plot_training_validation_curves(metrics_history, args.output)
    
    logger.logger.info("Generating per-trait performance visualization...")
    plot_per_trait_performance(metrics_history, args.output)
    
    logger.logger.info("Generating training time analysis...")
    plot_training_time_analysis(time_logs, args.output)
    
    logger.logger.info("Generating metrics correlation matrix...")
    try:
        create_correlation_matrix(metrics_history, args.output)
    except Exception as e:
        logger.logger.warning(f"Could not create correlation matrix: {e}")
    
    logger.logger.info("Generating combined training dashboard...")
    create_combined_training_dashboard(metrics_history, time_logs, lr_logs, args.output)
    
    logger.logger.info(f"All visualizations saved to {args.output}")
    logger.close()

if __name__ == "__main__":
    main()
