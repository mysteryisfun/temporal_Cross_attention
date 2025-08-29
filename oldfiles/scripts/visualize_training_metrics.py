"""
Visualization of training metrics for Cross-Attention Personality Model (Phase 4.2)

This script loads training history from saved files and generates visualizations
showing the model's performance across training epochs, with particular focus on:
- Overall loss and metrics trends
- Per-trait performance (OCEAN)
- Learning rate changes
- Correlation between different metrics

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/visualize_training_metrics.py --history results/training_history.npy --output results/training_visualizations
    }
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Cross-Attention Personality Model Training Metrics")
    parser.add_argument('--history', type=str, required=True, help='Path to training history .npy file')
    parser.add_argument('--output', type=str, default='results/training_visualizations', help='Output directory for visualizations')
    return parser.parse_args()

def plot_overall_metrics(history, output_dir):
    """
    Plot overall training and validation metrics.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save plots
    """
    # Create figure for loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    loss_path = os.path.join(output_dir, 'loss.png')
    plt.savefig(loss_path)
    plt.close()
    
    # Create figure for other metrics
    for metric in ['mae', 'mse', 'r_squared']:
        if metric in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[metric], label=f'Training {metric.upper()}')
            if f'val_{metric}' in history:
                plt.plot(history[f'val_{metric}'], label=f'Validation {metric.upper()}')
            plt.title(f'Model {metric.upper()}')
            plt.ylabel(metric.upper())
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save the plot
            metric_path = os.path.join(output_dir, f'{metric}.png')
            plt.savefig(metric_path)
            plt.close()

def plot_per_trait_metrics(history, output_dir):
    """
    Plot per-trait metrics for each of the OCEAN personality traits.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save plots
    """
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    trait_abbrs = [name[0] for name in trait_names]
    
    # Plot per-trait metrics
    for metric in ['mae', 'mse', 'r_squared']:
        # Check if we have per-trait metrics for this metric
        trait_keys = [f'{metric}_{abbr}' for abbr in trait_abbrs]
        valid_keys = [key for key in trait_keys if key in history]
        
        if valid_keys:
            plt.figure(figsize=(12, 7))
            
            for trait_idx, trait in enumerate(trait_names):
                key = f'{metric}_{trait[0]}'
                if key in history:
                    plt.plot(history[key], label=trait)
            
            plt.title(f'Per-Trait {metric.upper()}')
            plt.ylabel(metric.upper())
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save the plot
            metric_path = os.path.join(output_dir, f'per_trait_{metric}.png')
            plt.savefig(metric_path)
            plt.close()
    
    # Create a final comparison figure for R²
    plt.figure(figsize=(12, 7))
    r2_values = []
    labels = []
    
    for trait_idx, trait in enumerate(trait_names):
        key = f'r_squared_{trait[0]}'
        if key in history:
            r2_values.append(history[key][-1])  # Final epoch value
            labels.append(trait)
    
    if r2_values:
        plt.bar(labels, r2_values)
        plt.title('Final R² by Personality Trait')
        plt.ylabel('R²')
        plt.ylim(0, 1)  # R² is typically between 0 and 1
        
        # Add value labels on bars
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Save the plot
        r2_comparison_path = os.path.join(output_dir, 'r2_comparison.png')
        plt.savefig(r2_comparison_path)
        plt.close()

def create_correlation_heatmap(history, output_dir):
    """
    Create a heatmap showing correlations between different metrics.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save plots
    """
    # Extract metrics for correlation analysis
    metrics_to_correlate = []
    metric_names = []
    
    for key in history:
        # Skip validation metrics for clarity
        if key.startswith('val_'):
            continue
        
        metrics_to_correlate.append(history[key])
        metric_names.append(key)
    
    if len(metrics_to_correlate) > 1:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(metrics_to_correlate)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=metric_names, yticklabels=metric_names)
        plt.title('Metric Correlation Matrix')
        
        # Save the plot
        corr_path = os.path.join(output_dir, 'metric_correlation.png')
        plt.savefig(corr_path)
        plt.close()

def main():
    args = parse_args()
    logger = get_logger(experiment_name="visualize_training_metrics")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load training history
    logger.logger.info(f"Loading training history from {args.history}")
    history = np.load(args.history, allow_pickle=True).item()
    
    # Print available metrics
    logger.logger.info(f"Available metrics: {', '.join(history.keys())}")
    
    # Create visualizations
    logger.logger.info("Generating overall metrics plots...")
    plot_overall_metrics(history, args.output)
    
    logger.logger.info("Generating per-trait metrics plots...")
    plot_per_trait_metrics(history, args.output)
    
    logger.logger.info("Generating metric correlation heatmap...")
    create_correlation_heatmap(history, args.output)
    
    logger.logger.info(f"All visualizations saved to {args.output}")
    logger.close()

if __name__ == "__main__":
    main()
