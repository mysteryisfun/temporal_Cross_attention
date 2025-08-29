"""
Training Time Analysis Tool for Cross-Attention CNN (Phase 4.3)

This script analyzes training time logs to provide insights into:
- Training time distribution across epochs
- Performance bottlenecks
- Resource utilization patterns
- Batch processing efficiency
- Potential optimization opportunities

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/analyze_training_time.py --time_logs results/time_tracking/time_logs.json --metrics_logs results/batch_metrics/metrics_history.json --output results/training_time_analysis
    }
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Training Time Performance")
    parser.add_argument('--time_logs', type=str, required=True, help='Path to time_logs.json')
    parser.add_argument('--metrics_logs', type=str, required=True, help='Path to metrics_history.json')
    parser.add_argument('--output', type=str, default='results/training_time_analysis', help='Output directory for analysis results')
    return parser.parse_args()


def load_log_data(time_logs_path, metrics_logs_path):
    """
    Load time and metrics log data.
    
    Args:
        time_logs_path: Path to time logs JSON file
        metrics_logs_path: Path to metrics history JSON file
        
    Returns:
        tuple: (time_logs, metrics_logs)
    """
    # Load time logs
    with open(time_logs_path, 'r') as f:
        time_logs = json.load(f)
    
    # Load metrics logs
    with open(metrics_logs_path, 'r') as f:
        metrics_logs = json.load(f)
    
    return time_logs, metrics_logs


def analyze_epoch_times(time_logs, output_dir):
    """
    Analyze and visualize epoch-level training times.
    
    Args:
        time_logs: Dictionary containing time tracking data
        output_dir: Directory to save visualizations
    """
    # Convert to pandas DataFrame for easier analysis
    epoch_data = pd.DataFrame({
        'epoch': time_logs['epoch_numbers'],
        'time_seconds': time_logs['epoch_times']
    })
    
    # Basic statistics
    stats = {
        'total_epochs': len(epoch_data),
        'total_training_time': time_logs.get('total_training_time', sum(time_logs['epoch_times'])),
        'mean_epoch_time': np.mean(epoch_data['time_seconds']),
        'std_epoch_time': np.std(epoch_data['time_seconds']),
        'min_epoch_time': np.min(epoch_data['time_seconds']),
        'max_epoch_time': np.max(epoch_data['time_seconds'])
    }
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create bar plot with line showing cumulative time
    ax1 = plt.gca()
    ax1.bar(epoch_data['epoch'], epoch_data['time_seconds'], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time per Epoch (seconds)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Add cumulative time line
    ax2 = ax1.twinx()
    cumulative_time = np.cumsum(epoch_data['time_seconds'])
    ax2.plot(epoch_data['epoch'], cumulative_time, color='orangered', linewidth=2, marker='o')
    ax2.set_ylabel('Cumulative Training Time (seconds)', color='orangered')
    ax2.tick_params(axis='y', labelcolor='orangered')
    
    # Add horizontal line for mean epoch time
    ax1.axhline(stats['mean_epoch_time'], color='navy', linestyle='--', label=f'Mean: {stats["mean_epoch_time"]:.2f}s')
    
    # Add annotations
    plt.title('Training Time per Epoch Analysis')
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = "\n".join([
        f"Total Training Time: {stats['total_training_time']:.2f}s",
        f"Mean Epoch Time: {stats['mean_epoch_time']:.2f}s",
        f"Std Dev: {stats['std_epoch_time']:.2f}s",
        f"Min: {stats['min_epoch_time']:.2f}s",
        f"Max: {stats['max_epoch_time']:.2f}s"
    ])
    
    plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epoch_time_analysis.png'), dpi=300)
    plt.close()
    
    # Save statistics to JSON
    with open(os.path.join(output_dir, 'epoch_time_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)


def analyze_batch_times(time_logs, metrics_logs, output_dir):
    """
    Analyze and visualize batch-level training times.
    
    Args:
        time_logs: Dictionary containing time tracking data
        metrics_logs: Dictionary containing metrics history
        output_dir: Directory to save visualizations
    """
    # Check if we have batch timing data
    if not time_logs.get('batch_times') or len(time_logs['batch_times']) == 0:
        print("No batch timing data available for analysis")
        return
    
    # Convert to pandas DataFrame
    batch_data = pd.DataFrame({
        'batch': time_logs['batch_numbers'],
        'time_seconds': time_logs['batch_times']
    })
    
    # Add batch loss data if available
    if metrics_logs.get('batch') and metrics_logs['batch'].get('loss'):
        batch_loss = pd.DataFrame({
            'batch': metrics_logs['batch']['batch_numbers'],
            'loss': metrics_logs['batch']['loss']
        })
        # Merge data if batch numbers align
        if set(batch_data['batch']).issubset(set(batch_loss['batch'])):
            batch_data = pd.merge(batch_data, batch_loss, on='batch', how='left')
    
    # Calculate statistics
    stats = {
        'total_batches_timed': len(batch_data),
        'mean_batch_time': np.mean(batch_data['time_seconds']),
        'std_batch_time': np.std(batch_data['time_seconds']),
        'min_batch_time': np.min(batch_data['time_seconds']),
        'max_batch_time': np.max(batch_data['time_seconds']),
        'estimated_total_batch_time': np.mean(batch_data['time_seconds']) * 
                                    (max(time_logs['batch_numbers']) if time_logs['batch_numbers'] else 0)
    }
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot batch times
    ax1 = plt.gca()
    ax1.scatter(batch_data['batch'], batch_data['time_seconds'], alpha=0.5, 
               color='steelblue', label='Batch Processing Time')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Time per Batch (seconds)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Add rolling average line
    window_size = min(20, len(batch_data) // 4) if len(batch_data) > 4 else 2
    if len(batch_data) >= window_size:
        rolling_avg = batch_data['time_seconds'].rolling(window=window_size).mean()
        ax1.plot(batch_data['batch'], rolling_avg, color='navy', 
                linewidth=2, label=f'{window_size}-batch Moving Average')
    
    # If we have loss data, plot it on a secondary axis
    if 'loss' in batch_data.columns:
        ax2 = ax1.twinx()
        ax2.plot(batch_data['batch'], batch_data['loss'], color='orangered', 
                alpha=0.7, linewidth=1, label='Batch Loss')
        ax2.set_ylabel('Batch Loss', color='orangered')
        ax2.tick_params(axis='y', labelcolor='orangered')
    
    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    if 'loss' in batch_data.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # Add title and grid
    plt.title('Batch Processing Time Analysis')
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = "\n".join([
        f"Batches Analyzed: {stats['total_batches_timed']}",
        f"Mean Batch Time: {stats['mean_batch_time']:.4f}s",
        f"Std Dev: {stats['std_batch_time']:.4f}s",
        f"Min: {stats['min_batch_time']:.4f}s",
        f"Max: {stats['max_batch_time']:.4f}s"
    ])
    
    plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_time_analysis.png'), dpi=300)
    plt.close()
    
    # Save statistics to JSON
    with open(os.path.join(output_dir, 'batch_time_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)


def create_training_speedup_recommendations(time_logs, metrics_logs, output_dir):
    """
    Analyze training performance and provide optimization recommendations.
    
    Args:
        time_logs: Dictionary containing time tracking data
        metrics_logs: Dictionary containing metrics history
        output_dir: Directory to save recommendations
    """
    # List to store recommendations
    recommendations = []
    
    # 1. Analyze epoch time patterns
    if time_logs.get('epoch_times') and len(time_logs['epoch_times']) > 1:
        epoch_times = np.array(time_logs['epoch_times'])
        mean_time = np.mean(epoch_times)
        std_time = np.std(epoch_times)
        
        # Check for high variance in epoch times
        if std_time / mean_time > 0.2:  # More than 20% coefficient of variation
            recommendations.append({
                "issue": "High variance in epoch processing times",
                "details": f"Epoch times vary by {std_time/mean_time*100:.1f}% relative to the mean, "
                          f"suggesting inconsistent resource availability or data loading bottlenecks.",
                "recommendation": "Consider using caching mechanisms for data loading, or investigate "
                                "if other processes are competing for resources during training."
            })
        
        # Check for increasing epoch times
        if len(epoch_times) >= 5:
            early_epochs = epoch_times[:len(epoch_times)//2]
            late_epochs = epoch_times[len(epoch_times)//2:]
            if np.mean(late_epochs) > np.mean(early_epochs) * 1.2:  # 20% slower
                recommendations.append({
                    "issue": "Increasing epoch times as training progresses",
                    "details": f"Later epochs take {np.mean(late_epochs)/np.mean(early_epochs)*100-100:.1f}% "
                              f"more time than earlier epochs.",
                    "recommendation": "This could indicate memory leaks or increasing gradient computation complexity. "
                                    "Consider using memory profiling tools and gradient checkpointing to reduce memory usage."
                })
    
    # 2. Analyze batch time patterns
    if time_logs.get('batch_times') and len(time_logs['batch_times']) > 10:
        batch_times = np.array(time_logs['batch_times'])
        
        # Check for outlier batch times (more than 3 std devs from mean)
        mean_batch = np.mean(batch_times)
        std_batch = np.std(batch_times)
        outliers = batch_times[batch_times > mean_batch + 3*std_batch]
        
        if len(outliers) > 0:
            recommendations.append({
                "issue": f"Detected {len(outliers)} outlier batch times",
                "details": f"Some batches took significantly longer than average (>{mean_batch + 3*std_batch:.4f}s vs average {mean_batch:.4f}s)",
                "recommendation": "Consider checking for irregular data samples that might cause computational spikes, "
                                "or implement batch prefetching to smooth out data loading irregularities."
            })
    
    # 3. Estimate potential gains
    if time_logs.get('total_training_time'):
        total_time = time_logs['total_training_time']
        # Rough estimate of time spent in data loading (often ~20% of training time in non-optimized pipelines)
        estimated_data_loading = total_time * 0.2
        
        recommendations.append({
            "issue": "Potential for training pipeline optimization",
            "details": f"Based on total training time of {total_time:.2f}s, an estimated {estimated_data_loading:.2f}s "
                      f"might be spent in data loading and preprocessing.",
            "recommendation": "Consider implementing tf.data pipeline optimizations like prefetching, parallel processing, "
                            "and caching. This could potentially reduce training time by 10-15%."
        })
    
    # 4. Look at convergence vs time tradeoff
    if (metrics_logs.get('epoch') and metrics_logs['epoch'].get('loss') and 
        metrics_logs['epoch'].get('epoch_numbers') and time_logs.get('epoch_times')):
        
        # Match epoch times with losses
        epoch_losses = metrics_logs['epoch']['loss']
        
        if len(epoch_losses) > 5:
            # Check how much of training time was spent in the final 20% of improvement
            initial_loss = epoch_losses[0]
            final_loss = epoch_losses[-1]
            total_improvement = initial_loss - final_loss
            
            # Find epoch where we achieved 80% of total improvement
            for i, loss in enumerate(epoch_losses):
                if initial_loss - loss >= 0.8 * total_improvement:
                    epoch_80pct = i
                    break
            else:
                epoch_80pct = len(epoch_losses) - 1
            
            # Calculate time spent after 80% improvement
            time_after_80pct = sum(time_logs['epoch_times'][epoch_80pct:])
            total_epoch_time = sum(time_logs['epoch_times'])
            
            if time_after_80pct > 0.4 * total_epoch_time:  # More than 40% of time for final 20% improvement
                recommendations.append({
                    "issue": "Diminishing returns in training time",
                    "details": f"{time_after_80pct/total_epoch_time*100:.1f}% of training time was spent achieving "
                              f"the final 20% of model improvement.",
                    "recommendation": "Consider using more aggressive early stopping, reducing patience in ReduceLROnPlateau, "
                                    "or implementing a custom stopping criterion based on improvement rate vs. time cost."
                })
    
    # 5. General recommendation based on total time
    if time_logs.get('total_training_time'):
        # Check if total training time is over 1 hour
        if time_logs['total_training_time'] > 3600:
            recommendations.append({
                "issue": "Long overall training time",
                "details": f"Total training time of {time_logs['total_training_time']/3600:.2f} hours might impact "
                          f"research iteration speed.",
                "recommendation": "Consider using mixed-precision training (float16), increasing batch size if memory allows, "
                                "or implementing checkpoint-based resumable training to allow for shorter experimental runs."
            })
    
    # Save recommendations to file
    with open(os.path.join(output_dir, 'training_speedup_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=4)
    
    # Also create a human-readable version
    with open(os.path.join(output_dir, 'training_speedup_recommendations.txt'), 'w') as f:
        f.write("# TRAINING SPEEDUP RECOMMENDATIONS\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"## {i}. {rec['issue']}\n\n")
            f.write(f"**Details**: {rec['details']}\n\n")
            f.write(f"**Recommendation**: {rec['recommendation']}\n\n")
            f.write("---\n\n")


def create_training_time_dashboard(time_logs, metrics_logs, output_dir):
    """
    Create a comprehensive dashboard visualizing training time analysis.
    
    Args:
        time_logs: Dictionary containing time tracking data
        metrics_logs: Dictionary containing metrics history
        output_dir: Directory to save visualizations
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Epoch time histogram (top left)
    ax_hist = fig.add_subplot(gs[0, 0])
    epoch_times = np.array(time_logs['epoch_times'])
    ax_hist.hist(epoch_times, bins=min(10, len(epoch_times)), alpha=0.7, color='steelblue')
    ax_hist.set_title('Epoch Time Distribution')
    ax_hist.set_xlabel('Time (seconds)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True, alpha=0.3)
    
    # 2. Epoch time series (top middle)
    ax_series = fig.add_subplot(gs[0, 1])
    ax_series.plot(time_logs['epoch_numbers'], time_logs['epoch_times'], 
                  marker='o', linestyle='-', color='steelblue')
    ax_series.set_title('Epoch Time Series')
    ax_series.set_xlabel('Epoch')
    ax_series.set_ylabel('Time (seconds)')
    ax_series.grid(True, alpha=0.3)
    
    # 3. Total time breakdown (top right)
    ax_pie = fig.add_subplot(gs[0, 2])
    if time_logs.get('total_training_time'):
        # Rough estimation of time components
        total_time = time_logs['total_training_time']
        data_loading_time = total_time * 0.2  # Estimated
        forward_pass_time = total_time * 0.3  # Estimated
        backward_pass_time = total_time * 0.4  # Estimated
        other_time = total_time * 0.1  # Estimated
        
        labels = ['Data Loading', 'Forward Pass', 'Backward Pass', 'Other']
        sizes = [data_loading_time, forward_pass_time, backward_pass_time, other_time]
        
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                 colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold'])
        ax_pie.set_title('Estimated Time Breakdown')
    else:
        ax_pie.text(0.5, 0.5, 'No total time data available', 
                   ha='center', va='center', fontsize=12)
    
    # 4. Batch time scatterplot (middle left and middle)
    if time_logs.get('batch_times') and len(time_logs['batch_times']) > 0:
        ax_batch = fig.add_subplot(gs[1, :2])
        ax_batch.scatter(time_logs['batch_numbers'], time_logs['batch_times'], 
                        alpha=0.5, color='steelblue')
          # Add trend line if enough points
        if len(time_logs['batch_times']) > 5:
            try:
                z = np.polyfit(time_logs['batch_numbers'], time_logs['batch_times'], 1)
                p = np.poly1d(z)
                ax_batch.plot(time_logs['batch_numbers'], p(time_logs['batch_numbers']), 
                             color='red', linestyle='--', label=f'Trend: {z[0]:.2e}x + {z[1]:.4f}')
                ax_batch.legend()
            except np.linalg.LinAlgError as e:
                # Handle case where fit fails (e.g., not enough data points)
                ax_batch.text(0.5, 0.5, f'Could not fit trend line: {e}', 
                             transform=ax_batch.transAxes, ha='center', va='center')
                print(f"Warning: Could not fit trend line: {e}")
        
        ax_batch.set_title('Batch Processing Time')
        ax_batch.set_xlabel('Batch Number')
        ax_batch.set_ylabel('Time (seconds)')
        ax_batch.grid(True, alpha=0.3)
    
    # 5. Learning curve vs time (middle right)
    if (metrics_logs.get('epoch') and metrics_logs['epoch'].get('loss') and 
        metrics_logs['epoch'].get('val_loss') and time_logs.get('epoch_times')):
        
        ax_curve = fig.add_subplot(gs[1, 2])
        
        # Calculate cumulative time
        cum_time = np.cumsum(time_logs['epoch_times'])
        
        # Plot learning curves
        ax_curve.plot(cum_time, metrics_logs['epoch']['loss'], 
                     marker='o', linestyle='-', label='Training Loss')
        
        if len(metrics_logs['epoch']['val_loss']) == len(cum_time):
            ax_curve.plot(cum_time, metrics_logs['epoch']['val_loss'], 
                         marker='o', linestyle='-', label='Validation Loss')
        
        ax_curve.set_title('Learning Curve vs. Training Time')
        ax_curve.set_xlabel('Cumulative Time (seconds)')
        ax_curve.set_ylabel('Loss')
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend()
    
    # 6. Time efficiency metrics (bottom row)
    ax_metrics = fig.add_subplot(gs[2, :])
    
    # Calculate efficiency metrics
    metrics = []
    labels = []
    
    # Average time per epoch
    if time_logs.get('epoch_times'):
        metrics.append(np.mean(time_logs['epoch_times']))
        labels.append('Avg Time per Epoch')
    
    # Average time per batch
    if time_logs.get('batch_times'):
        metrics.append(np.mean(time_logs['batch_times']))
        labels.append('Avg Time per Batch')
    
    # Time to halve the loss
    if (metrics_logs.get('epoch') and metrics_logs['epoch'].get('loss') and 
        len(metrics_logs['epoch']['loss']) > 1 and time_logs.get('epoch_times')):
        
        initial_loss = metrics_logs['epoch']['loss'][0]
        target_loss = initial_loss / 2
        
        for i, loss in enumerate(metrics_logs['epoch']['loss']):
            if loss <= target_loss:
                time_to_halve = sum(time_logs['epoch_times'][:i+1])
                metrics.append(time_to_halve)
                labels.append('Time to Halve Loss')
                break
    
    # Total training time
    if time_logs.get('total_training_time'):
        metrics.append(time_logs['total_training_time'])
        labels.append('Total Training Time')
    
    # Create horizontal bar chart for these metrics
    y_pos = np.arange(len(labels))
    ax_metrics.barh(y_pos, metrics, align='center')
    ax_metrics.set_yticks(y_pos)
    ax_metrics.set_yticklabels(labels)
    ax_metrics.invert_yaxis()  # Labels read top-to-bottom
    ax_metrics.set_title('Training Time Efficiency Metrics')
    ax_metrics.set_xlabel('Time (seconds)')
    
    # Add values as text
    for i, v in enumerate(metrics):
        if v >= 100:
            text = f'{v:.1f}s'
        else:
            text = f'{v:.4f}s'
        ax_metrics.text(v + 0.1, i, text, va='center')
    
    # Add overall title
    plt.suptitle('Training Time Performance Analysis Dashboard', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save dashboard
    plt.savefig(os.path.join(output_dir, 'training_time_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    logger = get_logger(experiment_name="analyze_training_time")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.logger.info("Loading time and metrics logs...")
    try:
        time_logs, metrics_logs = load_log_data(args.time_logs, args.metrics_logs)
        logger.logger.info("Data loaded successfully")
    except Exception as e:
        logger.logger.error(f"Error loading log data: {e}")
        return
    
    # Run analyses
    logger.logger.info("Analyzing epoch-level training times...")
    analyze_epoch_times(time_logs, args.output)
    
    logger.logger.info("Analyzing batch-level training times...")
    analyze_batch_times(time_logs, metrics_logs, args.output)
    
    logger.logger.info("Creating training speedup recommendations...")
    create_training_speedup_recommendations(time_logs, metrics_logs, args.output)
    
    logger.logger.info("Creating comprehensive training time dashboard...")
    create_training_time_dashboard(time_logs, metrics_logs, args.output)
    
    logger.logger.info(f"All analysis results saved to {args.output}")
    logger.close()


if __name__ == "__main__":
    main()
