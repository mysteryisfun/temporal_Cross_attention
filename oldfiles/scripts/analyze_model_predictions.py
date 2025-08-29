"""
Analysis of model predictions for Cross-Attention Personality Model (Phase 4.2)

This script loads a trained model and evaluates its predictions on a dataset,
providing detailed analysis of per-trait performance and error distribution.

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/analyze_model_predictions.py --model results/personality_model_trained.h5 --static_features data/static_features.npy --dynamic_features data/dynamic_features.npy --labels data/labels.npy --output results/prediction_analysis
    }
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.models.personality_model import CompletePersonalityModel
from utils.logger import get_logger
from utils.metrics import RSquared, r_squared_metric

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Cross-Attention Personality Model Predictions")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights (.h5)')
    parser.add_argument('--static_features', type=str, required=True, help='Path to static features .npy')
    parser.add_argument('--dynamic_features', type=str, required=True, help='Path to dynamic features .npy')
    parser.add_argument('--labels', type=str, required=True, help='Path to OCEAN labels .npy')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='results/prediction_analysis', help='Output directory for analysis')
    return parser.parse_args()

def load_model_and_data(model_path, static_path, dynamic_path, labels_path):
    """
    Load the trained model and evaluation data.
    
    Args:
        model_path (str): Path to model weights
        static_path (str): Path to static features
        dynamic_path (str): Path to dynamic features
        labels_path (str): Path to labels
        
    Returns:
        tuple: (model, static_features, dynamic_features, labels)
    """
    # Load features and labels
    static = np.load(static_path)
    dynamic = np.load(dynamic_path)
    labels = np.load(labels_path)
    
    # Create model instance
    model = CompletePersonalityModel(
        static_dim=static.shape[1],
        dynamic_dim=dynamic.shape[1],
        fusion_dim=128,
        num_heads=4,
        dropout_rate=0.3
    )
    
    # Make a fake prediction to build the model
    _ = model((np.zeros((1, static.shape[1])), np.zeros((1, dynamic.shape[1]))))
    
    # Load weights
    model.load_weights(model_path)
    
    return model, static, dynamic, labels

def predict_in_batches(model, static, dynamic, batch_size=32):
    """
    Make predictions in batches to avoid memory issues.
    
    Args:
        model (tf.keras.Model): Trained model
        static (numpy.ndarray): Static features
        dynamic (numpy.ndarray): Dynamic features
        batch_size (int): Batch size
        
    Returns:
        numpy.ndarray: Predictions
    """
    n_samples = static.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_static = static[start_idx:end_idx]
        batch_dynamic = dynamic[start_idx:end_idx]
        
        batch_preds = model((batch_static, batch_dynamic), training=False).numpy()
        predictions.append(batch_preds)
    
    return np.vstack(predictions)

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        
    Returns:
        dict: Dictionary containing metrics
    """
    n_traits = y_true.shape[1]
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    # Overall metrics
    mse = np.mean(np.square(y_true - y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R-squared
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true, axis=0)))
    r2 = 1 - (ss_res / ss_tot)
    
    # Per-trait metrics
    trait_metrics = {}
    
    for i in range(n_traits):
        trait = trait_names[i]
        trait_true = y_true[:, i]
        trait_pred = y_pred[:, i]
        
        # Basic metrics
        trait_metrics[trait] = {
            'mse': np.mean(np.square(trait_true - trait_pred)),
            'mae': np.mean(np.abs(trait_true - trait_pred)),
            'mean_true': np.mean(trait_true),
            'mean_pred': np.mean(trait_pred),
            'std_true': np.std(trait_true),
            'std_pred': np.std(trait_pred),
            'min_true': np.min(trait_true),
            'max_true': np.max(trait_true),
            'min_pred': np.min(trait_pred),
            'max_pred': np.max(trait_pred),
        }
        
        # Calculate per-trait R-squared
        ss_res_trait = np.sum(np.square(trait_true - trait_pred))
        ss_tot_trait = np.sum(np.square(trait_true - np.mean(trait_true)))
        trait_metrics[trait]['r2'] = 1 - (ss_res_trait / ss_tot_trait)
        
        # Calculate correlation
        trait_metrics[trait]['pearson_r'], trait_metrics[trait]['pearson_p'] = stats.pearsonr(trait_true, trait_pred)
    
    return {
        'overall_mse': mse,
        'overall_mae': mae,
        'overall_r2': r2,
        'trait_metrics': trait_metrics
    }

def plot_predictions_vs_true(y_true, y_pred, trait_names, output_dir):
    """
    Create scatter plots of predicted vs. true values for each trait.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        trait_names (list): Names of personality traits
        output_dir (str): Output directory
    """
    n_traits = len(trait_names)
    
    for i in range(n_traits):
        plt.figure(figsize=(8, 8))
        
        trait_true = y_true[:, i]
        trait_pred = y_pred[:, i]
        
        # Calculate metrics for this trait
        pearson_r, _ = stats.pearsonr(trait_true, trait_pred)
        mse = np.mean(np.square(trait_true - trait_pred))
        
        # Get axis limits
        min_val = min(np.min(trait_true), np.min(trait_pred))
        max_val = max(np.max(trait_true), np.max(trait_pred))
        
        # Create scatter plot
        plt.scatter(trait_true, trait_pred, alpha=0.5)
        
        # Add identity line (perfect predictions)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        z = np.polyfit(trait_true, trait_pred, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(trait_true), p(np.sort(trait_true)), 'g-')
        
        # Add metrics to the plot
        plt.text(
            0.05, 0.95, 
            f'r = {pearson_r:.4f}\nMSE = {mse:.4f}', 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.title(f'{trait_names[i]} - Predicted vs. True Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'{trait_names[i].lower()}_scatter.png')
        plt.savefig(plot_path)
        plt.close()

def plot_error_distribution(y_true, y_pred, trait_names, output_dir):
    """
    Plot distribution of prediction errors for each trait.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        trait_names (list): Names of personality traits
        output_dir (str): Output directory
    """
    n_traits = len(trait_names)
    
    for i in range(n_traits):
        plt.figure(figsize=(10, 6))
        
        errors = y_pred[:, i] - y_true[:, i]
        
        # Create histogram
        sns.histplot(errors, kde=True)
        
        # Add vertical line at zero (no error)
        plt.axvline(0, color='r', linestyle='--')
        
        # Calculate summary statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Add statistics to the plot
        plt.text(
            0.05, 0.95, 
            f'Mean error = {mean_error:.4f}\nStd error = {std_error:.4f}', 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.title(f'{trait_names[i]} - Error Distribution')
        plt.xlabel('Prediction Error (Predicted - True)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'{trait_names[i].lower()}_error_dist.png')
        plt.savefig(plot_path)
        plt.close()

def plot_trait_correlations(y_true, y_pred, trait_names, output_dir):
    """
    Plot correlation matrix between traits for true and predicted values.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        trait_names (list): Names of personality traits
        output_dir (str): Output directory
    """
    # True value correlations
    plt.figure(figsize=(10, 8))
    true_corr = np.corrcoef(y_true.T)
    sns.heatmap(true_corr, annot=True, cmap='coolwarm', xticklabels=trait_names, yticklabels=trait_names)
    plt.title('True Value Trait Correlations')
    
    # Save the plot
    true_corr_path = os.path.join(output_dir, 'true_trait_correlations.png')
    plt.savefig(true_corr_path)
    plt.close()
    
    # Predicted value correlations
    plt.figure(figsize=(10, 8))
    pred_corr = np.corrcoef(y_pred.T)
    sns.heatmap(pred_corr, annot=True, cmap='coolwarm', xticklabels=trait_names, yticklabels=trait_names)
    plt.title('Predicted Value Trait Correlations')
    
    # Save the plot
    pred_corr_path = os.path.join(output_dir, 'predicted_trait_correlations.png')
    plt.savefig(pred_corr_path)
    plt.close()
    
    # Correlation difference (how much the model's predictions diverge from true correlations)
    plt.figure(figsize=(10, 8))
    corr_diff = pred_corr - true_corr
    sns.heatmap(corr_diff, annot=True, cmap='coolwarm', xticklabels=trait_names, yticklabels=trait_names, vmin=-1, vmax=1)
    plt.title('Trait Correlation Differences (Predicted - True)')
    
    # Save the plot
    diff_corr_path = os.path.join(output_dir, 'trait_correlation_differences.png')
    plt.savefig(diff_corr_path)
    plt.close()

def main():
    args = parse_args()
    logger = get_logger(experiment_name="analyze_model_predictions")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model and data
    logger.logger.info("Loading model and data...")
    model, static, dynamic, labels = load_model_and_data(
        args.model, args.static_features, args.dynamic_features, args.labels
    )
    
    # Make predictions
    logger.logger.info("Making predictions...")
    predictions = predict_in_batches(model, static, dynamic, args.batch_size)
    
    # Calculate metrics
    logger.logger.info("Calculating metrics...")
    metrics = calculate_metrics(labels, predictions)
    
    # Log overall metrics
    logger.logger.info(f"Overall MSE: {metrics['overall_mse']:.4f}")
    logger.logger.info(f"Overall MAE: {metrics['overall_mae']:.4f}")
    logger.logger.info(f"Overall R²: {metrics['overall_r2']:.4f}")
    
    # Log per-trait metrics
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    logger.logger.info("Per-trait metrics:")
    
    for trait in trait_names:
        trait_r2 = metrics['trait_metrics'][trait]['r2']
        trait_mse = metrics['trait_metrics'][trait]['mse']
        trait_corr = metrics['trait_metrics'][trait]['pearson_r']
        logger.logger.info(f"  {trait:15s} - R²: {trait_r2:.4f}, MSE: {trait_mse:.4f}, Correlation: {trait_corr:.4f}")
    
    # Create visualizations
    logger.logger.info("Creating prediction vs. true value scatter plots...")
    plot_predictions_vs_true(labels, predictions, trait_names, args.output)
    
    logger.logger.info("Creating error distribution plots...")
    plot_error_distribution(labels, predictions, trait_names, args.output)
    
    logger.logger.info("Creating trait correlation plots...")
    plot_trait_correlations(labels, predictions, trait_names, args.output)
    
    # Save metrics to file
    metrics_path = os.path.join(args.output, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        metrics_json = {}
        metrics_json['overall_mse'] = float(metrics['overall_mse'])
        metrics_json['overall_mae'] = float(metrics['overall_mae'])
        metrics_json['overall_r2'] = float(metrics['overall_r2'])
        
        metrics_json['trait_metrics'] = {}
        for trait, trait_dict in metrics['trait_metrics'].items():
            metrics_json['trait_metrics'][trait] = {k: float(v) for k, v in trait_dict.items()}
        
        json.dump(metrics_json, f, indent=2)
    
    logger.logger.info(f"Analysis results saved to {args.output}")
    logger.close()

if __name__ == "__main__":
    main()
