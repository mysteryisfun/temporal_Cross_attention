#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: evaluate_dynamic_feature_extractor.py
# Description: Comprehensive evaluation of the dynamic feature extractor

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.evaluation import FeatureExtractorEvaluator
from scripts.optical_flow_sequence_loader import OpticalFlowSequenceLoader
from utils.logger import get_logger

def evaluate_feature_extraction_accuracy(flow_dir, ground_truth_dir=None, model_path=None):
    """
    Evaluate the accuracy of feature extraction against ground truth if available.
    
    Args:
        flow_dir (str): Directory containing optical flow data.
        ground_truth_dir (str): Directory containing ground truth data.
        model_path (str): Path to the trained model.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger = get_logger(experiment_name="feature_extraction_accuracy")
    
    # Initialize feature extractor
    extractor = DynamicFeatureExtractor()
    if model_path and os.path.exists(model_path):
        extractor.load_model(model_path)
        logger.logger.info(f"Loaded model from {model_path}")
    
    # Find all flow files
    flow_files = []
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.endswith('_flow_raw.npy'):
                flow_files.append(os.path.join(root, file))
    
    if not flow_files:
        logger.logger.error(f"No flow files found in {flow_dir}")
        return None
    
    logger.logger.info(f"Found {len(flow_files)} flow files")
    
    # Extract features from all flow files
    all_features = []
    file_names = []
    
    for flow_file in tqdm(flow_files, desc="Extracting features"):
        try:
            # Load flow sequence
            flow_sequence = np.load(flow_file)
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess
            preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
            
            # Extract features
            features = extractor.extract_features(preprocessed)
            all_features.append(features[0])
            file_names.append(os.path.basename(flow_file))
        except Exception as e:
            logger.logger.error(f"Error processing {flow_file}: {str(e)}")
    
    if not all_features:
        logger.logger.error("No features could be extracted")
        return None
    
    # Convert to numpy array
    all_features = np.array(all_features)
    logger.logger.info(f"Extracted features shape: {all_features.shape}")
    
    # If ground truth is available, compute accuracy metrics
    if ground_truth_dir and os.path.exists(ground_truth_dir):
        # TODO: Implement comparison with ground truth if available
        pass
    
    # Compute feature statistics
    mean_feature = np.mean(all_features, axis=0)
    std_feature = np.std(all_features, axis=0)
    
    # Compute pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(all_features)
    
    # Compute silhouette score if we have labels (from directory structure)
    silhouette = None
    try:
        # Try to extract labels from directory structure
        labels = []
        for file_name in file_names:
            # Assume format: <label>_<sequence_id>_flow_raw.npy
            label = file_name.split('_')[0]
            labels.append(label)
        
        # Count unique labels
        unique_labels = set(labels)
        if len(unique_labels) >= 2:
            silhouette = silhouette_score(all_features, labels)
            logger.logger.info(f"Silhouette score: {silhouette}")
    except Exception as e:
        logger.logger.warning(f"Could not compute silhouette score: {str(e)}")
    
    return {
        'num_samples': len(all_features),
        'feature_dimensionality': all_features.shape[1],
        'mean_feature': mean_feature.tolist(),
        'std_feature': std_feature.tolist(),
        'mean_pairwise_similarity': float(np.mean(similarities)),
        'silhouette_score': float(silhouette) if silhouette is not None else None
    }

def visualize_feature_space(features, labels=None, output_path=None):
    """
    Visualize the feature space using dimensionality reduction.
    
    Args:
        features (numpy.ndarray): Feature vectors.
        labels (list): Labels for each feature vector.
        output_path (str): Path to save the visualization.
    """
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Apply PCA
    plt.subplot(1, 2, 1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    if labels:
        unique_labels = set(labels)
        for label in unique_labels:
            mask = [l == label for l in labels]
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=label, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    
    plt.title(f"PCA Projection of Features\nExplained Variance: {sum(pca.explained_variance_ratio_):.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    
    # Apply t-SNE
    plt.subplot(1, 2, 2)
    tsne = TSNE(n_components=2, perplexity=min(30, features.shape[0]-1), n_iter=1000)
    tsne_result = tsne.fit_transform(features)
    
    if labels:
        unique_labels = set(labels)
        for label in unique_labels:
            mask = [l == label for l in labels]
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=label, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
    
    plt.title("t-SNE Projection of Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def evaluate_temporal_consistency(flow_files, model_path=None, output_dir=None):
    """
    Evaluate the temporal consistency of features extracted from consecutive flow sequences.
    
    Args:
        flow_files (list): List of flow files ordered temporally.
        model_path (str): Path to the trained model.
        output_dir (str): Directory to save results.
        
    Returns:
        dict: Temporal consistency metrics.
    """
    logger = get_logger(experiment_name="temporal_consistency")
    
    # Initialize feature extractor
    extractor = DynamicFeatureExtractor()
    if model_path and os.path.exists(model_path):
        extractor.load_model(model_path)
        logger.logger.info(f"Loaded model from {model_path}")
    
    # Extract features from each flow file
    features = []
    
    for flow_file in tqdm(flow_files, desc="Extracting sequential features"):
        try:
            # Load flow sequence
            flow_sequence = np.load(flow_file)
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess
            preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
            
            # Extract features
            feature = extractor.extract_features(preprocessed)
            features.append(feature[0])
        except Exception as e:
            logger.logger.error(f"Error processing {flow_file}: {str(e)}")
    
    if len(features) < 2:
        logger.logger.error("Not enough features to evaluate temporal consistency")
        return None
    
    # Compute cosine similarities between consecutive features
    similarities = []
    for i in range(len(features) - 1):
        # Normalize feature vectors
        f1 = features[i] / np.linalg.norm(features[i])
        f2 = features[i+1] / np.linalg.norm(features[i+1])
        
        similarity = np.dot(f1, f2)
        similarities.append(similarity)
    
    # Compute statistics
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    # Visualize temporal consistency
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(similarities)
        plt.axhline(y=mean_similarity, color='r', linestyle='--', 
                   label=f"Mean Similarity: {mean_similarity:.4f}")
        plt.fill_between(range(len(similarities)), 
                        mean_similarity - std_similarity, 
                        mean_similarity + std_similarity, 
                        alpha=0.2, color='r', label=f"±1 Std: {std_similarity:.4f}")
        plt.title("Temporal Consistency of Features")
        plt.xlabel("Frame Pair Index")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(alpha=0.3)
        
        consistency_path = os.path.join(output_dir, "temporal_consistency.png")
        plt.savefig(consistency_path)
        plt.close()
        logger.logger.info(f"Saved temporal consistency plot to {consistency_path}")
        
        # Visualize feature evolution over time
        plt.figure(figsize=(12, 8))
        
        # Take a subset of feature dimensions for visualization
        num_dims = min(20, features[0].shape[0])
        
        # Create a 2D array of features over time
        feature_matrix = np.array([f[:num_dims] for f in features])
        
        plt.imshow(feature_matrix.T, aspect='auto', cmap='viridis')
        plt.colorbar(label="Feature Value")
        plt.title(f"Feature Evolution Over Time (First {num_dims} dimensions)")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Dimension")
        
        evolution_path = os.path.join(output_dir, "feature_evolution.png")
        plt.savefig(evolution_path)
        plt.close()
        logger.logger.info(f"Saved feature evolution plot to {evolution_path}")
    
    return {
        'mean_temporal_similarity': float(mean_similarity),
        'std_temporal_similarity': float(std_similarity),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities))
    }

def run_comprehensive_evaluation(flow_dir, output_dir, model_path=None):
    """
    Run a comprehensive evaluation of the dynamic feature extractor.
    
    Args:
        flow_dir (str): Directory containing optical flow data.
        output_dir (str): Directory to save evaluation results.
        model_path (str): Path to the trained model.
    """
    logger = get_logger(experiment_name="comprehensive_evaluation")
    logger.logger.info("Starting comprehensive evaluation of dynamic feature extractor")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Evaluate feature extraction accuracy
    logger.logger.info("Evaluating feature extraction accuracy")
    accuracy_metrics = evaluate_feature_extraction_accuracy(flow_dir, model_path=model_path)
    
    if accuracy_metrics:
        # Save metrics to JSON
        accuracy_path = os.path.join(output_dir, "feature_extraction_metrics.json")
        with open(accuracy_path, 'w') as f:
            json.dump(accuracy_metrics, f, indent=2)
        logger.logger.info(f"Saved accuracy metrics to {accuracy_path}")
    
    # 2. Use the FeatureExtractorEvaluator for additional metrics
    logger.logger.info("Running evaluation using FeatureExtractorEvaluator")
    evaluator = FeatureExtractorEvaluator(model_path=model_path)
    evaluation_results = evaluator.run_comprehensive_evaluation(
        flow_data_dir=flow_dir,
        output_dir=os.path.join(output_dir, "evaluator_results")
    )
    
    # 3. Evaluate temporal consistency
    logger.logger.info("Evaluating temporal consistency")
    
    # Find flow sequences that are temporally related
    # This depends on the dataset organization - we'll look for sequences from the same video
    video_dirs = []
    for item in os.listdir(flow_dir):
        item_path = os.path.join(flow_dir, item)
        if os.path.isdir(item_path):
            video_dirs.append(item_path)
    
    temporal_metrics = {}
    
    for video_dir in video_dirs[:5]:  # Limit to first 5 videos for efficiency
        video_name = os.path.basename(video_dir)
        logger.logger.info(f"Evaluating temporal consistency for {video_name}")
        
        # Get flow files for this video
        flow_files = []
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.endswith('_flow_raw.npy'):
                    flow_files.append(os.path.join(root, file))
        
        if len(flow_files) >= 2:
            # Sort by filename to approximate temporal order
            flow_files.sort()
            
            # Evaluate temporal consistency
            video_temporal_metrics = evaluate_temporal_consistency(
                flow_files=flow_files,
                model_path=model_path,
                output_dir=os.path.join(output_dir, "temporal", video_name)
            )
            
            if video_temporal_metrics:
                temporal_metrics[video_name] = video_temporal_metrics
    
    # Save temporal metrics to JSON
    if temporal_metrics:
        temporal_path = os.path.join(output_dir, "temporal_consistency_metrics.json")
        with open(temporal_path, 'w') as f:
            json.dump(temporal_metrics, f, indent=2)
        logger.logger.info(f"Saved temporal metrics to {temporal_path}")
    
    # 4. Create a summary report
    summary = {
        'feature_extraction_metrics': accuracy_metrics,
        'evaluator_results': evaluation_results,
        'temporal_metrics': temporal_metrics
    }
    
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.logger.info(f"Saved evaluation summary to {summary_path}")
    
    # Create a human-readable summary report
    report = [
        "# Dynamic Feature Extractor Evaluation Report",
        "",
        "## Feature Extraction Metrics",
        ""
    ]
    
    if accuracy_metrics:
        report.extend([
            f"- Number of samples: {accuracy_metrics['num_samples']}",
            f"- Feature dimensionality: {accuracy_metrics['feature_dimensionality']}",
            f"- Mean pairwise similarity: {accuracy_metrics['mean_pairwise_similarity']:.4f}",
            f"- Silhouette score: {accuracy_metrics['silhouette_score'] if accuracy_metrics['silhouette_score'] else 'N/A'}",
            ""
        ])
    
    report.extend([
        "## Temporal Consistency Metrics",
        ""
    ])
    
    if temporal_metrics:
        for video_name, metrics in temporal_metrics.items():
            report.extend([
                f"### {video_name}",
                f"- Mean temporal similarity: {metrics['mean_temporal_similarity']:.4f}",
                f"- Std temporal similarity: {metrics['std_temporal_similarity']:.4f}",
                f"- Min similarity: {metrics['min_similarity']:.4f}",
                f"- Max similarity: {metrics['max_similarity']:.4f}",
                ""
            ])
    
    report.extend([
        "## Evaluator Results",
        ""
    ])
    
    if evaluation_results:
        report.extend([
            f"- Average within-video consistency: {evaluation_results.get('avg_within_video_consistency', 'N/A')}",
            f"- Between-video discriminability: {evaluation_results.get('between_video_discriminability', 'N/A')}",
            ""
        ])
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    logger.logger.info(f"Saved evaluation report to {report_path}")
    
    logger.logger.info("Comprehensive evaluation completed")

def main():
    """
    Main function to run the comprehensive evaluation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate dynamic feature extractor")
    parser.add_argument("--flow_dir", type=str, required=True,
                       help="Directory containing optical flow data")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained dynamic feature extractor model")
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    run_comprehensive_evaluation(
        flow_dir=args.flow_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )

if __name__ == "__main__":
    main()
