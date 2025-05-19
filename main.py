"""
Main pipeline for Cross-Attention CNN Personality Trait Prediction

This script integrates data loading, preprocessing, static and dynamic feature extraction,
cross-attention fusion, and OCEAN trait prediction. It is designed for both training and inference.

Usage:
    # If virtual environment exists
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python main.py --mode inference --face_dir data/faces --flow_dir data/optical_flow
    }

Refer to docs/phase_3_complete_documentation.md for detailed instructions.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from utils.logger import get_logger
from models.static_feature_extractor.feature_extractor import StaticFeatureExtractor
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from src.models.personality_model import CompletePersonalityModel

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Attention CNN Pipeline")
    parser.add_argument('--mode', type=str, default='inference', choices=['inference', 'train'], help='Pipeline mode')
    parser.add_argument('--face_dir', type=str, required=True, help='Directory with face images')
    parser.add_argument('--flow_dir', type=str, required=True, help='Directory with optical flow sequences')
    parser.add_argument('--static_config', type=str, default=None, help='Path to static model config')
    parser.add_argument('--dynamic_config', type=str, default=None, help='Path to dynamic model config')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for feature extraction')
    parser.add_argument('--output', type=str, default='results/predictions.npy', help='Output file for predictions')
    return parser.parse_args()

# --- Data Loading Utilities ---
def load_face_images(face_dir, batch_size, extractor):
    """
    Loads and preprocesses face images from a directory.
    Returns a numpy array of preprocessed images.
    """
    import cv2
    images = []
    for fname in sorted(os.listdir(face_dir)):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(face_dir, fname))
            img = cv2.resize(img, (224, 224))
            img = extractor.preprocess_image(img)
            images.append(img)
    if not images:
        raise ValueError(f"No face images found in {face_dir}")
    images = np.vstack(images)
    return images

def load_optical_flow_sequences(flow_dir, batch_size, extractor):
    """
    Loads and preprocesses optical flow sequences from a directory.
    Returns a numpy array of preprocessed flow sequences.
    """
    flow_sequences = []
    for fname in sorted(os.listdir(flow_dir)):
        if fname.lower().endswith('.npy'):
            flow = np.load(os.path.join(flow_dir, fname))
            flow = extractor.preprocess_flow_sequence(flow)
            flow_sequences.append(flow)
    if not flow_sequences:
        raise ValueError(f"No optical flow .npy files found in {flow_dir}")
    flow_sequences = np.vstack(flow_sequences)
    return flow_sequences

# --- Main Pipeline ---
def main():
    args = parse_args()
    logger = get_logger(experiment_name="main_pipeline")
    logger.log_system_info()

    # Initialize feature extractors
    static_extractor = StaticFeatureExtractor(config_path=args.static_config)
    dynamic_extractor = DynamicFeatureExtractor(config_path=args.dynamic_config)

    # Load and preprocess data
    logger.logger.info("Loading and preprocessing face images...")
    face_images = load_face_images(args.face_dir, args.batch_size, static_extractor)
    logger.logger.info(f"Loaded {face_images.shape[0]} face images.")

    logger.logger.info("Loading and preprocessing optical flow sequences...")
    flow_sequences = load_optical_flow_sequences(args.flow_dir, args.batch_size, dynamic_extractor)
    logger.logger.info(f"Loaded {flow_sequences.shape[0]} optical flow sequences.")

    # Extract features
    logger.logger.info("Extracting static features...")
    static_features = static_extractor.extract_features(face_images)
    logger.logger.info(f"Static features shape: {static_features.shape}")

    logger.logger.info("Extracting dynamic features...")
    dynamic_features = dynamic_extractor.extract_features(flow_sequences)
    logger.logger.info(f"Dynamic features shape: {dynamic_features.shape}")

    # Initialize complete model
    model = CompletePersonalityModel(
        static_dim=static_features.shape[1],
        dynamic_dim=dynamic_features.shape[1],
        fusion_dim=128,
        num_heads=4,
        dropout_rate=0.3
    )

    # Inference or training
    if args.mode == 'inference':
        logger.logger.info("Running inference...")
        predictions = model((static_features, dynamic_features), training=False).numpy()
        logger.logger.info(f"Predictions shape: {predictions.shape}")
        np.save(args.output, predictions)
        logger.logger.info(f"Predictions saved to {args.output}")
    else:
        logger.logger.info("Training mode is not implemented in this main pipeline. Please use the dedicated training script.")

    logger.close()

if __name__ == "__main__":
    main()
