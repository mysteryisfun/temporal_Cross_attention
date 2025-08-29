import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.visualization import visualize_optical_flow_sequence
from utils.logger import get_logger

def load_optical_flow_sequence(flow_file):
    """
    Load an optical flow sequence from a file.
    
    Args:
        flow_file (str): Path to the optical flow file.
        
    Returns:
        numpy.ndarray: Optical flow sequence.
    """
    try:
        flow_sequence = np.load(flow_file)
        return flow_sequence
    except Exception as e:
        print(f"Error loading flow file {flow_file}: {str(e)}")
        return None

def visualize_feature_activations(extractor, flow_sequence, output_path=None):
    """
    Visualize model activations on an optical flow sequence.
    
    Args:
        extractor (DynamicFeatureExtractor): The feature extractor model.
        flow_sequence (numpy.ndarray): Optical flow sequence.
        output_path (str): Path to save the visualization.
        
    Returns:
        numpy.ndarray: Visualization image.
    """
    # Create visualizer
    from models.dynamic_feature_extractor.visualization import FeatureVisualization
    
    # Find the last convolutional layer
    layer_name = None
    for layer in reversed(extractor.model.layers):
        if isinstance(layer, tf.keras.layers.Conv3D):
            layer_name = layer.name
            break
    
    if layer_name is None:
        print("No convolutional layer found in the model")
        return None
    
    print(f"Using layer {layer_name} for visualization")
    
    # Create a visualization object
    visualizer = FeatureVisualization(
        model=extractor.model,
        layer_name=layer_name
    )
    
    # Ensure flow_sequence has batch dimension
    if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
        flow_sequence = np.expand_dims(flow_sequence, 0)
    
    # Preprocess flow sequence
    preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
    
    # Get activations
    activations = visualizer.get_intermediate_activations(preprocessed)
    
    # Visualize flow and activations
    visualizer.visualize_flow_and_activations(
        preprocessed,
        activations,
        save_path=output_path
    )
    
    # Also visualize temporal activation for the first feature
    if output_path:
        temporal_path = output_path.replace('.png', '_temporal.png')
        visualizer.visualize_temporal_activation(
            activations,
            feature_idx=0,
            save_path=temporal_path
        )
    
    return activations

def analyze_feature_importance(extractor, flow_sequence, output_path=None):
    """
    Analyze the importance of different frames in the optical flow sequence.
    
    Args:
        extractor (DynamicFeatureExtractor): The feature extractor model.
        flow_sequence (numpy.ndarray): Optical flow sequence.
        output_path (str): Path to save the visualization.
        
    Returns:
        tuple: Importance scores and top indices.
    """
    # Create visualizer
    from models.dynamic_feature_extractor.visualization import FeatureVisualization
    
    # Find the last convolutional layer
    layer_name = None
    for layer in reversed(extractor.model.layers):
        if isinstance(layer, tf.keras.layers.Conv3D):
            layer_name = layer.name
            break
    
    # Create a visualization object
    visualizer = FeatureVisualization(
        model=extractor.model,
        layer_name=layer_name
    )
    
    # Ensure flow_sequence has batch dimension
    if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
        flow_sequence = np.expand_dims(flow_sequence, 0)
    
    # Preprocess flow sequence
    preprocessed = extractor.preprocess_flow_sequence(flow_sequence)
    
    # Analyze feature importance
    importance, top_indices = visualizer.visualize_feature_importance(
        preprocessed,
        model=extractor.model,
        save_path=output_path
    )
    
    return importance, top_indices

def main():
    """
    Main function to test the dynamic feature extractor.
    """
    # Initialize logger
    logger = get_logger(experiment_name="dynamic_feature_extractor_test")
    logger.logger.info("Starting dynamic feature extractor test")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test Dynamic Feature Extractor with optical flow sequences")
    parser.add_argument("--data_dir", type=str, default="data/optical_flow",
                        help="Directory containing optical flow data")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_feature_extractor/test",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to process")
    args = parser.parse_args()
    
    # Initialize feature extractor
    feature_extractor = DynamicFeatureExtractor()
    logger.logger.info("Feature extractor initialized")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find optical flow sequences
    flow_files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('_flow_raw.npy'):
                flow_files.append(os.path.join(root, file))
    
    if not flow_files:
        logger.logger.error(f"No optical flow files found in {args.data_dir}")
        return
    
    logger.logger.info(f"Found {len(flow_files)} optical flow sequences")
    
    # Select a few random flow files for testing
    np.random.seed(42)  # For reproducibility
    selected_files = np.random.choice(flow_files, min(args.num_samples, len(flow_files)), replace=False)
    
    # Process each selected file
    for flow_file in selected_files:
        logger.logger.info(f"Processing flow file: {flow_file}")
        
        # Load optical flow sequence
        flow_sequence = load_optical_flow_sequence(flow_file)
        
        if flow_sequence is None:
            continue
        
        logger.logger.info(f"Flow sequence shape: {flow_sequence.shape}")
        
        # Create base filename for outputs
        base_filename = os.path.basename(flow_file).split('.')[0]
        
        # Visualize the optical flow sequence
        flow_viz_path = os.path.join(args.output_dir, f"{base_filename}_flow_visualization.png")
        visualize_optical_flow_sequence(flow_sequence, save_path=flow_viz_path)
        logger.logger.info(f"Flow visualization saved to {flow_viz_path}")
        
        # Preprocess the sequence for feature extraction
        if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
            flow_batch = np.expand_dims(flow_sequence, 0)
        else:
            flow_batch = flow_sequence
        
        preprocessed = feature_extractor.preprocess_flow_sequence(flow_batch)
        
        # Extract features
        features = feature_extractor.extract_features(preprocessed)
        logger.logger.info(f"Extracted features shape: {features.shape}")
        
        # Visualize feature activations
        activations_path = os.path.join(args.output_dir, f"{base_filename}_activations.png")
        visualize_feature_activations(feature_extractor, flow_sequence, output_path=activations_path)
        logger.logger.info(f"Activations visualization saved to {activations_path}")
        
        # Analyze feature importance
        importance_path = os.path.join(args.output_dir, f"{base_filename}_importance.png")
        importance, top_indices = analyze_feature_importance(feature_extractor, flow_sequence, output_path=importance_path)
        logger.logger.info(f"Feature importance visualization saved to {importance_path}")
        logger.logger.info(f"Top important frames: {top_indices}")
    
    logger.logger.info("Dynamic feature extractor test completed")

if __name__ == "__main__":
    main()
