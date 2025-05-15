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

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.static_feature_extractor.feature_extractor import StaticFeatureExtractor
from utils.logger import get_logger

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for feature extraction.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing (height, width).
        
    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Load image and convert to numpy array
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    
    return img_array

def visualize_features(features, num_features=64, figsize=(10, 10)):
    """
    Visualize extracted features as a grid of heatmaps.
    
    Args:
        features (numpy.ndarray): Features with shape [batch_size, feature_dim].
        num_features (int): Number of features to visualize.
        figsize (tuple): Figure size.
    """
    # Ensure we don't try to visualize more features than we have
    num_features = min(num_features, features.shape[1])
    
    # Create a grid layout
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Normalize features for better visualization
    norm_features = (features[0] - np.min(features[0])) / (np.max(features[0]) - np.min(features[0]) + 1e-8)
    
    # Plot each feature
    for i in range(num_features):
        axes[i].imshow([[norm_features[i]]], cmap='viridis')
        axes[i].set_title(f"Feature {i}")
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(os.path.join("results", "visualizations"), exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", "feature_visualization.png"))
    plt.close()

def visualize_attention_on_image(image, features, output_path):
    """
    Create a heatmap visualization of features attention on the original image.
    This is a simplified visualization and not actual attention.
    
    Args:
        image (numpy.ndarray): Original image.
        features (numpy.ndarray): Extracted features.
        output_path (str): Path to save the visualization.
    """
    # Normalize features to [0, 1] range for visualization
    feature_sum = np.sum(np.abs(features[0]))
    normalized_sum = feature_sum / np.max(feature_sum) if np.max(feature_sum) > 0 else feature_sum
    
    # Resize to image dimensions
    heatmap = cv2.resize(np.array([normalized_sum]), (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, superimposed_img)

def main():
    """
    Main function to test the static feature extractor.
    """
    # Initialize logger
    logger = get_logger(experiment_name="feature_extractor_test")
    logger.logger.info("Starting feature extractor test")
    
    # Initialize feature extractor
    feature_extractor = StaticFeatureExtractor()
    logger.logger.info("Feature extractor initialized")
    
    # Create output directories
    os.makedirs(os.path.join("results", "visualizations"), exist_ok=True)
    
    # Load sample face images
    faces_dir = os.path.join("data", "faces")
    
    # Get a list of all face directories
    face_dirs = next(os.walk(faces_dir))[1]
    
    if not face_dirs:
        logger.logger.error(f"No face directories found in {faces_dir}")
        return
    
    # Select a few random face directories for testing
    np.random.seed(42)  # For reproducibility
    selected_dirs = np.random.choice(face_dirs, min(5, len(face_dirs)), replace=False)
    
    for face_dir in selected_dirs:
        face_dir_path = os.path.join(faces_dir, face_dir)
        
        # Get all images in the face directory
        images = glob.glob(os.path.join(face_dir_path, "*.jpg"))
        
        if not images:
            logger.logger.warning(f"No images found in {face_dir_path}")
            continue
        
        # Process all images in the directory
        for img_path in images:
            logger.logger.info(f"Processing image: {img_path}")
            
            # Load and preprocess image
            img = load_and_preprocess_image(img_path)
            
            # Create a copy of the original image for visualization
            original_img = cv2.imread(img_path)
            
            # Preprocess image for the model
            preprocessed_img = feature_extractor.preprocess_image(img)
            
            # Extract features
            features = feature_extractor.extract_features(preprocessed_img)
            
            logger.logger.info(f"Extracted features shape: {features.shape}")
            
            # Visualize features
            output_path = os.path.join("results", "visualizations", f"{os.path.basename(img_path)}_features.jpg")
            visualize_attention_on_image(original_img, features, output_path)
            
            logger.logger.info(f"Visualization saved to {output_path}")
            
            # Only process one image per directory for brevity
            break
    
    # Also visualize feature activations for the last processed image
    visualize_features(features)
    logger.logger.info("Feature visualization completed")
    
    logger.logger.info("Static feature extractor test completed successfully")

if __name__ == "__main__":
    main()
