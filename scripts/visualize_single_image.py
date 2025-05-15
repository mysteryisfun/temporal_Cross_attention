import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.static_feature_extractor.feature_extractor import StaticFeatureExtractor
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def visualize_features_on_image(image_path, output_path=None, model_path=None):
    """
    Generate a visualization of CNN features overlaid on the input face image.
    
    Args:
        image_path (str): Path to the input face image
        output_path (str): Path to save the visualization (if None, will show the image)
        model_path (str): Path to a saved model (if None, will create a new model)
    
    Returns:
        numpy.ndarray: Visualization image
    """
    # Initialize feature extractor
    feature_extractor = StaticFeatureExtractor()
    
    if model_path and os.path.exists(model_path):
        feature_extractor.load_model(model_path)
        print(f"Loaded model from {model_path}")
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    original_img = cv2.imread(image_path)
    
    # Resize original image for display
    display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load for model processing
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    preprocessed_img = feature_extractor.preprocess_image(img_array)
    
    # Create a modified model that outputs the activation of the last convolutional layer
    base_model = feature_extractor.model
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        print("No convolutional layer found in the model")
        return
    
    print(f"Using layer {last_conv_layer.name} for visualization")
    
    # Create a model that outputs both the final features and the last conv layer activations
    activation_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    
    # Get activations and features
    last_conv_output, features = activation_model.predict(preprocessed_img)
    print(f"Feature shape: {features.shape}")
    print(f"Activation shape: {last_conv_output.shape}")
    
    # Create Class Activation Map (CAM)
    # Take average of all feature maps
    cam = np.mean(last_conv_output[0], axis=-1)
    
    # Normalize CAM
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    
    # Resize CAM to input image size
    original_height, original_width = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (original_width, original_height))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    alpha = 0.4
    overlay = np.uint8(display_img * (1 - alpha) + heatmap * alpha)
    
    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(display_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(heatmap)
    axs[1].set_title('Feature Activation Map')
    axs[1].axis('off')
    
    axs[2].imshow(overlay)
    axs[2].set_title('Overlay')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()
    else:
        plt.show()
    
    # Also create a detailed visualization of each feature map
    # This is optional and can be resource-intensive with many feature maps
    n_max_feature_maps = 16  # Limit to first 16 feature maps
    n_feature_maps = min(n_max_feature_maps, last_conv_output.shape[-1])
    
    # Create a grid layout
    grid_size = int(np.ceil(np.sqrt(n_feature_maps)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Flatten axes for easier indexing
    axs = axs.flatten()
    
    # Plot each feature map
    for i in range(n_feature_maps):
        # Get the ith feature map
        feature_map = last_conv_output[0, :, :, i]
        
        # Normalize for visualization
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-10)
        
        # Display
        axs[i].imshow(feature_map, cmap='viridis')
        axs[i].set_title(f'Feature Map {i+1}')
        axs[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n_feature_maps, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    
    # Save or show feature maps
    if output_path:
        feature_maps_path = os.path.splitext(output_path)[0] + "_feature_maps.png"
        plt.savefig(feature_maps_path)
        print(f"Feature maps visualization saved to {feature_maps_path}")
        plt.close()
    else:
        plt.show()
    
    return overlay

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize CNN features on a face image")
    parser.add_argument("--image_path", type=str, required=True,
                      help="Path to the input face image")
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to save the visualization")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to a saved model")
    args = parser.parse_args()
    
    # Generate visualization
    visualize_features_on_image(args.image_path, args.output_path, args.model_path)

if __name__ == "__main__":
    main()
