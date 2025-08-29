"""
Feature Map Visualization Tool

This script provides utilities to visualize feature maps from any Keras model layer.
- Extracts and plots feature maps for a given input
- Supports saving visualizations for analysis

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python .\scripts\visualize_feature_maps.py
    }
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_maps(model, layer_name, input_data, save_dir=None, max_maps=8):
    """
    Visualizes feature maps from a specific layer for a given input.
    Args:
        model: Keras model
        layer_name: Name of the layer to visualize
        input_data: Input tensor (batch_size, ...)
        save_dir: Directory to save the plots (optional)
        max_maps: Maximum number of feature maps to plot
    """
    # Create a sub-model to output the feature maps
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = feature_model.predict(input_data)
    if feature_maps.ndim == 4:
        # For conv layers: (batch, height, width, channels)
        maps = feature_maps[0]
        num_maps = min(maps.shape[-1], max_maps)
        plt.figure(figsize=(15, 5))
        for i in range(num_maps):
            plt.subplot(1, num_maps, i+1)
            plt.imshow(maps[..., i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Feature {i}')
        plt.suptitle(f'Feature Maps from {layer_name}')
        if save_dir:
            plt.savefig(f'{save_dir}/feature_maps_{layer_name}.png')
        plt.show()
    else:
        print(f'Feature maps from {layer_name} are not 4D, shape: {feature_maps.shape}')

def main():
    # Example: Visualize feature maps from a simple model (replace with actual project model)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv1'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', name='conv2'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5)
    ])
    dummy_input = np.random.rand(1, 32, 32, 3)
    visualize_feature_maps(model, 'conv1', dummy_input)
    visualize_feature_maps(model, 'conv2', dummy_input)

if __name__ == "__main__":
    main()
