import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

class FeatureVisualization:
    """
    Utility class for visualizing CNN features and activation maps.
    """
    
    def __init__(self, model=None, layer_name=None):
        """
        Initialize the visualization tool.
        
        Args:
            model (tf.keras.Model): The model to visualize features from.
            layer_name (str): Name of the layer to extract features from.
        """
        self.model = model
        self.layer_name = layer_name
        
        # If a specific layer is provided, create a new model that outputs that layer's activations
        if model is not None and layer_name is not None:
            self.intermediate_model = Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
        else:
            self.intermediate_model = None
    
    def get_intermediate_activations(self, image):
        """
        Get activations from a specific layer for a given image.
        
        Args:
            image (numpy.ndarray): Input image with shape [1, height, width, channels].
            
        Returns:
            numpy.ndarray: Layer activations.
        """
        if self.intermediate_model is None:
            raise ValueError("Model and layer_name must be provided during initialization")
        
        activations = self.intermediate_model.predict(image)
        return activations
    
    def visualize_layer_activations(self, activations, num_features=64, figsize=(12, 12), save_path=None):
        """
        Visualize layer activations as a grid of feature maps.
        
        Args:
            activations (numpy.ndarray): Layer activations.
            num_features (int): Number of feature maps to visualize.
            figsize (tuple): Figure size.
            save_path (str): Path to save the visualization.
        """
        # Ensure we don't visualize more features than we have
        num_features = min(num_features, activations.shape[-1])
        
        # Create a grid layout
        grid_size = int(np.ceil(np.sqrt(num_features)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
        # Flatten the axes array for easier indexing
        axes = axes.flatten()
        
        # Get the first image's activations (assuming batch size of 1)
        act = activations[0]
        
        # Plot each feature map
        for i in range(num_features):
            feature_map = act[:, :, i]
            
            # Normalize feature map
            feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f"Filter {i}")
            axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_class_activation_map(self, image, class_idx=None, layer_name=None):
        """
        Create a Class Activation Map (CAM) for a given image.
        
        Args:
            image (numpy.ndarray): Input image with shape [1, height, width, channels].
            class_idx (int): Index of the class to visualize. If None, uses the predicted class.
            layer_name (str): Name of the layer to use for CAM. If None, uses the last convolutional layer.
            
        Returns:
            numpy.ndarray: Class activation map.
        """
        if self.model is None:
            raise ValueError("Model must be provided during initialization")
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        # Create a model that outputs both the predictions and the activations of the target layer
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass
            conv_output, predictions = grad_model(image)
            
            # Use the predicted class index if none is provided
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the prediction for the specified class
            class_output = predictions[:, class_idx]
        
        # Compute gradients of the class output with respect to the convolutional layer output
        grads = tape.gradient(class_output, conv_output)
        
        # Pool the gradients over all axes except the channel axis
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by its importance
        conv_output = conv_output[0]
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        # Post-process the CAM
        cam = tf.maximum(cam, 0)  # ReLU
        cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + tf.keras.backend.epsilon())
        cam = cam.numpy()
        
        return cam
    
    def overlay_cam_on_image(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET, save_path=None):
        """
        Overlay a Class Activation Map on an image.
        
        Args:
            image (numpy.ndarray): Original image without preprocessing.
            cam (numpy.ndarray): Class activation map.
            alpha (float): Transparency factor for the overlay.
            colormap: OpenCV colormap for the heatmap.
            save_path (str): Path to save the visualization.
            
        Returns:
            numpy.ndarray: Image with CAM overlay.
        """
        # Resize CAM to match image dimensions
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        
        # Convert BGR to RGB for matplotlib if using OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
        
        # Superimpose the heatmap on the original image
        superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
        return superimposed_img


def visualize_model_filters(model, layer_name, filter_index, input_shape=(224, 224, 3), num_iterations=30, save_path=None):
    """
    Visualize what patterns activate a specific filter in a convolutional layer.
    
    Args:
        model (tf.keras.Model): The model containing the layer.
        layer_name (str): Name of the convolutional layer.
        filter_index (int): Index of the filter to visualize.
        input_shape (tuple): Shape of the input image (height, width, channels).
        num_iterations (int): Number of gradient ascent iterations.
        save_path (str): Path to save the visualization.
        
    Returns:
        numpy.ndarray: Pattern that maximally activates the filter.
    """
    # Create a model that outputs the target layer's activations
    layer_output = model.get_layer(layer_name).output
    loss = tf.reduce_mean(layer_output[:, :, :, filter_index])
    
    # Create a gradient ascent process
    @tf.function
    def compute_loss_and_grads(img):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss_value = loss(model(img))
        grads = tape.gradient(loss_value, img)
        # Normalize gradients
        grads = tf.math.l2_normalize(grads)
        return loss_value, grads
    
    # Start from a gray image with some noise
    img = tf.random.uniform((1,) + input_shape, minval=0.4, maxval=0.6)
    
    # Perform gradient ascent
    step_size = 1.0
    for i in range(num_iterations):
        loss_value, grads = compute_loss_and_grads(img)
        img += grads * step_size
        # Clip image to valid range [0, 1]
        img = tf.clip_by_value(img, 0, 1)
    
    # Convert to numpy and post-process for visualization
    img = img[0].numpy()
    img = (img - img.mean()) / (img.std() + 1e-8) * 0.1 + 0.5
    img = np.clip(img, 0, 1)
    
    if save_path:
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Filter {filter_index} in layer {layer_name}")
        plt.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    
    return img
