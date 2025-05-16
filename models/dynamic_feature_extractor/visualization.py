import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.models import Model
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import io
from PIL import Image
import imageio

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

class FeatureVisualization:
    """
    Utility class for visualizing 3D CNN features and activation maps from optical flow sequences.
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
    
    def get_intermediate_activations(self, flow_sequence):
        """
        Get activations from a specific layer for a given optical flow sequence.
        
        Args:
            flow_sequence (numpy.ndarray): Input optical flow sequence with shape [1, frames, height, width, channels].
            
        Returns:
            numpy.ndarray: Layer activations.
        """
        if self.intermediate_model is None:
            raise ValueError("Model and layer_name must be provided during initialization")
        
        activations = self.intermediate_model.predict(flow_sequence)
        return activations
    
    def visualize_layer_activations(self, activations, frame_idx=0, num_features=64, figsize=(12, 12), save_path=None):
        """
        Visualize layer activations for a specific frame in the sequence.
        
        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            frame_idx (int): Index of the frame to visualize.
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
        
        # Get the activations for the specified frame
        act = activations[0, frame_idx] if len(activations.shape) > 4 else activations[0]
        
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
    
    def visualize_temporal_activation(self, activations, feature_idx=0, save_path=None):
        """
        Visualize how a specific feature activates over time.
        
        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_idx (int): Index of the feature to visualize.
            save_path (str): Path to save the visualization.
        """
        # Get the number of frames
        num_frames = activations.shape[1]
        
        # Create a figure with subplots for each frame
        fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))
        
        if num_frames == 1:
            axes = [axes]  # Make iterable for single frame case
        
        # For each frame, visualize the selected feature map
        for i in range(num_frames):
            feature_map = activations[0, i, :, :, feature_idx]
            
            # Normalize feature map
            feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
        
        plt.suptitle(f"Temporal Activation of Feature {feature_idx}")
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_flow_and_activations(self, flow_sequence, activations, frame_indices=None, save_path=None):
        """
        Visualize optical flow and corresponding feature activations.
        
        Args:
            flow_sequence (numpy.ndarray): Optical flow sequence with shape [1, frames, height, width, channels].
            activations (numpy.ndarray): Layer activations.
            frame_indices (list): Indices of frames to visualize. If None, uses all frames (up to 10).
            save_path (str): Path to save the visualization.
        """
        # Get flow sequence without batch dimension
        flow = flow_sequence[0]
        
        # Determine frames to visualize
        if frame_indices is None:
            num_frames = min(flow.shape[0], 10)  # Limit to 10 frames
            frame_indices = list(range(num_frames))
        else:
            num_frames = len(frame_indices)
        
        # Create a figure with 2 rows: flow and activations
        fig, axes = plt.subplots(2, num_frames, figsize=(4 * num_frames, 8))
        
        for i, frame_idx in enumerate(frame_indices):
            # Visualize optical flow
            flow_frame = flow[frame_idx]
            flow_rgb = self._flow_to_rgb(flow_frame)
            axes[0, i].imshow(flow_rgb)
            axes[0, i].set_title(f"Flow {frame_idx}")
            axes[0, i].axis('off')
            
            # Visualize feature activation (average over all channels)
            if len(activations.shape) > 4:
                # Activations with temporal dimension [1, frames, height, width, channels]
                act = activations[0, frame_idx]
            else:
                # Activations without temporal dimension [1, height, width, channels]
                act = activations[0]
            
            # Average over all channels
            avg_act = np.mean(act, axis=-1)
            
            # Normalize
            avg_act = (avg_act - np.min(avg_act)) / (np.max(avg_act) - np.min(avg_act) + 1e-8)
            
            axes[1, i].imshow(avg_act, cmap='viridis')
            axes[1, i].set_title(f"Activation {frame_idx}")
            axes[1, i].axis('off')
        
        plt.suptitle("Optical Flow and Feature Activations")
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def _flow_to_rgb(self, flow):
        """
        Convert optical flow to RGB visualization.
        
        Args:
            flow (numpy.ndarray): Optical flow with shape [height, width, 2].
            
        Returns:
            numpy.ndarray: RGB visualization of the flow.
        """
        # Extract u and v components
        u, v = flow[:, :, 0], flow[:, :, 1]
        
        # Calculate magnitude and angle
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) + np.pi
        
        # Normalize magnitude for visualization
        magnitude = np.clip(magnitude, 0, 1)
        
        # Convert to HSV
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / (2 * np.pi)  # Hue based on flow direction
        hsv[..., 1] = 255                        # Full saturation
        hsv[..., 2] = np.clip(magnitude * 255, 0, 255).astype(np.uint8)  # Value based on flow magnitude
        
        # Convert HSV to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return rgb
    
    def visualize_temporal_feature_evolution(self, activations, feature_indices=None, save_path=None):
        """
        Visualize how selected features evolve over time across all frames.

        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_indices (list): Indices of features to visualize. If None, selects first 5 features.
            save_path (str): Path to save the visualization.
        """
        # Ensure we have temporal activations
        if len(activations.shape) != 5:
            raise ValueError("Activations must have shape [1, frames, height, width, channels]")

        # Get dimensions
        num_frames = activations.shape[1]
        num_features = activations.shape[-1]

        # Select features to visualize
        if feature_indices is None:
            feature_indices = list(range(min(5, num_features)))

        # Create figure
        fig, axes = plt.subplots(len(feature_indices), 1, figsize=(12, 3 * len(feature_indices)))
        if len(feature_indices) == 1:
            axes = [axes]

        # For each selected feature
        for i, feature_idx in enumerate(feature_indices):
            # Calculate mean activation across spatial dimensions for each frame
            mean_activations = np.mean(activations[0, :, :, :, feature_idx], axis=(1, 2))

            # Plot temporal evolution
            axes[i].plot(range(num_frames), mean_activations, marker='o', linestyle='-', linewidth=2)
            axes[i].set_title(f"Feature {feature_idx} Temporal Evolution")
            axes[i].set_xlabel("Frame")
            axes[i].set_ylabel("Mean Activation")
            axes[i].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_3d_feature_activation(self, activations, feature_idx=0, save_path=None):
        """
        Create a 3D visualization of a feature's activation over time and space.

        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_idx (int): Index of the feature to visualize.
            save_path (str): Path to save the visualization.
        """
        # Ensure we have temporal activations
        if len(activations.shape) != 5:
            raise ValueError("Activations must have shape [1, frames, height, width, channels]")

        # Extract the feature activations
        feature_act = activations[0, :, :, :, feature_idx]

        # Create a 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for 3D plotting
        num_frames, height, width = feature_act.shape
        x, y = np.meshgrid(range(width), range(height))

        # Plot a selection of frames in 3D space
        num_frames_to_plot = min(num_frames, 8)  # Limit number of frames to avoid clutter
        frame_indices = np.linspace(0, num_frames-1, num_frames_to_plot, dtype=int)

        # Use a colormap
        cmap = plt.cm.viridis

        for i, frame_idx in enumerate(frame_indices):
            z = frame_idx * np.ones_like(x)

            # Get activations for this frame
            activations_frame = feature_act[frame_idx]

            # Normalize activations for coloring
            norm_act = (activations_frame - np.min(activations_frame)) / (np.max(activations_frame) - np.min(activations_frame) + 1e-8)

            # Plot the surface
            surf = ax.plot_surface(x, y, z, facecolors=cmap(norm_act),
                                 alpha=0.7, rstride=2, cstride=2)

        # Set labels and title
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Frame')
        ax.set_title(f'3D Visualization of Feature {feature_idx} Activation')

        # Add a colorbar
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Normalized Activation')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_feature_evolution_animation(self, activations, feature_idx=0, save_path=None):
        """
        Create an animation showing how a feature evolves over time.

        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_idx (int): Index of the feature to visualize.
            save_path (str): Path to save the animation as a gif.
        """
        # Ensure we have temporal activations
        if len(activations.shape) != 5:
            raise ValueError("Activations must have shape [1, frames, height, width, channels]")

        # Extract feature activations
        feature_act = activations[0, :, :, :, feature_idx]
        num_frames = feature_act.shape[0]

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Normalize across all frames for consistent color mapping
        vmin = np.min(feature_act)
        vmax = np.max(feature_act)

        # Function to update the figure for each frame
        def update(frame):
            ax.clear()
            im = ax.imshow(feature_act[frame], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Feature {feature_idx}, Frame {frame}')
            ax.axis('off')
            return [im]

        # Create the animation
        anim = FuncAnimation(fig, update, frames=range(num_frames), blit=True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save as gif
            anim.save(save_path, writer='pillow', fps=5)
            plt.close()
        else:
            plt.show()

    def create_comparative_temporal_view(self, activations, feature_indices=None,
                                         flow_sequence=None, save_path=None):
        """
        Create a comparative view of multiple features over time, optionally with original flow.

        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_indices (list): Indices of features to visualize. If None, selects first 3 features.
            flow_sequence (numpy.ndarray): Original flow sequence for comparison.
            save_path (str): Path to save the visualization.
        """
        # Ensure we have temporal activations
        if len(activations.shape) != 5:
            raise ValueError("Activations must have shape [1, frames, height, width, channels]")

        # Get dimensions
        num_frames = activations.shape[1]
        num_features = activations.shape[-1]

        # Select features to visualize
        if feature_indices is None:
            feature_indices = list(range(min(3, num_features)))

        # Determine number of rows in the plot
        num_rows = len(feature_indices) + (1 if flow_sequence is not None else 0)

        # Create figure
        fig, axes = plt.subplots(num_rows, num_frames,
                                figsize=(2 * num_frames, 2 * num_rows))

        # If we only have one frame, ensure axes is 2D
        if num_frames == 1:
            axes = axes.reshape(num_rows, 1)

        # Plot flow sequence if provided
        row_idx = 0
        if flow_sequence is not None:
            for frame_idx in range(num_frames):
                flow_frame = flow_sequence[0, frame_idx]
                flow_rgb = self._flow_to_rgb(flow_frame)
                axes[row_idx, frame_idx].imshow(flow_rgb)
                axes[row_idx, frame_idx].set_title(f"Flow {frame_idx}")
                axes[row_idx, frame_idx].axis('off')
            row_idx += 1

        # Plot feature activations
        for i, feature_idx in enumerate(feature_indices):
            for frame_idx in range(num_frames):
                # Get feature activation for this frame
                feature_act = activations[0, frame_idx, :, :, feature_idx]

                # Normalize for visualization
                norm_act = (feature_act - np.min(feature_act)) / (np.max(feature_act) - np.min(feature_act) + 1e-8)

                # Plot
                axes[row_idx + i, frame_idx].imshow(norm_act, cmap='viridis')
                axes[row_idx + i, frame_idx].set_title(f"F{feature_idx}, T{frame_idx}")
                axes[row_idx + i, frame_idx].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_feature_activation_heatmap(self, activations, feature_idx=0, save_path=None):
        """
        Create a heatmap visualization of a feature's activation over time and space.
        
        Args:
            activations (numpy.ndarray): Layer activations with shape [1, frames, height, width, channels].
            feature_idx (int): Index of the feature to visualize.
            save_path (str): Path to save the visualization.
        """
        # Ensure we have temporal activations
        if len(activations.shape) != 5:
            raise ValueError("Activations must have shape [1, frames, height, width, channels]")
            
        # Extract the feature activations
        feature_act = activations[0, :, :, :, feature_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create temporal average of activations
        avg_activation = np.mean(feature_act, axis=0)
        
        # Normalize for visualization
        norm_act = (avg_activation - np.min(avg_activation)) / (np.max(avg_activation) - np.min(avg_activation) + 1e-8)
        
        # Create heatmap
        im = ax.imshow(norm_act, cmap='hot', interpolation='nearest')
        ax.set_title(f'Temporal Average Activation Heatmap for Feature {feature_idx}')
        
        # Add colorbar
        fig.colorbar(im, ax=ax, label='Normalized Activation')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def visualize_optical_flow_sequence(flow_sequence, save_path=None):
    """
    Visualize an optical flow sequence.
    
    Args:
        flow_sequence (numpy.ndarray): Optical flow sequence with shape [frames, height, width, channels].
        save_path (str): Path to save the visualization.
    """
    # Create a FeatureVisualization instance for using its _flow_to_rgb method
    vis = FeatureVisualization()
    
    # Determine number of frames
    num_frames = flow_sequence.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
    
    if num_frames == 1:
        axes = [axes]  # Make iterable for single frame case
    
    # Visualize each frame
    for i in range(num_frames):
        flow_frame = flow_sequence[i]
        flow_rgb = vis._flow_to_rgb(flow_frame)
        axes[i].imshow(flow_rgb)
        axes[i].set_title(f"Flow {i}")
        axes[i].axis('off')
    
    plt.suptitle("Optical Flow Sequence")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
