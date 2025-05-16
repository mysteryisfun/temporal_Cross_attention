import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import datetime
import json
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization, Activation
from keras.layers import AveragePooling3D, Dropout, Flatten, Dense
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.dynamic_feature_extractor.serialization import ModelSerializer

class DynamicFeatureExtractor:
    """
    Dynamic feature extractor based on 3D CNN architectures for optical flow sequences.
    Uses I3D (Inflated 3D ConvNet) for feature extraction from optical flow data.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the dynamic feature extractor with the specified configuration.
        
        Args:
            config_path (str): Path to the model configuration file.
                              If None, default configuration is used.
        """
        # Initialize logger
        self.logger = get_logger(experiment_name="dynamic_feature_extractor")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Build model
        self.model = self._build_model()
        
        # Log model architecture
        self.logger.log_model_architecture(self.model)
    
    def _load_config(self, config_path):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (str): Path to the model configuration file.
            
        Returns:
            dict: Configuration parameters.
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.logger.warning(f"Error loading configuration from {config_path}: {str(e)}")
                self.logger.logger.warning("Using default configuration")
        
        # Default configuration
        default_config = {
            'model_type': 'i3d',
            'input_shape': (16, 224, 224, 2),  # (frames, height, width, channels=2 for optical flow)
            'pretrained': True,
            'freeze_layers': [],
            'output_features': 1024,
            'dropout_rate': 0.5
        }
        self.logger.logger.info(f"Using default configuration: {default_config}")
        return default_config
    
    def _build_model(self):
        """
        Build the dynamic feature extraction model based on the configuration.
        This implementation is based on the I3D architecture for optical flow.
        
        Returns:
            tf.keras.Model: Feature extraction model.
        """
        # Get model parameters from config
        model_type = self.config.get('model_type', 'i3d').lower()
        input_shape = self.config.get('input_shape', (16, 224, 224, 2))
        pretrained = self.config.get('pretrained', True)
        freeze_layers = self.config.get('freeze_layers', [])
        output_features = self.config.get('output_features', 1024)
        dropout_rate = self.config.get('dropout_rate', 0.5)
        
        # Create input tensor
        input_tensor = Input(shape=input_shape, name='dynamic_input')
        
        if model_type == 'i3d':
            # Build I3D model for optical flow
            x = self._build_i3d_architecture(input_tensor)
            
            # Add final layers
            x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='avg_pool')(x)
            x = Dropout(dropout_rate, name='dropout')(x)
            x = Flatten(name='flatten')(x)
            
            # Create the final model
            model = Model(inputs=input_tensor, outputs=x, name='dynamic_feature_extractor')
            
            # Apply layer freezing if needed
            if freeze_layers and pretrained:
                for layer_idx in freeze_layers:
                    if 0 <= layer_idx < len(model.layers):
                        model.layers[layer_idx].trainable = False
                        self.logger.logger.info(f"Freezing layer {layer_idx}: {model.layers[layer_idx].name}")
        else:
            # Default to I3D if specified model is not supported
            self.logger.logger.warning(f"Model type {model_type} not supported, using I3D")
            # Recursively call with i3d model type
            return self._build_model()
        
        return model
    
    def _build_i3d_architecture(self, input_tensor):
        """
        Build the I3D architecture for optical flow input.
        
        Args:
            input_tensor (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor from the I3D backbone.
        """
        # I3D architecture as described in the paper
        # "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
        
        # Initial convolution
        x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same', name='conv1')(input_tensor)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu', name='relu1')(x)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='pool1')(x)
        
        # Inception block 2
        x = self._inception_block(x, [64, 96, 128, 16, 32, 32], name='inception_2a')
        x = self._inception_block(x, [128, 128, 192, 32, 96, 64], name='inception_2b')
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='pool2')(x)
        
        # Inception block 3
        x = self._inception_block(x, [192, 96, 208, 16, 48, 64], name='inception_3a')
        x = self._inception_block(x, [160, 112, 224, 24, 64, 64], name='inception_3b')
        x = self._inception_block(x, [128, 128, 256, 24, 64, 64], name='inception_3c')
        x = self._inception_block(x, [112, 144, 288, 32, 64, 64], name='inception_3d')
        x = self._inception_block(x, [256, 160, 320, 32, 128, 128], name='inception_3e')
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='pool3')(x)
        
        # Inception block 4
        x = self._inception_block(x, [256, 160, 320, 32, 128, 128], name='inception_4a')
        x = self._inception_block(x, [384, 192, 384, 48, 128, 128], name='inception_4b')
        
        return x
    
    def _inception_block(self, x, filters, name):
        """
        Create an Inception block for the I3D model.
        
        Args:
            x (tf.Tensor): Input tensor.
            filters (list): List of filter sizes for the different branches.
            name (str): Base name for the inception block.
            
        Returns:
            tf.Tensor: Output tensor from the inception block.
        """
        # Branch 1: 1x1x1 convolution
        branch1 = Conv3D(filters[0], (1, 1, 1), padding='same', name=f'{name}_b1_conv')(x)
        branch1 = BatchNormalization(name=f'{name}_b1_bn')(branch1)
        branch1 = Activation('relu', name=f'{name}_b1_relu')(branch1)
        
        # Branch 2: 1x1x1 -> 3x3x3 convolutions
        branch2 = Conv3D(filters[1], (1, 1, 1), padding='same', name=f'{name}_b2_conv1')(x)
        branch2 = BatchNormalization(name=f'{name}_b2_bn1')(branch2)
        branch2 = Activation('relu', name=f'{name}_b2_relu1')(branch2)
        branch2 = Conv3D(filters[2], (3, 3, 3), padding='same', name=f'{name}_b2_conv2')(branch2)
        branch2 = BatchNormalization(name=f'{name}_b2_bn2')(branch2)
        branch2 = Activation('relu', name=f'{name}_b2_relu2')(branch2)
        
        # Branch 3: 1x1x1 -> 3x3x3 convolutions
        branch3 = Conv3D(filters[3], (1, 1, 1), padding='same', name=f'{name}_b3_conv1')(x)
        branch3 = BatchNormalization(name=f'{name}_b3_bn1')(branch3)
        branch3 = Activation('relu', name=f'{name}_b3_relu1')(branch3)
        branch3 = Conv3D(filters[4], (3, 3, 3), padding='same', name=f'{name}_b3_conv2')(branch3)
        branch3 = BatchNormalization(name=f'{name}_b3_bn2')(branch3)
        branch3 = Activation('relu', name=f'{name}_b3_relu2')(branch3)
        
        # Branch 4: 3x3x3 max pooling -> 1x1x1 convolution
        branch4 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name=f'{name}_b4_pool')(x)
        branch4 = Conv3D(filters[5], (1, 1, 1), padding='same', name=f'{name}_b4_conv')(branch4)
        branch4 = BatchNormalization(name=f'{name}_b4_bn')(branch4)
        branch4 = Activation('relu', name=f'{name}_b4_relu')(branch4)
          # Concatenate all branches
        x = tf.keras.layers.Concatenate(axis=-1, name=f'{name}_concat')([branch1, branch2, branch3, branch4])
        
        return x
    
    def extract_features(self, flow_sequences):
        """
        Extract features from optical flow sequences.
        
        Args:
            flow_sequences (np.ndarray): Batch of optical flow sequences with shape 
                                        [batch_size, frames, height, width, channels].
                                        Should be preprocessed appropriately.
        
        Returns:
            np.ndarray: Extracted features with shape [batch_size, output_features].
        """
        features = self.model.predict(flow_sequences)
        return features

    def preprocess_flow_sequence(self, flow_sequence):
        """
        Preprocess a single optical flow sequence for feature extraction.
        
        Args:
            flow_sequence (np.ndarray): Optical flow sequence with shape 
                                       [frames, height, width, channels].
        
        Returns:
            np.ndarray: Preprocessed flow sequence with shape [1, frames, height, width, channels].
        """
        # Standardize the flow values
        flow_sequence = self._standardize_flow(flow_sequence)
        # Add batch dimension if not present
        if len(flow_sequence.shape) == 4:
            flow_sequence = np.expand_dims(flow_sequence, 0)
        return flow_sequence
        
    def _standardize_flow(self, flow_sequence):
        """
        Standardize optical flow values to a consistent range.
        
        Args:
            flow_sequence (np.ndarray): Optical flow sequence.
            
        Returns:
            np.ndarray: Standardized flow sequence.
        """
        # Optical flow values are typically in the range [-20, 20]
        # We scale them to [-1, 1] for better conditioning
        return np.clip(flow_sequence / 20.0, -1.0, 1.0)
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model.
            
        Returns:
            str: Path to the saved model directory.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        self.logger.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def save_model_with_serializer(self, base_dir, metadata=None, include_optimizer=True, save_format='tf'):
        """
        Save the model using the ModelSerializer class.
        
        Args:
            base_dir (str): Base directory where models are stored.
            metadata (dict): Additional metadata to save with the model.
            include_optimizer (bool): Whether to include optimizer state.
            save_format (str): Format to save the model ('tf' or 'h5').
            
        Returns:
            str: Path to the saved model directory.
        """
        from models.dynamic_feature_extractor.serialization import ModelSerializer
        
        # Create a serializer instance
        serializer = ModelSerializer(model=self.model)
          # Add model parameters to metadata
        model_metadata = {
            "input_shape": self.config.get('input_shape'),
            "model_type": self.config.get('model_type'),
            "dropout_rate": self.config.get('dropout_rate')
        }
        
        # Combine with user-provided metadata
        if metadata:
            model_metadata.update(metadata)
        
        # Save the model with the serializer
        save_path = serializer.save_model(
            base_dir=base_dir,
            include_optimizer=include_optimizer,
            save_format=save_format,
            metadata=model_metadata
        )
        
        if save_path:
            self.logger.logger.info(f"Model saved with serializer to {save_path}")
        else:
            self.logger.logger.error("Failed to save model with serializer")
        
        return save_path
    
    def load_model(self, filepath):
        """
        Load the model from disk.
        
        Args:
            filepath (str): Path to load the model from.
            
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            self.logger.logger.warning(f"Model file not found: {filepath}")
            return False
            
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.logger.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.logger.error(f"Error loading model from {filepath}: {str(e)}")
            return False
    
    def load_pretrained_weights(self, weights_path=None):
        """
        Load pretrained I3D weights.
        
        Args:
            weights_path (str): Path to the pretrained weights.
                                If None, attempts to download from a predefined URL.
                                
        Returns:
            bool: True if weights were loaded successfully, False otherwise.
        """
        # If weights_path is provided and exists, load the weights
        if weights_path and os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                self.logger.logger.info(f"Loaded pretrained weights from {weights_path}")
                return True
            except Exception as e:
                self.logger.logger.error(f"Error loading weights from {weights_path}: {str(e)}")
                return False
        
        # Otherwise, attempt to download standard pretrained weights
        # This is a simplified version - in a real implementation, you would 
        # download from TF model garden or equivalent
        self.logger.logger.warning("Pretrained weights path not provided or not found.")
        self.logger.logger.info("Training from scratch. For best results, provide pretrained weights.")
        return False


if __name__ == "__main__":
    # Simple test
    extractor = DynamicFeatureExtractor()
    print("Model initialized successfully")
