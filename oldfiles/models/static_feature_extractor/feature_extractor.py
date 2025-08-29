import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Input
import yaml
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger

class StaticFeatureExtractor:
    """
    Static feature extractor based on 2D CNN architectures.
    Uses pre-trained models (ResNet50, VGG16, InceptionV3) for feature extraction from face images.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the static feature extractor with the specified configuration.
        
        Args:
            config_path (str): Path to the model configuration file.
                               If None, default configuration is used.
        """
        # Initialize logger
        self.logger = get_logger(experiment_name="static_feature_extractor")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Build model
        self.model = self._build_model()
        
        # Log model architecture
        self.logger.log_model_architecture(self.model)
    
    def _load_config(self, config_path):
        """
        Load model configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: Model configuration.
        """
        if config_path is None:
            config_path = os.path.join("config", "model_config", "model_config.yaml")
        
        self.logger.logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Extract just the static_cnn section
                static_config = config.get('static_cnn', {})
                self.logger.logger.info(f"Configuration loaded: {static_config}")
                return static_config
        except Exception as e:
            self.logger.logger.error(f"Error loading configuration: {str(e)}")
            # Default configuration
            default_config = {
                'base_model': 'resnet50',
                'pretrained': True,
                'freeze_layers': [0, 1, 2, 3],
                'output_features': 2048,
                'pooling': 'avg'
            }
            self.logger.logger.info(f"Using default configuration: {default_config}")
            return default_config
    
    def _build_model(self):
        """
        Build the static feature extraction model based on the configuration.
        
        Returns:
            tf.keras.Model: Feature extraction model.
        """
        # Get model parameters from config
        base_model_name = self.config.get('base_model', 'resnet50').lower()
        pretrained = self.config.get('pretrained', True)
        freeze_layers = self.config.get('freeze_layers', [])
        output_features = self.config.get('output_features', 2048)
        pooling_type = self.config.get('pooling', 'avg')
        
        weights = 'imagenet' if pretrained else None
        input_shape = (224, 224, 3)  # Standard input size for most pre-trained models
        
        # Create input tensor
        input_tensor = Input(shape=input_shape, name='static_input')
        
        # Select base model based on configuration
        if base_model_name == 'resnet50':
            base_model = ResNet50(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                input_shape=input_shape,
                pooling=None
            )
            self.logger.logger.info("Using ResNet50 as base model")
        else:
            # Default to ResNet50 if specified model is not supported yet
            self.logger.logger.warning(f"Base model {base_model_name} not supported, using ResNet50 instead")
            base_model = ResNet50(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor,
                input_shape=input_shape,
                pooling=None
            )
        
        # Freeze layers if specified
        if freeze_layers and pretrained:
            for layer_idx in freeze_layers:
                if 0 <= layer_idx < len(base_model.layers):
                    base_model.layers[layer_idx].trainable = False
                    self.logger.logger.info(f"Freezing layer {layer_idx}: {base_model.layers[layer_idx].name}")
        
        # Select pooling layer based on configuration
        x = base_model.output
        if pooling_type == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            self.logger.logger.info("Using Global Average Pooling")
        elif pooling_type == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)
            self.logger.logger.info("Using Global Max Pooling")
        
        # Create the final model
        model = Model(inputs=input_tensor, outputs=x, name='static_feature_extractor')
        
        return model
    
    def extract_features(self, images):
        """
        Extract features from face images.
        
        Args:
            images (tf.Tensor or numpy.ndarray): Batch of images with shape [batch_size, height, width, channels].
                                                Images should be preprocessed appropriately.
        
        Returns:
            numpy.ndarray: Extracted features with shape [batch_size, output_features].
        """
        features = self.model.predict(images)
        return features
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for feature extraction.
        
        Args:
            image (tf.Tensor or numpy.ndarray): Input image with shape [height, width, channels].
        
        Returns:
            tf.Tensor: Preprocessed image with shape [1, height, width, channels].
        """
        if hasattr(tf.keras.applications, 'resnet50'):
            # Use the appropriate preprocessing function based on the base model
            if self.config.get('base_model', '').lower() == 'resnet50':
                preprocessed = tf.keras.applications.resnet50.preprocess_input(image)
            else:
                # Default to ResNet50 preprocessing
                preprocessed = tf.keras.applications.resnet50.preprocess_input(image)
        else:
            # If specific preprocessing is not available, perform basic normalization
            preprocessed = image / 127.5 - 1.0
        
        # Add batch dimension if not present
        if len(preprocessed.shape) == 3:
            preprocessed = tf.expand_dims(preprocessed, 0)
        
        return preprocessed
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images for feature extraction.
        
        Args:
            images (tf.Tensor or numpy.ndarray): Batch of images with shape [batch_size, height, width, channels].
        
        Returns:
            tf.Tensor: Preprocessed images with shape [batch_size, height, width, channels].
        """
        if hasattr(tf.keras.applications, 'resnet50'):
            # Use the appropriate preprocessing function based on the base model
            if self.config.get('base_model', '').lower() == 'resnet50':
                return tf.keras.applications.resnet50.preprocess_input(images)
            else:
                # Default to ResNet50 preprocessing
                return tf.keras.applications.resnet50.preprocess_input(images)
        else:
            # If specific preprocessing is not available, perform basic normalization
            return images / 127.5 - 1.0
    
    def save_model(self, filepath):
        """
        Save the feature extraction model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        self.model.save(filepath)
        self.logger.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the feature extraction model from a file.
        
        Args:
            filepath (str): Path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath)
        self.logger.logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Simple test
    extractor = StaticFeatureExtractor()
    print("Model initialized successfully")
