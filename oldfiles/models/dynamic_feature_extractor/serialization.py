#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialization.py
# Description: Module for model serialization and version management

import os
import json
import yaml
import time
import shutil
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path
import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger

class ModelSerializer:
    """
    Class for handling model serialization, versioning, and loading.
    """
    
    def __init__(self, model=None, model_name="dynamic_feature_extractor", version=None):
        """
        Initialize the model serializer.
        
        Args:
            model (tf.keras.Model): Model to serialize.
            model_name (str): Name of the model.
            version (str): Version identifier. If None, generates timestamp-based version.
        """
        self.model = model
        self.model_name = model_name
        self.version = version or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(experiment_name=f"{model_name}_serializer")
    
    def save_model(self, base_dir, include_optimizer=True, save_format='tf', 
                  config=None, metadata=None, save_weights_only=False):
        """
        Save the model with associated metadata.
        
        Args:
            base_dir (str): Base directory where models are stored.
            include_optimizer (bool): Whether to include optimizer state.
            save_format (str): Format to save the model ('tf' or 'h5').
            config (dict): Model configuration dictionary.
            metadata (dict): Additional metadata to save.
            save_weights_only (bool): If True, saves only weights instead of full model.
            
        Returns:
            str: Path to the saved model.
        """
        if self.model is None:
            self.logger.logger.error("No model provided for serialization")
            return None
        
        # Create version directory
        version_dir = os.path.join(base_dir, self.model_name, f"v{self.version}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Prepare metadata
        current_time = datetime.datetime.now().isoformat()
        full_metadata = {
            "model_name": self.model_name,
            "version": self.version,
            "saved_at": current_time,
            "include_optimizer": include_optimizer,
            "save_format": save_format,
            "tensorflow_version": tf.__version__,
        }
        
        # Include model architecture summary
        try:
            string_list = []
            self.model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
            model_summary = '\n'.join(string_list)
            full_metadata["model_summary"] = model_summary
        except Exception as e:
            self.logger.logger.warning(f"Could not generate model summary: {str(e)}")
        
        # Include model configuration
        if config:
            full_metadata["config"] = config
        
        # Include additional metadata
        if metadata:
            full_metadata.update(metadata)
        
        # Save metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Save model
        try:
            model_path = os.path.join(version_dir, "model")
            
            if save_weights_only:
                weights_path = os.path.join(version_dir, "weights")
                self.model.save_weights(weights_path)
                self.logger.logger.info(f"Model weights saved to {weights_path}")
            else:
                self.model.save(model_path, include_optimizer=include_optimizer, save_format=save_format)
                self.logger.logger.info(f"Model saved to {model_path}")
            
            # Save model configuration
            if isinstance(config, dict):
                config_path = os.path.join(version_dir, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                self.logger.logger.info(f"Model configuration saved to {config_path}")
            
            return version_dir
        
        except Exception as e:
            self.logger.logger.error(f"Error saving model: {str(e)}")
            return None
    
    @classmethod
    def load_model(cls, model_path, custom_objects=None):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model.
            custom_objects (dict): Dictionary mapping names to custom classes or functions.
            
        Returns:
            tuple: (loaded_model, metadata)
        """
        logger = get_logger(experiment_name="model_serializer")
        
        try:
            # Check if path exists
            if not os.path.exists(model_path):
                logger.logger.error(f"Model path not found: {model_path}")
                return None, None
            
            # Load metadata if it exists
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    logger.logger.warning(f"Error loading metadata: {str(e)}")
            
            # Load model
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.logger.info(f"Model loaded from {model_path}")
            
            return model, metadata
        
        except Exception as e:
            logger.logger.error(f"Error loading model: {str(e)}")
            return None, None
    
    @classmethod
    def load_weights(cls, model, weights_path):
        """
        Load weights into a model.
        
        Args:
            model (tf.keras.Model): Model to load weights into.
            weights_path (str): Path to the saved weights.
            
        Returns:
            bool: Whether the weights were loaded successfully.
        """
        logger = get_logger(experiment_name="model_serializer")
        
        try:
            model.load_weights(weights_path)
            logger.logger.info(f"Weights loaded from {weights_path}")
            return True
        except Exception as e:
            logger.logger.error(f"Error loading weights: {str(e)}")
            return False
    
    @classmethod
    def list_available_versions(cls, base_dir, model_name=None):
        """
        List all available versions of a model.
        
        Args:
            base_dir (str): Base directory where models are stored.
            model_name (str): Name of the model. If None, lists all models.
            
        Returns:
            dict: Dictionary mapping model names to lists of versions.
        """
        logger = get_logger(experiment_name="model_serializer")
        
        if not os.path.exists(base_dir):
            logger.logger.warning(f"Base directory not found: {base_dir}")
            return {}
        
        model_versions = {}
        
        if model_name:
            model_dir = os.path.join(base_dir, model_name)
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                versions = [d for d in os.listdir(model_dir) 
                           if os.path.isdir(os.path.join(model_dir, d))]
                model_versions[model_name] = sorted(versions)
        else:
            # List all models
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    model_dir = os.path.join(base_dir, item)
                    versions = [d for d in os.listdir(model_dir) 
                               if os.path.isdir(os.path.join(model_dir, d))]
                    model_versions[item] = sorted(versions)
        
        return model_versions
    
    @classmethod
    def get_latest_version(cls, base_dir, model_name):
        """
        Get the latest version of a model.
        
        Args:
            base_dir (str): Base directory where models are stored.
            model_name (str): Name of the model.
            
        Returns:
            str: Latest version or None if not found.
        """
        versions = cls.list_available_versions(base_dir, model_name).get(model_name, [])
        return versions[-1] if versions else None
    
    @classmethod
    def load_latest_model(cls, base_dir, model_name, custom_objects=None):
        """
        Load the latest version of a model.
        
        Args:
            base_dir (str): Base directory where models are stored.
            model_name (str): Name of the model.
            custom_objects (dict): Dictionary mapping names to custom classes or functions.
            
        Returns:
            tuple: (loaded_model, metadata, version)
        """
        logger = get_logger(experiment_name="model_serializer")
        
        latest_version = cls.get_latest_version(base_dir, model_name)
        if not latest_version:
            logger.logger.warning(f"No versions found for model {model_name}")
            return None, None, None
        
        model_path = os.path.join(base_dir, model_name, latest_version, "model")
        model, metadata = cls.load_model(model_path, custom_objects)
        
        return model, metadata, latest_version
    
    @classmethod
    def export_for_inference(cls, model, export_dir):
        """
        Export a model for inference (TensorFlow SavedModel format).
        
        Args:
            model (tf.keras.Model): Model to export.
            export_dir (str): Directory to export the model to.
        
        Returns:
            str: Path to the exported model.
        """
        logger = get_logger(experiment_name="model_serializer")
        
        try:
            os.makedirs(export_dir, exist_ok=True)
            
            # Use standard Keras export (let Keras handle signatures)
            model.save(export_dir, save_format='tf')
            logger.logger.info(f"Model exported for inference to {export_dir}")
            
            return export_dir
        except Exception as e:
            logger.logger.error(f"Error exporting model: {str(e)}")
            return None

if __name__ == "__main__":
    # Test functionality
    print("Model serialization module loaded")
