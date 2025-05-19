"""
Feature Fusion Module and Personality Prediction Head

This module implements the feature fusion component and prediction head for personality trait prediction,
building upon the cross-attention mechanism to combine static (face) and dynamic (motion) features.

The module outputs predictions for the OCEAN personality traits:
- Openness
- Conscientiousness 
- Extraversion
- Agreeableness
- Neuroticism
"""

import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
from src.models.cross_attention import CrossAttention

class FeatureFusionModule(tf.keras.layers.Layer):
    """
    Feature Fusion Module that combines static and dynamic features using cross-attention
    and produces a joint representation suitable for personality trait prediction.
    """
    
    def __init__(self, fusion_dim=128, dropout_rate=0.3):
        """
        Initialize the feature fusion module.
        
        Args:
            fusion_dim: Dimension of the fused feature representation
            dropout_rate: Dropout rate for regularization
        """
        super(FeatureFusionModule, self).__init__()
        self.fusion_dim = fusion_dim
        self.dropout = Dropout(dropout_rate)
        self.batch_norm = BatchNormalization()
        
    def call(self, fused_features, training=False):
        """
        Process the fused features from cross-attention.
        
        Args:
            fused_features: Features from the cross-attention mechanism
            training: Whether in training mode (for dropout)
            
        Returns:
            Processed fusion features
        """
        # Apply batch normalization and dropout
        x = self.batch_norm(fused_features, training=training)
        x = self.dropout(x, training=training)
        return x


class PersonalityPredictionHead(tf.keras.layers.Layer):
    """
    Prediction head for OCEAN personality traits that takes fused features and outputs trait scores.
    """
    
    def __init__(self, hidden_dim=64):
        """
        Initialize the personality prediction head.
        
        Args:
            hidden_dim: Dimension of the hidden layer
        """
        super(PersonalityPredictionHead, self).__init__()
        self.hidden_layer = Dense(hidden_dim, activation='relu')
        # Final layer with 5 outputs for the OCEAN traits
        self.output_layer = Dense(5, activation='linear')  # Linear for regression of trait scores
        
    def call(self, fusion_features):
        """
        Predict personality traits from fusion features.
        
        Args:
            fusion_features: Fused static and dynamic features
            
        Returns:
            Predicted OCEAN personality trait scores [batch_size, 5]
        """
        x = self.hidden_layer(fusion_features)
        predictions = self.output_layer(x)
        return predictions


class CompletePersonalityModel(Model):
    """
    End-to-end model that integrates cross-attention, feature fusion, and prediction head
    for personality trait prediction from static and dynamic features.
    """
    
    def __init__(self, static_dim=512, dynamic_dim=256, fusion_dim=128, num_heads=4, dropout_rate=0.3):
        """
        Initialize the complete personality trait prediction model.
        
        Args:
            static_dim: Dimension of static features (face CNN output)
            dynamic_dim: Dimension of dynamic features (optical flow CNN output)
            fusion_dim: Dimension of the fused representation
            num_heads: Number of attention heads in cross-attention
            dropout_rate: Dropout rate for regularization
        """
        super(CompletePersonalityModel, self).__init__()
        
        # Cross-Attention mechanism
        self.cross_attention = CrossAttention(static_dim, dynamic_dim, fusion_dim, num_heads)
        
        # Feature fusion module
        self.fusion_module = FeatureFusionModule(fusion_dim, dropout_rate)
        
        # Prediction head
        self.prediction_head = PersonalityPredictionHead()
        
    def call(self, inputs, training=False):
        """
        Forward pass through the complete model.
        
        Args:
            inputs: Tuple of (static_features, dynamic_features)
            training: Whether in training mode
            
        Returns:
            Predicted OCEAN personality trait scores [batch_size, 5]
        """
        static_features, dynamic_features = inputs
        
        # Apply cross-attention
        cross_attention_output = self.cross_attention(static_features, dynamic_features)
        
        # Apply feature fusion
        fusion_output = self.fusion_module(cross_attention_output, training=training)
        
        # Generate predictions
        predictions = self.prediction_head(fusion_output)
        
        return predictions
    
    def get_config(self):
        """
        Get configuration for the model.
        """
        config = super().get_config()
        config.update({
            'static_dim': self.cross_attention.static_proj.units,
            'dynamic_dim': self.cross_attention.dynamic_proj.units,
            'fusion_dim': self.fusion_module.fusion_dim,
            'num_heads': self.cross_attention.attention.num_heads,
            'dropout_rate': self.fusion_module.dropout.rate,
        })
        return config
