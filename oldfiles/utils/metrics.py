"""
Custom metrics for personality trait prediction models.

This module implements:
1. R-squared (coefficient of determination) metric
2. Per-trait metrics (MAE, MSE, R-squared for each OCEAN trait)

Usage:
    from utils.metrics import RSquared, r_squared_metric, per_trait_metrics
    
    # Add to model.compile
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse', r_squared_metric] + per_trait_metrics()
    )
"""

import tensorflow as tf
import numpy as np

class RSquared(tf.keras.metrics.Metric):
    """
    R-squared metric (coefficient of determination) for regression.
    
    R² = 1 - (sum of squared residuals) / (total sum of squares)
    Values range from 0 to 1, with 1 being a perfect fit.
    """
    
    def __init__(self, name='r_squared', **kwargs):
        super(RSquared, self).__init__(name=name, **kwargs)
        # Sum of squared residuals: Σ(y_true - y_pred)²
        self.ss_residual = self.add_weight(name='ss_residual', initializer='zeros')
        # Total sum of squares: Σ(y_true - mean(y_true))²
        self.ss_total = self.add_weight(name='ss_total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state with batch values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
        """
        # Convert to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute residuals
        residuals = tf.math.subtract(y_true, y_pred)
        
        # Compute mean of y_true along the batch dimension (axis 0)
        y_mean = tf.reduce_mean(y_true, axis=0)
        
        # Compute squared differences from mean
        squared_differences = tf.math.square(tf.math.subtract(y_true, y_mean))
        
        # Compute sum of squared residuals and total sum of squares
        ss_residual_batch = tf.reduce_sum(tf.math.square(residuals))
        ss_total_batch = tf.reduce_sum(squared_differences)
        
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            ss_residual_batch = tf.multiply(ss_residual_batch, sample_weight)
            ss_total_batch = tf.multiply(ss_total_batch, sample_weight)
        
        # Update the running totals
        self.ss_residual.assign_add(ss_residual_batch)
        self.ss_total.assign_add(ss_total_batch)
    
    def result(self):
        """
        Compute the R-squared value from the accumulated statistics.
        
        Returns:
            R-squared value as a float32 tensor
        """
        # If denominator is zero, return zero (prevents division by zero)
        # This happens when all y_true values are identical
        if tf.equal(self.ss_total, 0.0):
            return tf.constant(0.0, dtype=tf.float32)
        
        # R² = 1 - (SS_residual / SS_total)
        r_squared = tf.math.subtract(
            tf.constant(1.0, dtype=tf.float32),
            tf.math.divide_no_nan(self.ss_residual, self.ss_total)
        )
        
        # Clip to [0, 1] as negative values indicate worse fit than mean
        # For regression problems, we're typically only interested in [0, 1]
        return tf.clip_by_value(r_squared, 0.0, 1.0)
    
    def reset_state(self):
        """Reset the metric state between epochs."""
        self.ss_residual.assign(0.0)
        self.ss_total.assign(0.0)


# Shorthand for using the R-squared metric
r_squared_metric = RSquared()


class PerTraitMetric(tf.keras.metrics.Metric):
    """
    Base class for per-trait metrics.
    
    This class provides the structure for computing metrics for a specific
    personality trait (one of the OCEAN dimensions).
    """
    
    def __init__(self, trait_index, metric_fn, name, **kwargs):
        """
        Initialize the per-trait metric.
        
        Args:
            trait_index: Index of the trait (0-4 for OCEAN)
            metric_fn: Function to compute the metric
            name: Name of the metric
        """
        super(PerTraitMetric, self).__init__(name=name, **kwargs)
        self.trait_index = trait_index
        self.metric_fn = metric_fn
        self.trait_metric = metric_fn()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state with batch values.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
        """
        # Extract values for the specific trait
        y_true_trait = y_true[:, self.trait_index]
        y_pred_trait = y_pred[:, self.trait_index]
        
        # Update the underlying metric
        self.trait_metric.update_state(y_true_trait, y_pred_trait, sample_weight)
    
    def result(self):
        """
        Compute the result from the underlying metric.
        
        Returns:
            The metric value as a tensor
        """
        return self.trait_metric.result()
    
    def reset_state(self):
        """Reset the metric state between epochs."""
        self.trait_metric.reset_state()


def per_trait_metrics():
    """
    Create a list of per-trait metrics for all OCEAN dimensions.
    
    Returns:
        List of metrics for tracking per-trait performance
    """
    trait_names = ['O', 'C', 'E', 'A', 'N']
    metrics = []
    
    for i, trait in enumerate(trait_names):
        # Mean Absolute Error per trait
        metrics.append(PerTraitMetric(
            trait_index=i,
            metric_fn=tf.keras.metrics.MeanAbsoluteError,
            name=f'mae_{trait}'
        ))
        
        # Mean Squared Error per trait
        metrics.append(PerTraitMetric(
            trait_index=i,
            metric_fn=tf.keras.metrics.MeanSquaredError,
            name=f'mse_{trait}'
        ))
        
        # R-squared per trait
        metrics.append(PerTraitMetric(
            trait_index=i,
            metric_fn=RSquared,
            name=f'r_squared_{trait}'
        ))
    
    return metrics
