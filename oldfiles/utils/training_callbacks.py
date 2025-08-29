"""
Custom callbacks for training monitoring in Cross-Attention CNN (Phase 4.3)

This module provides specialized Keras callbacks for:
- Per-batch metric logging
- Learning rate tracking
- Training time monitoring
- Epoch-level metric recording

Usage:
    from utils.training_callbacks import PerformanceMonitorCallback, TimeTrackingCallback
    
    # Add to callbacks list in model.fit()
    callbacks = [
        PerformanceMonitorCallback(log_dir='logs/batch_metrics'),
        TimeTrackingCallback(),
        # Other callbacks...
    ]
"""

import os
import time
import json
import numpy as np
import tensorflow as tf


class PerformanceMonitorCallback(tf.keras.callbacks.Callback):
    """
    A callback for detailed batch-level and epoch-level performance monitoring.
    
    This callback tracks metrics at both batch and epoch level, enabling fine-grained
    analysis of training dynamics.
    
    Attributes:
        log_dir (str): Directory to save batch metrics and logs
        batch_log_frequency (int): How often to log batch metrics (every N batches)
        logger (object): Reference to a logger object (if provided)
        metrics_history (dict): Dictionary to store all tracked metrics
    """
    
    def __init__(self, log_dir='logs/batch_metrics', batch_log_frequency=10, logger=None):
        """
        Initialize the performance monitor callback.
        
        Args:
            log_dir (str): Directory to save batch metrics and logs
            batch_log_frequency (int): How often to log batch metrics (every N batches)
            logger (object): Reference to a logger object (optional)
        """
        super(PerformanceMonitorCallback, self).__init__()
        self.log_dir = log_dir
        self.batch_log_frequency = batch_log_frequency
        self.logger = logger
        self.metrics_history = {
            'batch': {
                'loss': [],
                'metrics': {},
                'learning_rate': [],
                'batch_numbers': []
            },
            'epoch': {
                'loss': [],
                'val_loss': [],
                'metrics': {},
                'val_metrics': {},
                'learning_rate': [],
                'epoch_numbers': []
            }
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """
        Called at the start of training.
        
        Args:
            logs: Dictionary of logs (unused)
        """
        self.train_start_time = time.time()
        
        # Initialize metric tracking dictionaries for batch-level tracking
        if hasattr(self.model, 'metrics_names'):
            for metric in self.model.metrics_names:
                if metric != 'loss':
                    self.metrics_history['batch']['metrics'][metric] = []
                    self.metrics_history['epoch']['metrics'][metric] = []
                    self.metrics_history['epoch']['val_metrics'][metric] = []
        
        # Log the start of training
        if self.logger:
            self.logger.logger.info(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.logger.info(f"Monitoring batch-level metrics every {self.batch_log_frequency} batches")
    
    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch.
        
        Args:
            batch (int): Current batch index
            logs (dict): Dictionary containing batch metrics
        """
        logs = logs or {}
        
        # Only log every N batches to avoid excessive logging
        if batch % self.batch_log_frequency == 0:
            # Extract and store the current learning rate
            lr = self.model.optimizer.lr
            if hasattr(lr, 'numpy'):
                current_lr = float(lr.numpy())
            else:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            
            # Log batch metrics
            self.metrics_history['batch']['batch_numbers'].append(batch)
            self.metrics_history['batch']['learning_rate'].append(current_lr)
            
            if 'loss' in logs:
                self.metrics_history['batch']['loss'].append(float(logs['loss']))
            
            # Log other metrics
            for metric_name, metric_value in logs.items():
                if metric_name != 'loss':
                    if metric_name not in self.metrics_history['batch']['metrics']:
                        self.metrics_history['batch']['metrics'][metric_name] = []
                    self.metrics_history['batch']['metrics'][metric_name].append(float(metric_value))
            
            # Log to logger if available
            if self.logger:
                self.logger.logger.debug(f"Batch {batch} - Loss: {logs.get('loss', 'N/A'):.4f}, LR: {current_lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        
        Args:
            epoch (int): Current epoch index
            logs (dict): Dictionary containing epoch metrics
        """
        logs = logs or {}
        
        # Extract and store the current learning rate
        lr = self.model.optimizer.lr
        if hasattr(lr, 'numpy'):
            current_lr = float(lr.numpy())
        else:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # Store epoch data
        self.metrics_history['epoch']['epoch_numbers'].append(epoch)
        self.metrics_history['epoch']['learning_rate'].append(current_lr)
        
        if 'loss' in logs:
            self.metrics_history['epoch']['loss'].append(float(logs['loss']))
        
        if 'val_loss' in logs:
            self.metrics_history['epoch']['val_loss'].append(float(logs['val_loss']))
        
        # Store metrics
        for metric_name, metric_value in logs.items():
            if metric_name not in ['loss', 'val_loss']:
                if 'val_' in metric_name:
                    base_metric_name = metric_name.replace('val_', '')
                    if base_metric_name not in self.metrics_history['epoch']['val_metrics']:
                        self.metrics_history['epoch']['val_metrics'][base_metric_name] = []
                    self.metrics_history['epoch']['val_metrics'][base_metric_name].append(float(metric_value))
                else:
                    if metric_name not in self.metrics_history['epoch']['metrics']:
                        self.metrics_history['epoch']['metrics'][metric_name] = []
                    self.metrics_history['epoch']['metrics'][metric_name].append(float(metric_value))
        
        # Save the current state of metrics to a file
        metrics_path = os.path.join(self.log_dir, 'metrics_history.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log to logger if available
        if self.logger:
            elapsed_time = time.time() - self.train_start_time
            self.logger.logger.info(f"Epoch {epoch+1} completed in {elapsed_time:.2f}s - "
                                    f"Loss: {logs.get('loss', 'N/A'):.4f}, "
                                    f"Val Loss: {logs.get('val_loss', 'N/A'):.4f}, "
                                    f"LR: {current_lr:.6f}")
    
    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        
        Args:
            logs: Dictionary of logs (unused)
        """
        total_training_time = time.time() - self.train_start_time
        
        # Add training time to metrics history
        self.metrics_history['total_training_time_seconds'] = total_training_time
        self.metrics_history['total_training_time_formatted'] = time.strftime(
            '%H:%M:%S', time.gmtime(total_training_time))
        
        # Save the final metrics history
        metrics_path = os.path.join(self.log_dir, 'metrics_history.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log completion
        if self.logger:
            hours, remainder = divmod(total_training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.logger.logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            self.logger.logger.info(f"Final metrics history saved to {metrics_path}")


class TimeTrackingCallback(tf.keras.callbacks.Callback):
    """
    A callback for tracking training time at epoch and batch levels.
    
    Attributes:
        log_dir (str): Directory to save time logs
        batch_log_frequency (int): How often to log batch times
        logger (object): Reference to a logger object (if provided)
    """
    
    def __init__(self, log_dir='logs/time_tracking', batch_log_frequency=50, logger=None):
        """
        Initialize the time tracking callback.
        
        Args:
            log_dir (str): Directory to save time logs
            batch_log_frequency (int): How often to log batch times
            logger (object): Reference to a logger object (optional)
        """
        super(TimeTrackingCallback, self).__init__()
        self.log_dir = log_dir
        self.batch_log_frequency = batch_log_frequency
        self.logger = logger
        self.time_logs = {
            'epoch_times': [],
            'batch_times': [],
            'batch_numbers': [],
            'epoch_numbers': [],
            'total_training_time': 0
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """Initialize timers at the start of training."""
        self.train_start_time = time.time()
        self.batch_start_time = time.time()
        self.epoch_start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        """
        Track time for each batch.
        
        Args:
            batch (int): Current batch index
            logs (dict): Dictionary containing batch metrics (unused)
        """
        batch_time = time.time() - self.batch_start_time
        
        # Only log every N batches to avoid excessive logging
        if batch % self.batch_log_frequency == 0:
            self.time_logs['batch_times'].append(batch_time)
            self.time_logs['batch_numbers'].append(batch)
            
            # Log to logger if available
            if self.logger:
                self.logger.logger.debug(f"Batch {batch} completed in {batch_time:.4f}s")
        
        # Reset timer for next batch
        self.batch_start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Initialize epoch timer."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Track time for each epoch.
        
        Args:
            epoch (int): Current epoch index
            logs (dict): Dictionary containing epoch metrics (unused)
        """
        epoch_time = time.time() - self.epoch_start_time
        self.time_logs['epoch_times'].append(epoch_time)
        self.time_logs['epoch_numbers'].append(epoch)
        
        # Save the time logs
        time_logs_path = os.path.join(self.log_dir, 'time_logs.json')
        with open(time_logs_path, 'w') as f:
            json.dump(self.time_logs, f, indent=2)
        
        # Log to logger if available
        if self.logger:
            self.logger.logger.info(f"Epoch {epoch+1} training time: {epoch_time:.2f}s")
    
    def on_train_end(self, logs=None):
        """Track total training time."""
        total_training_time = time.time() - self.train_start_time
        self.time_logs['total_training_time'] = total_training_time
        
        # Save the final time logs
        time_logs_path = os.path.join(self.log_dir, 'time_logs.json')
        with open(time_logs_path, 'w') as f:
            json.dump(self.time_logs, f, indent=2)
        
        # Log to logger if available
        if self.logger:
            hours, remainder = divmod(total_training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.logger.logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            self.logger.logger.info(f"Time logs saved to {time_logs_path}")


class LearningRateTrackerCallback(tf.keras.callbacks.Callback):
    """
    A callback for advanced learning rate tracking.
    
    This callback specifically focuses on capturing learning rate
    changes during training, especially when using schedulers
    like ReduceLROnPlateau.
    
    Attributes:
        log_dir (str): Directory to save learning rate logs
        logger (object): Reference to a logger object (if provided)
    """
    
    def __init__(self, log_dir='logs/lr_tracking', logger=None):
        """
        Initialize the learning rate tracker callback.
        
        Args:
            log_dir (str): Directory to save learning rate logs
            logger (object): Reference to a logger object (optional)
        """
        super(LearningRateTrackerCallback, self).__init__()
        self.log_dir = log_dir
        self.logger = logger
        self.lr_logs = {
            'epoch': [],
            'learning_rate': [],
            'is_reduction': []
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Keep track of the previous learning rate
        self.prev_lr = None
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Check and log the current learning rate at the beginning of each epoch.
        
        Args:
            epoch (int): Current epoch index
            logs (dict): Dictionary containing logs (unused)
        """
        # Get current learning rate
        lr = self.model.optimizer.lr
        if hasattr(lr, 'numpy'):
            current_lr = float(lr.numpy())
        else:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # Check if this is a reduction in learning rate
        is_reduction = False
        if self.prev_lr is not None and current_lr < self.prev_lr:
            is_reduction = True
            if self.logger:
                self.logger.logger.info(f"Learning rate reduced: {self.prev_lr:.6f} -> {current_lr:.6f}")
        
        # Update the logs
        self.lr_logs['epoch'].append(epoch)
        self.lr_logs['learning_rate'].append(current_lr)
        self.lr_logs['is_reduction'].append(is_reduction)
        
        # Save the logs
        lr_logs_path = os.path.join(self.log_dir, 'lr_logs.json')
        with open(lr_logs_path, 'w') as f:
            json.dump(self.lr_logs, f, indent=2)
        
        # Update previous learning rate
        self.prev_lr = current_lr
    def on_train_end(self, logs=None):
        """
        Log a summary of learning rate changes at the end of training.
        
        Args:
            logs (dict): Dictionary containing logs (unused)
        """
        # Count the number of learning rate reductions
        num_reductions = sum(self.lr_logs['is_reduction'])
        
        # Save the logs
        lr_logs_path = os.path.join(self.log_dir, 'lr_logs.json')
        with open(lr_logs_path, 'w') as f:
            json.dump(self.lr_logs, f, indent=2)
        
        # Log to logger if available
        if self.logger:
            self.logger.logger.info(f"Learning rate adjusted {num_reductions} times during training")
            self.logger.logger.info(f"Initial learning rate: {self.lr_logs['learning_rate'][0]:.6f}")
            self.logger.logger.info(f"Final learning rate: {self.lr_logs['learning_rate'][-1]:.6f}")
