"""
Unified Training System Module for Cross-Attention CNN (Phase 4.3)

This module integrates all training-related components into a cohesive system:
- Data loading and batch generation
- Training configuration and hyperparameter management
- Advanced training callbacks
- Performance monitoring
- Visualization tools
- Training progress tracking

Usage:
    from src.training.training_system import TrainingSystem
    
    # Initialize the training system
    training_system = TrainingSystem(
        static_features_path=static_path,
        dynamic_features_path=dynamic_path,
        labels_path=labels_path,
        output_dir=output_dir
    )
    
    # Train the model
    model, history = training_system.train_model(model, epochs=50, batch_size=32)
    
    # Analyze training results
    training_system.analyze_results(model, history)
"""

import os
import time
import numpy as np
import tensorflow as tf
from utils.logger import get_logger
from utils.metrics import RSquared, r_squared_metric, per_trait_metrics
from utils.training_callbacks import (
    PerformanceMonitorCallback, 
    TimeTrackingCallback, 
    LearningRateTrackerCallback
)


class TrainingSystem:
    """
    Comprehensive training system that integrates all Phase 4.3 components
    into a cohesive training pipeline.
    
    This class provides a unified interface for:
    - Loading and processing training data
    - Configuring training hyperparameters
    - Setting up advanced monitoring callbacks
    - Training models with comprehensive logging
    - Analyzing and visualizing training results
    """
    
    def __init__(self, 
                 static_features_path=None, 
                 dynamic_features_path=None, 
                 labels_path=None, 
                 val_split=0.2,
                 output_dir='results/training_output',
                 experiment_name="personality_model_training",
                 logger=None):
        """
        Initialize the training system.
        
        Args:
            static_features_path (str): Path to static features .npy file
            dynamic_features_path (str): Path to dynamic features .npy file
            labels_path (str): Path to labels .npy file
            val_split (float): Validation split ratio (0.0-1.0)
            output_dir (str): Directory to save training outputs
            experiment_name (str): Name for the experiment for logging
            logger: Optional pre-configured logger instance
        """
        # Initialize logger
        self.logger = logger or get_logger(experiment_name=experiment_name)
        self.logger.log_system_info()
        
        # Store configuration
        self.static_features_path = static_features_path
        self.dynamic_features_path = dynamic_features_path
        self.labels_path = labels_path
        self.val_split = val_split
        self.output_dir = output_dir
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data if paths are provided
        self.data_loaded = False
        if static_features_path and dynamic_features_path and labels_path:
            self._load_data()
    
    def _load_data(self):
        """
        Load static features, dynamic features, and labels and split into training/validation sets.
        """
        self.logger.logger.info("Loading training data...")
        
        try:
            # Load data from files
            static = np.load(self.static_features_path)
            dynamic = np.load(self.dynamic_features_path)
            labels = np.load(self.labels_path)
            
            # Verify dimensions
            assert static.shape[0] == dynamic.shape[0] == labels.shape[0], "Mismatched sample counts"
            
            # Shuffle data
            idx = np.arange(static.shape[0])
            np.random.shuffle(idx)
            static, dynamic, labels = static[idx], dynamic[idx], labels[idx]
            
            # Split into training and validation sets
            split = int(static.shape[0] * (1 - self.val_split))
            self.train_data = (static[:split], dynamic[:split], labels[:split])
            self.val_data = (static[split:], dynamic[split:], labels[split:])
            
            # Log data loading summary
            self.logger.logger.info(f"Data loaded successfully:")
            self.logger.logger.info(f"  Train samples: {self.train_data[0].shape[0]}")
            self.logger.logger.info(f"  Validation samples: {self.val_data[0].shape[0]}")
            self.logger.logger.info(f"  Static feature dimension: {static.shape[1]}")
            self.logger.logger.info(f"  Dynamic feature dimension: {dynamic.shape[1]}")
            self.logger.logger.info(f"  Labels dimension: {labels.shape[1]}")
            
            # Update data loaded flag
            self.data_loaded = True
            
        except Exception as e:
            self.logger.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def batch_generator(self, static, dynamic, labels, batch_size):
        """
        Generate batches of data for training.
        
        Args:
            static: Static feature array
            dynamic: Dynamic feature array
            labels: Labels array
            batch_size: Batch size
            
        Yields:
            tuple: ((static_batch, dynamic_batch), labels_batch)
        """
        n = static.shape[0]
        while True:
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                yield (static[batch_idx], dynamic[batch_idx]), labels[batch_idx]
    
    def setup_training_directories(self):
        """
        Create the directory structure for training outputs.
        
        Returns:
            dict: Dictionary containing paths to all output directories
        """
        # Create main directories
        tensorboard_dir = os.path.join(self.output_dir, 'tensorboard_logs')
        batch_metrics_dir = os.path.join(self.output_dir, 'batch_metrics')
        time_tracking_dir = os.path.join(self.output_dir, 'time_tracking')
        lr_tracking_dir = os.path.join(self.output_dir, 'lr_tracking')
        visualization_dir = os.path.join(self.output_dir, 'training_visualizations')
        attention_dir = os.path.join(self.output_dir, 'attention_visualizations')
        analysis_dir = os.path.join(self.output_dir, 'training_time_analysis')
        
        # Create all directories
        dirs = {
            'tensorboard': tensorboard_dir,
            'batch_metrics': batch_metrics_dir,
            'time_tracking': time_tracking_dir,
            'lr_tracking': lr_tracking_dir,
            'visualization': visualization_dir,
            'attention': attention_dir,
            'analysis': analysis_dir
        }
        
        for directory in dirs.values():
            os.makedirs(directory, exist_ok=True)
        
        return dirs
    
    def configure_callbacks(self, directories):
        """
        Configure the training callbacks.
        
        Args:
            directories (dict): Dictionary of output directories
            
        Returns:
            list: List of callbacks for training
        """
        # Set up learning rate scheduler
        initial_learning_rate = 1e-3
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        self.logger.logger.info(f"Learning rate scheduler configured with initial LR: {initial_learning_rate}")
        
        # Custom callback for detailed metric logging
        class MetricLoggerCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                
                # Log all metrics to the research logger
                self.logger.log_metrics(logs, step=epoch)
                
                # Log trait-specific metrics
                trait_metrics = {}
                for trait_idx, trait_name in enumerate(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']):
                    for metric in ['mae', 'mse', 'r_squared']:
                        metric_key = f'{metric}_{trait_name[0]}'
                        if metric_key in logs:
                            trait_metrics[f'{trait_name}_{metric}'] = logs[metric_key]
                
                if trait_metrics:
                    self.logger.log_metrics(trait_metrics, step=epoch)
        
        # Create callback list
        callbacks = [
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save best weights
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'checkpoint_weights.h5'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=1
            ),
            # Learning rate scheduler
            lr_scheduler,
            # TensorBoard callback for visualization
            tf.keras.callbacks.TensorBoard(
                log_dir=directories['tensorboard'],
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            # Custom metric logger
            MetricLoggerCallback(self.logger),
            # Phase 4.3 advanced monitoring callbacks
            PerformanceMonitorCallback(
                log_dir=directories['batch_metrics'],
                batch_log_frequency=10,
                logger=self.logger
            ),
            TimeTrackingCallback(
                log_dir=directories['time_tracking'],
                batch_log_frequency=20,
                logger=self.logger
            ),
            LearningRateTrackerCallback(
                log_dir=directories['lr_tracking'],
                logger=self.logger
            )
        ]
        
        return callbacks, initial_learning_rate
    
    def train_model(self, model, epochs=50, batch_size=32):
        """
        Train the model with the loaded data.
        
        Args:
            model: Model to train (instance of tf.keras.Model or compatible)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            
        Returns:
            tuple: (trained_model, training_history)
        """
        # Verify data is loaded
        if not self.data_loaded:
            if self.static_features_path and self.dynamic_features_path and self.labels_path:
                self._load_data()
            else:
                raise ValueError("Training data not loaded and paths not provided.")
        
        # Extract data for convenience
        static_tr, dynamic_tr, y_tr = self.train_data
        static_val, dynamic_val, y_val = self.val_data
        
        # Set up directories and callbacks
        directories = self.setup_training_directories()
        callbacks, initial_learning_rate = self.configure_callbacks(directories)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss='mse',
            metrics=['accuracy', 'mae', 'mse', r_squared_metric] + per_trait_metrics()
        )
        
        # Make a fake prediction to build the model
        dummy_static = np.zeros((1, static_tr.shape[1]), dtype=np.float32)
        dummy_dynamic = np.zeros((1, dynamic_tr.shape[1]), dtype=np.float32)
        _ = model((dummy_static, dummy_dynamic))
        
        # Log model architecture and hyperparameters
        self.logger.log_model_architecture(model)
        hyperparams = {
            'static_dim': static_tr.shape[1],
            'dynamic_dim': dynamic_tr.shape[1],
            'fusion_dim': getattr(model, 'fusion_dim', 128),
            'num_heads': getattr(model.cross_attention.attention, 'num_heads', 4),
            'dropout_rate': getattr(model, 'dropout_rate', 0.3),
            'batch_size': batch_size,
            'initial_learning_rate': initial_learning_rate,
            'optimizer': 'Adam',
            'epochs': epochs
        }
        self.logger.log_hyperparameters(hyperparams)
        
        # Start training
        self.logger.logger.info("Starting training...")
        training_start_time = time.time()
        
        # Train the model
        history = model.fit(
            self.batch_generator(static_tr, dynamic_tr, y_tr, batch_size),
            steps_per_epoch=max(1, static_tr.shape[0] // batch_size),
            validation_data=self.batch_generator(static_val, dynamic_val, y_val, batch_size),
            validation_steps=max(1, static_val.shape[0] // batch_size),
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Calculate total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Save the model weights
        weights_path = os.path.join(self.output_dir, 'final_weights.h5')
        model.save_weights(weights_path)
        self.logger.logger.info(f"Model weights saved to {weights_path}")
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.npy')
        np.save(history_path, history.history)
        self.logger.logger.info(f"Training history saved to {history_path}")
        
        return model, history
    
    def analyze_results(self, model, history, sample_size=10):
        """
        Analyze and visualize training results.
        
        Args:
            model: Trained model
            history: Training history object
            sample_size: Number of samples to use for attention visualization
        """
        self.logger.logger.info("Analyzing training results...")
        
        # Create output directories if not already created
        directories = self.setup_training_directories()
        
        # Extract metrics from history
        epochs_trained = len(history.history['loss'])
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Log training summary
        self.logger.logger.info(f"Training completed in {epochs_trained} epochs")
        self.logger.logger.info(f"Final training loss: {final_loss:.4f}")
        self.logger.logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        # Log per-trait metrics if available
        for trait_idx, trait_name in enumerate(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']):
            trait_key = f'r_squared_{trait_name[0]}'
            if trait_key in history.history:
                final_trait_r2 = history.history[trait_key][-1]
                self.logger.logger.info(f"{trait_name} R²: {final_trait_r2:.4f}")
        
        # Generate visualizations
        try:
            self.logger.logger.info("Generating training visualizations...")
            
            # Import visualization functions
            from scripts.visualize_advanced_training import (
                load_json_file, 
                create_combined_training_dashboard,
                plot_learning_rate_changes,
                plot_training_validation_curves,
                plot_batch_performance,
                plot_per_trait_performance,
                create_correlation_matrix
            )
            
            # Load tracking data
            metrics_history = load_json_file(os.path.join(directories['batch_metrics'], 'metrics_history.json'))
            time_logs = load_json_file(os.path.join(directories['time_tracking'], 'time_logs.json'))
            lr_logs = load_json_file(os.path.join(directories['lr_tracking'], 'lr_logs.json'))
            
            # Generate visualizations
            self.logger.logger.info("Creating combined training dashboard...")
            create_combined_training_dashboard(metrics_history, time_logs, lr_logs, directories['visualization'])
            
            self.logger.logger.info("Plotting learning rate changes...")
            plot_learning_rate_changes(lr_logs, directories['visualization'])
            
            self.logger.logger.info("Plotting training/validation curves...")
            plot_training_validation_curves(metrics_history, directories['visualization'])
            
            self.logger.logger.info("Plotting batch performance...")
            plot_batch_performance(metrics_history, directories['visualization'])
            
            self.logger.logger.info("Plotting per-trait performance...")
            plot_per_trait_performance(metrics_history, directories['visualization'])
            
            # Create metrics correlation matrix
            try:
                self.logger.logger.info("Creating metrics correlation matrix...")
                create_correlation_matrix(metrics_history, directories['visualization'])
            except Exception as e:
                self.logger.logger.warning(f"Failed to create correlation matrix: {e}")
            
            # Run training time analysis
            self.logger.logger.info("Analyzing training time performance...")
            from scripts.analyze_training_time import (
                analyze_epoch_times,
                analyze_batch_times,
                create_training_speedup_recommendations,
                create_training_time_dashboard
            )
            
            try:
                # Run analysis components
                self.logger.logger.info("Analyzing epoch times...")
                analyze_epoch_times(time_logs, directories['analysis'])
                
                self.logger.logger.info("Analyzing batch times...")
                analyze_batch_times(time_logs, metrics_history, directories['analysis'])
                
                self.logger.logger.info("Creating training speedup recommendations...")
                create_training_speedup_recommendations(time_logs, metrics_history, directories['analysis'])
                
                self.logger.logger.info("Creating training time dashboard...")
                create_training_time_dashboard(time_logs, metrics_history, directories['analysis'])
                
                self.logger.logger.info(f"Training time analysis saved to {directories['analysis']}")
            except Exception as e:
                self.logger.logger.warning(f"Some training time analysis components failed: {e}")
        
            # Generate attention visualizations if the model supports it
            if hasattr(model, 'extract_attention_weights'):
                self.logger.logger.info("Generating attention visualizations...")
                from scripts.enhanced_attention_visualization import (
                    visualize_multi_head_attention,
                    visualize_attention_comparison,
                    create_attention_heatmap_dashboard
                )
                
                # Extract attention weights for a small sample
                sample_size = min(sample_size, self.val_data[0].shape[0])
                sample_static = self.val_data[0][:sample_size]
                sample_dynamic = self.val_data[1][:sample_size]
                
                try:
                    # Extract attention weights
                    attention_weights = model.extract_attention_weights((sample_static, sample_dynamic))
                    
                    # Create visualizations
                    self.logger.logger.info("Creating multi-head attention visualization...")
                    for i in range(min(3, sample_size)):
                        output_path = os.path.join(directories['attention'], f'multi_head_attention_sample_{i+1}.png')
                        visualize_multi_head_attention(attention_weights, i, output_path)
                    
                    self.logger.logger.info("Creating attention comparison visualization...")
                    visualize_attention_comparison(attention_weights, os.path.join(directories['attention'], 'attention_comparison.png'))
                    
                    self.logger.logger.info("Creating attention dashboard...")
                    create_attention_heatmap_dashboard(attention_weights, os.path.join(directories['attention'], 'attention_dashboard.png'))
                    
                    self.logger.logger.info(f"Attention visualizations saved to {directories['attention']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to generate attention visualizations: {e}")
            else:
                self.logger.logger.info("Skipping attention visualizations (model doesn't support attention weight extraction)")
            
            self.logger.logger.info("All training visualizations and analyses completed successfully.")
        except Exception as e:
            self.logger.logger.warning(f"Failed to generate some visualizations or analyses: {e}")
        
        self.logger.logger.info("Training analysis completed successfully.")


def train_personality_model(static_features_path, dynamic_features_path, labels_path, 
                           output_path, val_split=0.2, batch_size=32, epochs=50):
    """
    Convenience function for training a personality model with the unified training system.
    
    Args:
        static_features_path: Path to static features .npy file
        dynamic_features_path: Path to dynamic features .npy file
        labels_path: Path to labels .npy file
        output_path: Path to save the trained model weights
        val_split: Validation split ratio (0.0-1.0)
        batch_size: Batch size for training
        epochs: Number of epochs to train
        
    Returns:
        tuple: (model, history)
    """
    from src.models.personality_model import CompletePersonalityModel
    
    # Initialize the training system
    output_dir = os.path.dirname(output_path)
    training_system = TrainingSystem(
        static_features_path=static_features_path,
        dynamic_features_path=dynamic_features_path,
        labels_path=labels_path,
        val_split=val_split,
        output_dir=output_dir,
        experiment_name="train_personality_model"
    )
    
    # Load data
    if not training_system.data_loaded:
        training_system._load_data()
    
    # Initialize model
    static_dim = training_system.train_data[0].shape[1]
    dynamic_dim = training_system.train_data[1].shape[1]
    
    model = CompletePersonalityModel(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        fusion_dim=128,
        num_heads=4,
        dropout_rate=0.3
    )
    
    # Only train cross-attention and output dense layers
    for layer in model.layers:
        if not isinstance(layer, (tf.keras.layers.MultiHeadAttention, 
                                  tf.keras.layers.Dense, 
                                  tf.keras.layers.LayerNormalization, 
                                  tf.keras.layers.Dropout, 
                                  tf.keras.layers.BatchNormalization)):
            layer.trainable = False
    
    # Train the model
    model, history = training_system.train_model(model, epochs=epochs, batch_size=batch_size)
    
    # Analyze results
    training_system.analyze_results(model, history)
    
    # Save final weights to the specific path
    model.save_weights(output_path)
    
    return model, history
