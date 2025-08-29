"""
Training pipeline for Cross-Attention Personality Model (Phase 4.3)

- Loads pre-extracted static and dynamic features and OCEAN labels
- Implements data loader and batch generator
- Trains only the cross-attention and output dense layers
- Includes validation loop
- Advanced logging and metrics tracking
- Training time and batch-level performance monitoring

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/train_personality_model.py --static_features data/static_features.npy --dynamic_features data/dynamic_features.npy --labels data/labels.npy --val_split 0.2 --batch_size 32 --epochs 50 --output results/personality_model_trained.h5
    }

See docs/phase_4_training_pipeline.md for details.
"""

import os
import argparse
import numpy as np
import time
import tensorflow as tf
from src.models.personality_model import CompletePersonalityModel
from utils.logger import get_logger
from utils.metrics import RSquared, r_squared_metric, per_trait_metrics
from utils.training_callbacks import PerformanceMonitorCallback, TimeTrackingCallback, LearningRateTrackerCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Train Cross-Attention Personality Model")
    parser.add_argument('--static_features', type=str, required=True, help='Path to static features .npy')
    parser.add_argument('--dynamic_features', type=str, required=True, help='Path to dynamic features .npy')
    parser.add_argument('--labels', type=str, required=True, help='Path to OCEAN labels .npy')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--output', type=str, default='results/personality_model_trained.h5', help='Output model file')
    return parser.parse_args()

def load_data(static_path, dynamic_path, labels_path, val_split):
    static = np.load(static_path)
    dynamic = np.load(dynamic_path)
    labels = np.load(labels_path)
    assert static.shape[0] == dynamic.shape[0] == labels.shape[0], "Mismatched sample counts"
    # Shuffle
    idx = np.arange(static.shape[0])
    np.random.shuffle(idx)
    static, dynamic, labels = static[idx], dynamic[idx], labels[idx]
    # Split
    split = int(static.shape[0] * (1 - val_split))
    train = (static[:split], dynamic[:split], labels[:split])
    val = (static[split:], dynamic[split:], labels[split:])
    return train, val

def batch_generator(static, dynamic, labels, batch_size):
    n = static.shape[0]
    while True:
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            yield (static[batch_idx], dynamic[batch_idx]), labels[batch_idx]

def main():
    args = parse_args()
    logger = get_logger(experiment_name="train_personality_model")
    logger.log_system_info()

    # Load data
    logger.logger.info("Loading features and labels...")
    (static_tr, dynamic_tr, y_tr), (static_val, dynamic_val, y_val) = load_data(
        args.static_features, args.dynamic_features, args.labels, args.val_split)
    logger.logger.info(f"Train samples: {static_tr.shape[0]}, Validation samples: {static_val.shape[0]}")

    # Model
    model = CompletePersonalityModel(
        static_dim=static_tr.shape[1],
        dynamic_dim=dynamic_tr.shape[1],
        fusion_dim=128,
        num_heads=4,
        dropout_rate=0.3
    )
    # Only train cross-attention and output dense layers
    for layer in model.layers:
        if not isinstance(layer, (tf.keras.layers.MultiHeadAttention, tf.keras.layers.Dense, tf.keras.layers.LayerNormalization, tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
            layer.trainable = False
    logger.logger.info("Set only cross-attention and output dense layers as trainable.")    # Setup learning rate scheduler - reduce LR when validation loss plateaus
    initial_learning_rate = 1e-3
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    logger.logger.info(f"Learning rate scheduler configured with initial LR: {initial_learning_rate}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='mse',
        metrics=['accuracy','mae', 'mse', r_squared_metric] + per_trait_metrics()
    )
      # Make a fake prediction to build the model
    dummy_static = np.zeros((1, static_tr.shape[1]), dtype=np.float32)
    dummy_dynamic = np.zeros((1, dynamic_tr.shape[1]), dtype=np.float32)
    _ = model((dummy_static, dummy_dynamic))
    
    # Log model architecture and metrics
    logger.log_model_architecture(model)
    
    # Log hyperparameters
    hyperparams = {
        'static_dim': static_tr.shape[1],
        'dynamic_dim': dynamic_tr.shape[1],
        'fusion_dim': 128,
        'num_heads': 4,
        'dropout_rate': 0.3,
        'batch_size': args.batch_size,
        'initial_learning_rate': initial_learning_rate,
        'optimizer': 'Adam'
    }
    logger.log_hyperparameters(hyperparams)
    # Training
    logger.logger.info("Starting training...")    # Create output directories
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for various logs
    tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
    batch_metrics_dir = os.path.join(output_dir, 'batch_metrics')
    time_tracking_dir = os.path.join(output_dir, 'time_tracking')
    lr_tracking_dir = os.path.join(output_dir, 'lr_tracking')
    
    # Create all directories
    for directory in [tensorboard_dir, batch_metrics_dir, time_tracking_dir, lr_tracking_dir]:
        os.makedirs(directory, exist_ok=True)

    # Set up checkpoints to save weights during training
    checkpoint_path = os.path.join(output_dir, 'checkpoint_weights.h5')

    # Custom callback for detailed metric logging
    class MetricLoggerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            
            # Log all metrics to the research logger
            logger.log_metrics(logs, step=epoch)
            
            # Log trait-specific metrics
            trait_metrics = {}
            for trait_idx, trait_name in enumerate(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']):
                for metric in ['mae', 'mse', 'r_squared']:
                    metric_key = f'{metric}_{trait_name[0]}'
                    if metric_key in logs:
                        trait_metrics[f'{trait_name}_{metric}'] = logs[metric_key]
            
            if trait_metrics:
                logger.log_metrics(trait_metrics, step=epoch)    # Set up callbacks list with our custom callbacks
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
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Learning rate scheduler
        lr_scheduler,
        # TensorBoard callback for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        # Our legacy custom metric logger
        MetricLoggerCallback(),
        # New Phase 4.3 callbacks for advanced monitoring
        PerformanceMonitorCallback(
            log_dir=batch_metrics_dir,
            batch_log_frequency=10,
            logger=logger
        ),
        TimeTrackingCallback(
            log_dir=time_tracking_dir,
            batch_log_frequency=20,
            logger=logger
        ),
        LearningRateTrackerCallback(
            log_dir=lr_tracking_dir,
            logger=logger
        )
    ]
    
    # Record training start time
    training_start_time = time.time()
    history = model.fit(
        batch_generator(static_tr, dynamic_tr, y_tr, args.batch_size),
        steps_per_epoch=max(1, static_tr.shape[0] // args.batch_size),
        validation_data=batch_generator(static_val, dynamic_val, y_val, args.batch_size),
        validation_steps=max(1, static_val.shape[0] // args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks    )
    logger.logger.info("Training complete. Saving model weights...")
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Log training history
    logger.logger.info("Analyzing training results...")
    
    # Extract metrics from history
    epochs_trained = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Log training summary
    logger.logger.info(f"Training completed in {epochs_trained} epochs")
    logger.logger.info(f"Final training loss: {final_loss:.4f}")
    logger.logger.info(f"Final validation loss: {final_val_loss:.4f}")
    
    # Log per-trait metrics if available
    for trait_idx, trait_name in enumerate(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']):
        trait_key = f'r_squared_{trait_name[0]}'
        if trait_key in history.history:
            final_trait_r2 = history.history[trait_key][-1]
            logger.logger.info(f"{trait_name} R²: {final_trait_r2:.4f}")
    
    # Save training history to a file for later analysis
    history_path = os.path.join(os.path.dirname(args.output), 'training_history.npy')
    np.save(history_path, history.history)
    logger.logger.info(f"Training history saved to {history_path}")
      # Save the model weights in H5 format - this works for all model types
    model.save_weights(args.output)
    logger.logger.info(f"Model weights saved to {args.output}")
    
    # Generate training visualizations
    logger.logger.info("Generating training visualizations...")
    try:
        # Import visualization functions here to avoid circular imports
        from scripts.visualize_advanced_training import (
            load_json_file, 
            create_combined_training_dashboard,
            plot_learning_rate_changes,
            plot_training_validation_curves,
            plot_batch_performance,
            plot_per_trait_performance,
            create_correlation_matrix
        )
        
        # Load data from the tracking logs
        metrics_history = load_json_file(os.path.join(batch_metrics_dir, 'metrics_history.json'))
        time_logs = load_json_file(os.path.join(time_tracking_dir, 'time_logs.json'))
        lr_logs = load_json_file(os.path.join(lr_tracking_dir, 'lr_logs.json'))
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, 'training_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate comprehensive visualizations
        logger.logger.info("Creating combined training dashboard...")
        create_combined_training_dashboard(metrics_history, time_logs, lr_logs, viz_dir)
        
        logger.logger.info("Plotting learning rate changes...")
        plot_learning_rate_changes(lr_logs, viz_dir)
        
        logger.logger.info("Plotting training/validation curves...")
        plot_training_validation_curves(metrics_history, viz_dir)
        
        logger.logger.info("Plotting batch performance...")
        plot_batch_performance(metrics_history, viz_dir)
        
        logger.logger.info("Plotting per-trait performance...")
        plot_per_trait_performance(metrics_history, viz_dir)
        
        # Create metrics correlation matrix
        try:
            logger.logger.info("Creating metrics correlation matrix...")
            create_correlation_matrix(metrics_history, viz_dir)
        except Exception as e:
            logger.logger.warning(f"Failed to create correlation matrix: {e}")
            
        # Run training time analysis
        logger.logger.info("Analyzing training time performance...")
        from scripts.analyze_training_time import (
            analyze_epoch_times,
            analyze_batch_times,
            create_training_speedup_recommendations,
            create_training_time_dashboard
        )
        
        # Create analysis directory
        analysis_dir = os.path.join(output_dir, 'training_time_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Run analysis components
        try:
            logger.logger.info("Analyzing epoch times...")
            analyze_epoch_times(time_logs, analysis_dir)
            
            logger.logger.info("Analyzing batch times...")
            analyze_batch_times(time_logs, metrics_history, analysis_dir)
            
            logger.logger.info("Creating training speedup recommendations...")
            create_training_speedup_recommendations(time_logs, metrics_history, analysis_dir)
            
            logger.logger.info("Creating training time dashboard...")
            create_training_time_dashboard(time_logs, metrics_history, analysis_dir)
            
            logger.logger.info(f"Training time analysis saved to {analysis_dir}")
        except Exception as e:
            logger.logger.warning(f"Some training time analysis components failed: {e}")
        
        # Generate attention visualizations if we have attention weights
        if hasattr(model, 'extract_attention_weights'):
            logger.logger.info("Generating attention visualizations...")
            from scripts.enhanced_attention_visualization import (
                visualize_multi_head_attention,
                visualize_attention_comparison,
                create_attention_heatmap_dashboard
            )
            
            # Create visualization directory
            attn_dir = os.path.join(output_dir, 'attention_visualizations')
            os.makedirs(attn_dir, exist_ok=True)
            
            # Extract attention weights for a small sample
            sample_size = min(10, static_val.shape[0])
            sample_static = static_val[:sample_size]
            sample_dynamic = dynamic_val[:sample_size]
            
            try:
                # Extract attention weights using the model's method
                attention_weights = model.extract_attention_weights((sample_static, sample_dynamic))
                
                # Create visualizations
                logger.logger.info("Creating multi-head attention visualization...")
                for i in range(min(3, sample_size)):
                    output_path = os.path.join(attn_dir, f'multi_head_attention_sample_{i+1}.png')
                    visualize_multi_head_attention(attention_weights, i, output_path)
                
                logger.logger.info("Creating attention comparison visualization...")
                visualize_attention_comparison(attention_weights, os.path.join(attn_dir, 'attention_comparison.png'))
                
                logger.logger.info("Creating attention dashboard...")
                create_attention_heatmap_dashboard(attention_weights, os.path.join(attn_dir, 'attention_dashboard.png'))
                
                logger.logger.info(f"Attention visualizations saved to {attn_dir}")
            except Exception as e:
                logger.logger.warning(f"Failed to generate attention visualizations: {e}")
        else:
            logger.logger.info("Skipping attention visualizations (model doesn't support attention weight extraction)")
        
        logger.logger.info("All training visualizations and analyses completed successfully.")
    except Exception as e:
        logger.logger.warning(f"Failed to generate some visualizations or analyses: {e}")
    
    logger.logger.info("Training pipeline completed successfully.")
    logger.close()

if __name__ == "__main__":
    main()
