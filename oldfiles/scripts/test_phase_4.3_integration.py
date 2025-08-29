"""
Test script for Phase 4.3 integration with the training pipeline.

This script performs a mini version of the training loop to verify that:
1. The cross-attention model with attention extraction works properly
2. The training callbacks are integrated and generate the expected output files
3. The visualizations and analysis scripts work with the generated data

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/test_phase_4.3_integration.py
    }
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.models.personality_model import CompletePersonalityModel
from utils.training_callbacks import (
    PerformanceMonitorCallback,
    TimeTrackingCallback,
    LearningRateTrackerCallback
)
from utils.logger import get_logger

def create_test_directories():
    """Create test directories for output files."""
    test_dirs = [
        'results/test_integration',
        'results/test_integration/batch_metrics',
        'results/test_integration/time_tracking',
        'results/test_integration/lr_tracking',
        'results/test_integration/visualizations',
        'results/test_integration/attention_visualizations',
        'results/test_integration/training_time_analysis'
    ]
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
    
    return test_dirs[0]  # Return the base test directory

def generate_dummy_data():
    """Generate dummy data for testing the training pipeline."""
    # Generate dummy training data
    X_train = np.random.rand(200, 10).astype(np.float32)
    y_train = np.random.rand(200, 5).astype(np.float32)  # 5 outputs for OCEAN traits
    
    # Generate dummy validation data
    X_val = np.random.rand(50, 10).astype(np.float32)
    y_val = np.random.rand(50, 5).astype(np.float32)
    
    return X_train, y_train, X_val, y_val

def main():
    # Initialize logger
    logger = get_logger(experiment_name="test_phase_4.3_integration")
    logger.logger.info("Starting Phase 4.3 integration test")
    
    # Create test directories
    test_dir = create_test_directories()
    logger.logger.info(f"Test output directory: {test_dir}")
    
    # Generate dummy data
    static_tr, y_tr, static_val, y_val = generate_dummy_data()
    dynamic_tr = np.random.rand(static_tr.shape[0], 8).astype(np.float32)
    dynamic_val = np.random.rand(static_val.shape[0], 8).astype(np.float32)
    
    logger.logger.info(f"Generated dummy training data: static={static_tr.shape}, dynamic={dynamic_tr.shape}, labels={y_tr.shape}")
    logger.logger.info(f"Generated dummy validation data: static={static_val.shape}, dynamic={dynamic_val.shape}, labels={y_val.shape}")
    
    # Create model
    model = CompletePersonalityModel(
        static_dim=static_tr.shape[1],
        dynamic_dim=dynamic_tr.shape[1],
        fusion_dim=16,
        num_heads=4,
        dropout_rate=0.3
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001,
            verbose=1
        ),
        PerformanceMonitorCallback(
            log_dir=os.path.join(test_dir, 'batch_metrics'),
            batch_log_frequency=5,
            logger=logger
        ),
        TimeTrackingCallback(
            log_dir=os.path.join(test_dir, 'time_tracking'),
            batch_log_frequency=10,
            logger=logger
        ),
        LearningRateTrackerCallback(
            log_dir=os.path.join(test_dir, 'lr_tracking'),
            logger=logger
        )
    ]
    
    # Train the model with callbacks
    logger.logger.info("Training model with callbacks...")
    history = model.fit(
        [static_tr, dynamic_tr], y_tr,
        validation_data=([static_val, dynamic_val], y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )
    
    # Check if callback output files were created
    metrics_file = os.path.join(test_dir, 'batch_metrics', 'metrics_history.json')
    time_logs_file = os.path.join(test_dir, 'time_tracking', 'time_logs.json')
    lr_logs_file = os.path.join(test_dir, 'lr_tracking', 'lr_logs.json')
    
    logger.logger.info(f"Metrics file exists: {os.path.exists(metrics_file)}")
    logger.logger.info(f"Time logs file exists: {os.path.exists(time_logs_file)}")
    logger.logger.info(f"LR logs file exists: {os.path.exists(lr_logs_file)}")
    
    # Test attention weight extraction
    logger.logger.info("Testing attention weight extraction...")
    try:
        attention_weights = model.extract_attention_weights(([static_val[:5], dynamic_val[:5]]))
        logger.logger.info(f"Extracted attention weights with shape: {attention_weights.shape}")
        
        # Test attention visualization
        from scripts.enhanced_attention_visualization import (
            visualize_multi_head_attention,
            visualize_attention_comparison,
            create_attention_heatmap_dashboard
        )
        
        # Create visualization directory
        attn_dir = os.path.join(test_dir, 'attention_visualizations')
        os.makedirs(attn_dir, exist_ok=True)
        
        # Generate visualizations
        logger.logger.info("Generating attention visualizations...")
        visualize_multi_head_attention(
            attention_weights, 
            sample_idx=0, 
            output_path=os.path.join(attn_dir, 'multi_head_attention.png')
        )
        
        visualize_attention_comparison(
            attention_weights,
            output_path=os.path.join(attn_dir, 'attention_comparison.png')
        )
        
        create_attention_heatmap_dashboard(
            attention_weights,
            output_path=os.path.join(attn_dir, 'attention_dashboard.png')
        )
        
        logger.logger.info("Attention visualizations generated successfully.")
    except Exception as e:
        logger.logger.error(f"Error testing attention weight extraction: {e}")
    
    # Test advanced visualization
    logger.logger.info("Testing advanced training visualizations...")
    try:
        from scripts.visualize_advanced_training import (
            load_json_file, 
            plot_learning_rate_changes,
            plot_batch_performance,
            plot_training_validation_curves,
            plot_training_time_analysis,
            create_combined_training_dashboard
        )
        
        # Load data from log files
        metrics_history = load_json_file(metrics_file)
        time_logs = load_json_file(time_logs_file)
        lr_logs = load_json_file(lr_logs_file)
        
        # Create visualization directory
        viz_dir = os.path.join(test_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations
        plot_learning_rate_changes(lr_logs, viz_dir)
        plot_batch_performance(metrics_history, viz_dir)
        plot_training_validation_curves(metrics_history, viz_dir)
        plot_training_time_analysis(time_logs, viz_dir)
        create_combined_training_dashboard(metrics_history, time_logs, lr_logs, viz_dir)
        
        logger.logger.info("Training visualizations generated successfully.")
    except Exception as e:
        logger.logger.error(f"Error generating training visualizations: {e}")
    
    # Test training time analysis
    logger.logger.info("Testing training time analysis...")
    try:
        from scripts.analyze_training_time import (
            analyze_epoch_times,
            analyze_batch_times,
            create_training_speedup_recommendations,
            create_training_time_dashboard
        )
        
        # Create analysis directory
        analysis_dir = os.path.join(test_dir, 'training_time_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Run analysis
        analyze_epoch_times(time_logs, analysis_dir)
        analyze_batch_times(time_logs, metrics_history, analysis_dir)
        create_training_speedup_recommendations(time_logs, metrics_history, analysis_dir)
        create_training_time_dashboard(time_logs, metrics_history, analysis_dir)
        
        logger.logger.info("Training time analysis completed successfully.")
    except Exception as e:
        logger.logger.error(f"Error in training time analysis: {e}")
    
    logger.logger.info("Phase 4.3 integration test completed")
    logger.close()

if __name__ == "__main__":
    main()
