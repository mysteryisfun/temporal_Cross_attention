"""
Testing script for Phase 4.3 components.

This script tests all the components we've implemented for Phase 4.3:
1. Training callbacks
2. Advanced visualization tools
3. Attention visualization
4. Training time analysis

Usage:
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python scripts/test_phase_4.3_components.py
    }
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.training_callbacks import (
    PerformanceMonitorCallback,
    TimeTrackingCallback,
    LearningRateTrackerCallback
)
from utils.logger import get_logger

def create_test_directories():
    """Create test directories for output files."""
    test_dirs = [
        'results/test_phase_4.3',
        'results/test_phase_4.3/batch_metrics',
        'results/test_phase_4.3/time_tracking',
        'results/test_phase_4.3/lr_tracking',
        'results/test_phase_4.3/visualizations',
        'results/test_phase_4.3/attention_visualizations',
        'results/test_phase_4.3/training_time_analysis'
    ]
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
    
    return test_dirs[0]  # Return the base test directory

def generate_dummy_data():
    """Generate dummy data for testing the training pipeline."""
    # Generate dummy training data
    X_train = np.random.rand(100, 10).astype(np.float32)
    y_train = np.random.rand(100, 5).astype(np.float32)  # 5 outputs for OCEAN traits
    
    # Generate dummy validation data
    X_val = np.random.rand(30, 10).astype(np.float32)
    y_val = np.random.rand(30, 5).astype(np.float32)
    
    return X_train, y_train, X_val, y_val

def create_dummy_model():
    """Create a simple model for testing callbacks."""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(16, activation='relu'),
        Dense(5, activation='linear')  # 5 outputs for OCEAN traits
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def test_training_callbacks(X_train, y_train, X_val, y_val, test_dir, logger):
    """Test the training callbacks functionality."""
    logger.logger.info("Testing training callbacks...")
    
    # Create model
    model = create_dummy_model()
    
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        callbacks=callbacks,
        verbose=2
    )
    
    # Check if callback output files were created
    metrics_file = os.path.join(test_dir, 'batch_metrics', 'metrics_history.json')
    time_logs_file = os.path.join(test_dir, 'time_tracking', 'time_logs.json')
    lr_logs_file = os.path.join(test_dir, 'lr_tracking', 'lr_logs.json')
    
    metrics_exist = os.path.exists(metrics_file)
    time_logs_exist = os.path.exists(time_logs_file)
    lr_logs_exist = os.path.exists(lr_logs_file)
    
    logger.logger.info(f"Metrics file exists: {metrics_exist}")
    logger.logger.info(f"Time logs file exists: {time_logs_exist}")
    logger.logger.info(f"LR logs file exists: {lr_logs_exist}")
    
    # Return testing status and file paths
    return all([metrics_exist, time_logs_exist, lr_logs_exist]), {
        'metrics_file': metrics_file,
        'time_logs_file': time_logs_file,
        'lr_logs_file': lr_logs_file
    }

def test_visualization_script(file_paths, test_dir, logger):
    """Test the visualization script functionality."""
    logger.logger.info("Testing visualization script...")
    
    # Import visualization functions
    try:
        from scripts.visualize_advanced_training import (
            load_json_file, 
            plot_learning_rate_changes,
            plot_batch_performance,
            plot_training_validation_curves,
            plot_training_time_analysis
        )
        
        # Load data from log files
        metrics_history = load_json_file(file_paths['metrics_file'])
        time_logs = load_json_file(file_paths['time_logs_file'])
        lr_logs = load_json_file(file_paths['lr_logs_file'])
        
        # Output directory for visualizations
        vis_dir = os.path.join(test_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate visualizations
        logger.logger.info("Generating learning rate change visualization...")
        plot_learning_rate_changes(lr_logs, vis_dir)
        
        logger.logger.info("Generating batch performance visualization...")
        plot_batch_performance(metrics_history, vis_dir)
        
        logger.logger.info("Generating training/validation curves...")
        plot_training_validation_curves(metrics_history, vis_dir)
        
        logger.logger.info("Generating training time analysis...")
        plot_training_time_analysis(time_logs, vis_dir)
        
        # Check if visualization files were created
        vis_files = [
            os.path.join(vis_dir, 'learning_rate_changes.png'),
            os.path.join(vis_dir, 'batch_performance.png'),
            os.path.join(vis_dir, 'train_val_loss.png'),
            os.path.join(vis_dir, 'training_time_analysis.png')
        ]
        
        vis_success = all([os.path.exists(f) for f in vis_files])
        logger.logger.info(f"Visualization generation successful: {vis_success}")
        
        return vis_success
        
    except Exception as e:
        logger.logger.error(f"Error testing visualization script: {e}")
        return False

def test_attention_visualization(test_dir, logger):
    """Test the attention visualization functionality."""
    logger.logger.info("Testing attention visualization...")
    
    try:
        # Create dummy attention weights
        # Shape: [batch_size, num_heads, query_dim, key_dim]
        attention_weights = np.random.rand(5, 4, 1, 1).astype(np.float32)
        
        # Import visualization functions (partial test)
        from scripts.enhanced_attention_visualization import (
            visualize_multi_head_attention,
            visualize_attention_comparison,
            create_attention_heatmap_dashboard
        )
        
        # Output directory
        attn_dir = os.path.join(test_dir, 'attention_visualizations')
        os.makedirs(attn_dir, exist_ok=True)
        
        # Generate visualizations
        logger.logger.info("Generating multi-head attention visualization...")
        visualize_multi_head_attention(
            attention_weights, 
            sample_idx=0, 
            output_path=os.path.join(attn_dir, 'multi_head_attention.png')
        )
        
        logger.logger.info("Generating attention comparison visualization...")
        visualize_attention_comparison(
            attention_weights,
            output_path=os.path.join(attn_dir, 'attention_comparison.png')
        )
        
        logger.logger.info("Generating attention dashboard...")
        create_attention_heatmap_dashboard(
            attention_weights,
            output_path=os.path.join(attn_dir, 'attention_dashboard.png')
        )
        
        # Check if visualization files were created
        attn_files = [
            os.path.join(attn_dir, 'multi_head_attention.png'),
            os.path.join(attn_dir, 'attention_comparison.png'),
            os.path.join(attn_dir, 'attention_dashboard.png')
        ]
        
        attn_success = all([os.path.exists(f) for f in attn_files])
        logger.logger.info(f"Attention visualization generation successful: {attn_success}")
        
        return attn_success
        
    except Exception as e:
        logger.logger.error(f"Error testing attention visualization: {e}")
        return False

def test_training_time_analysis(file_paths, test_dir, logger):
    """Test the training time analysis functionality."""
    logger.logger.info("Testing training time analysis...")
    
    try:
        # Import functions from analysis script
        from scripts.analyze_training_time import (
            analyze_epoch_times,
            analyze_batch_times,
            create_training_speedup_recommendations,
            create_training_time_dashboard
        )
        
        # Load data from log files
        with open(file_paths['time_logs_file'], 'r') as f:
            time_logs = json.load(f)
        
        with open(file_paths['metrics_file'], 'r') as f:
            metrics_logs = json.load(f)
        
        # Output directory
        analysis_dir = os.path.join(test_dir, 'training_time_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Run analysis functions
        logger.logger.info("Analyzing epoch times...")
        analyze_epoch_times(time_logs, analysis_dir)
        
        logger.logger.info("Analyzing batch times...")
        analyze_batch_times(time_logs, metrics_logs, analysis_dir)
        
        logger.logger.info("Creating training speedup recommendations...")
        create_training_speedup_recommendations(time_logs, metrics_logs, analysis_dir)
        
        logger.logger.info("Creating training time dashboard...")
        create_training_time_dashboard(time_logs, metrics_logs, analysis_dir)
        
        # Check if analysis files were created
        analysis_files = [
            os.path.join(analysis_dir, 'epoch_time_analysis.png'),
            os.path.join(analysis_dir, 'epoch_time_stats.json'),
            os.path.join(analysis_dir, 'training_speedup_recommendations.json'),
            os.path.join(analysis_dir, 'training_time_dashboard.png')
        ]
        
        # Batch time analysis may not be available if no batch times were logged
        if time_logs.get('batch_times') and len(time_logs['batch_times']) > 0:
            analysis_files.extend([
                os.path.join(analysis_dir, 'batch_time_analysis.png'),
                os.path.join(analysis_dir, 'batch_time_stats.json')
            ])
        
        analysis_success = all([os.path.exists(f) for f in analysis_files])
        logger.logger.info(f"Training time analysis successful: {analysis_success}")
        
        return analysis_success
        
    except Exception as e:
        logger.logger.error(f"Error testing training time analysis: {e}")
        return False

def main():
    # Initialize logger
    logger = get_logger(experiment_name="test_phase_4.3")
    logger.logger.info("Starting Phase 4.3 component tests")
    
    # Create test directories
    test_dir = create_test_directories()
    logger.logger.info(f"Test output directory: {test_dir}")
    
    # Generate dummy data
    X_train, y_train, X_val, y_val = generate_dummy_data()
    logger.logger.info(f"Generated dummy training data: {X_train.shape}, {y_train.shape}")
    logger.logger.info(f"Generated dummy validation data: {X_val.shape}, {y_val.shape}")
    
    # Test components
    test_results = {}
    
    # 1. Test training callbacks
    callbacks_success, file_paths = test_training_callbacks(
        X_train, y_train, X_val, y_val, test_dir, logger
    )
    test_results['training_callbacks'] = callbacks_success
    
    # 2. Test visualization script
    if callbacks_success:
        vis_success = test_visualization_script(file_paths, test_dir, logger)
        test_results['visualization_script'] = vis_success
    else:
        logger.logger.warning("Skipping visualization script test due to callback test failure")
        test_results['visualization_script'] = False
    
    # 3. Test attention visualization
    attn_success = test_attention_visualization(test_dir, logger)
    test_results['attention_visualization'] = attn_success
    
    # 4. Test training time analysis
    if callbacks_success:
        analysis_success = test_training_time_analysis(file_paths, test_dir, logger)
        test_results['training_time_analysis'] = analysis_success
    else:
        logger.logger.warning("Skipping training time analysis test due to callback test failure")
        test_results['training_time_analysis'] = False
    
    # Print summary of test results
    logger.logger.info("\n===== TEST RESULTS SUMMARY =====")
    all_passed = True
    for component, success in test_results.items():
        status = "PASSED" if success else "FAILED"
        logger.logger.info(f"{component}: {status}")
        all_passed = all_passed and success
    
    if all_passed:
        logger.logger.info("\n✅ All Phase 4.3 components are working correctly!")
    else:
        logger.logger.warning("\n⚠️ Some Phase 4.3 components failed testing. Check the logs for details.")
    
    # Save test results
    with open(os.path.join(test_dir, 'test_results.json'), 'w') as f:
        json.dump({k: "passed" if v else "failed" for k, v in test_results.items()}, f, indent=2)
    
    logger.logger.info(f"Test results saved to {os.path.join(test_dir, 'test_results.json')}")
    logger.logger.info("Phase 4.3 component tests completed")
    
    logger.close()

if __name__ == "__main__":
    main()
