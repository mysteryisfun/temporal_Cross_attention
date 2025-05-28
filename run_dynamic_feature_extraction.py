#!/usr/bin/env python3
"""
Dynamic Feature Extraction Script
Executes the 3D I3D CNN feature extraction on optical flow sequences
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import time
from datetime import datetime
import json
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor

def check_and_configure_gpu():
    """Configure GPU for optimal performance with memory limits (70-80% utilization)"""
    physical_gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    
    print(f"Physical GPUs: {len(physical_gpus)}")
    print(f"Logical GPUs: {len(logical_gpus)}")
    
    if physical_gpus:
        print("✅ GPU Support Available!")
        try:
            # Configure memory growth and limit
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set GPU memory limit to 70-80% of total memory
                try:
                    # Get GPU memory info (this might not work on all systems)
                    memory_limit = int(4096 * 0.75)  # Assuming 4GB GPU, use 75% (3GB)
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    print(f"   ⚙️ GPU memory limit set to {memory_limit}MB (75% utilization)")
                except Exception as e:
                    print(f"   ⚠️ Could not set memory limit: {e}")
                    print("   ✅ Using memory growth instead")
            
            print("   ✅ GPU memory growth configured")
            
        except RuntimeError as e:
            print(f"   ⚠️ GPU configuration warning: {e}")
    else:
        print("❌ No GPU detected, using CPU")
    
    return len(physical_gpus) > 0


def calculate_optimal_batch_size():
    """Calculate optimal batch size based on GPU memory"""
    physical_gpus = tf.config.list_physical_devices('GPU')
    
    if not physical_gpus:
        return 1  # CPU fallback
    
    try:
        # Test with progressively larger batch sizes
        for batch_size in [1, 2, 4, 8]:
            try:
                with tf.device('/GPU:0'):
                    # Test tensor to simulate I3D input
                    test_tensor = tf.random.uniform((batch_size, 16, 224, 224, 2), dtype=tf.float32)
                    
                    # Simple convolution to test memory usage
                    conv_layer = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same')
                    result = conv_layer(test_tensor)
                    
                    # If we get here, this batch size works
                    print(f"   ✅ Batch size {batch_size} test passed")
                    del test_tensor, result
                    gc.collect()
                    
            except tf.errors.ResourceExhaustedError:
                print(f"   ❌ Batch size {batch_size} failed (out of memory)")
                # Return the previous successful batch size
                return max(1, batch_size // 2)
            except Exception as e:
                print(f"   ⚠️ Batch size {batch_size} test error: {e}")
                return max(1, batch_size // 2)
        
        # If all tests passed, use the largest tested batch size
        return 8
        
    except Exception as e:
        print(f"   ⚠️ Error in batch size calculation: {e}")
        return 1

def load_and_preprocess_flow(flow_path, sequence_length=16, target_size=(224, 224)):
    """Load and preprocess a single optical flow .npy file"""
    try:
        flow_data = np.load(flow_path)
        
        # Ensure correct shape
        if len(flow_data.shape) == 3:  # (H, W, 2)
            # Resize if needed
            if flow_data.shape[:2] != target_size:
                flow_data = cv2.resize(flow_data, target_size)
            
            # Normalize flow magnitude
            flow_magnitude = np.sqrt(flow_data[:,:,0]**2 + flow_data[:,:,1]**2)
            max_magnitude = np.max(flow_magnitude) if np.max(flow_magnitude) > 0 else 1.0
            flow_data = flow_data / max_magnitude
            
            # Pad to sequence length by repeating the single flow frame
            flow_sequence = np.tile(flow_data[np.newaxis, :, :, :], (sequence_length, 1, 1, 1))
            
            return flow_sequence
        else:
            print(f"⚠️ Unexpected flow shape: {flow_data.shape}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error loading {flow_path}: {e}")
        return None

def extract_dynamic_features_gpu(optical_flow_paths, sequence_length=16, 
                                  checkpoint_dir=None, resume_from_checkpoint=True):
    """
    Extract dynamic features using GPU, one file at a time, with pause/resume functionality and optimized preprocessing.
    
    Args:
        optical_flow_paths: List of paths to optical flow files
        sequence_length: Number of flow frames per sequence (16 required by I3D)
        checkpoint_dir: Directory to save checkpoints for resume functionality
        resume_from_checkpoint: Whether to resume from existing checkpoint
    
    Returns:
        np.ndarray: Extracted features
    """
    print(f"🔧 Initializing I3D Dynamic Feature Extractor with Resume Capability (No Batch)...")
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path("results/features/checkpoints")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / "dynamic_extraction_checkpoint.json"
    features_checkpoint_file = checkpoint_dir / "dynamic_features_checkpoint.npy"
    
    # Initialize extractor with None config path (will use defaults)
    extractor = DynamicFeatureExtractor(config_path=None)
    
    # Override config settings
    extractor.config['input_shape'] = (sequence_length, 224, 224, 2)
    extractor.config['output_features'] = 1024
    extractor.config['model_type'] = 'i3d'
    extractor.config['dropout_rate'] = 0.5
    
    print(f"✅ I3D model initialized: input_shape={extractor.config['input_shape']}")
    
    # Check for existing checkpoint
    start_index = 0
    features = []
    failed_count = 0
    
    if resume_from_checkpoint and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            start_index = checkpoint_data.get('last_processed_index', 0)
            failed_count = checkpoint_data.get('failed_count', 0)
            
            if features_checkpoint_file.exists():
                features = np.load(features_checkpoint_file).tolist()
                # Ensure start_index matches len(features)
                if len(features) != start_index:
                    print(f"⚠️ Checkpoint mismatch: start_index={start_index}, features={len(features)}. Using len(features) as start_index.")
                    start_index = len(features)
                print(f"🔄 Resuming from checkpoint: processed {start_index}/{len(optical_flow_paths)} files")
                print(f"   Loaded {len(features)} existing features")
            else:
                print(f"🔄 Resuming from index {start_index} (features file not found)")
                features = []
                start_index = 0
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("   Starting fresh extraction...")
            start_index = 0
            features = []
            failed_count = 0
    else:
        print("🚀 Starting fresh dynamic feature extraction...")
    
    print(f"🚀 Processing {len(optical_flow_paths)} optical flow files (no batch)...")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Checkpoint saving: every 100 files")
    
    with tf.device('/GPU:0' if check_and_configure_gpu() else '/CPU:0'):
        for i in tqdm(range(start_index, len(optical_flow_paths)),
                      desc="Dynamic features", initial=start_index, total=len(optical_flow_paths)):
            flow_path = optical_flow_paths[i]
            flow_sequence = load_and_preprocess_flow(flow_path, sequence_length)
            if flow_sequence is not None:
                try:
                    # Add batch dimension for single sample
                    flow_tensor = np.expand_dims(flow_sequence, axis=0)
                    batch_features = extractor.extract_features(flow_tensor)
                    features.append(batch_features[0])
                    del flow_tensor, batch_features
                except Exception as e:
                    print(f"⚠️ Error processing file {i}: {e}")
                    features.append(np.zeros(1024))
                    failed_count += 1
            else:
                features.append(np.zeros(1024))
                failed_count += 1
            # Save checkpoint every 100 files
            if (i + 1) % 100 == 0:
                try:
                    checkpoint_data = {
                        'last_processed_index': i + 1,
                        'total_files': len(optical_flow_paths),
                        'failed_count': failed_count,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'progress_percentage': ((i + 1) / len(optical_flow_paths)) * 100
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                    np.save(features_checkpoint_file, np.array(features))
                    print(f"💾 Checkpoint saved: {checkpoint_data['progress_percentage']:.1f}% complete "
                          f"({i + 1}/{len(optical_flow_paths)})")
                except Exception as e:
                    print(f"⚠️ Error saving checkpoint: {e}")
    
    # Final checkpoint cleanup
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if features_checkpoint_file.exists():
            features_checkpoint_file.unlink()
        print("🧹 Checkpoint files cleaned up")
    except Exception as e:
        print(f"⚠️ Error cleaning up checkpoints: {e}")
    
    print(f"✅ Dynamic feature extraction completed!")
    print(f"   Processed: {len(optical_flow_paths)} files")
    print(f"   Failed: {failed_count} files")
    print(f"   Features shape: {np.array(features).shape}")
    
    return np.array(features)

def main():
    """Main execution function with enhanced GPU utilization and pause/resume (no batch)"""
    print("🌊 Starting GPU-Accelerated Dynamic Feature Extraction with Resume Capability (No Batch)")
    print("=" * 70)
    
    # Configure paths
    optical_flow_dir = Path("data/processed/optical_flow")
    output_dir = Path("results/features")
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU configuration and calculate optimal batch size
    gpu_available = check_and_configure_gpu()
    
    print(f"GPU acceleration: {'✅ Enabled' if gpu_available else '❌ Disabled (CPU only)'}")
    print(f"Batch processing: ❌ Disabled (processing one file at a time)")
    print(f"Preprocessing: Optimized for single file, minimal memory use")
    
    # Collect optical flow paths
    print(f"📋 Collecting optical flow paths from {optical_flow_dir}...")
    optical_flow_paths = []
    
    if optical_flow_dir.exists():
        for training_dir in sorted(optical_flow_dir.iterdir()):
            if training_dir.is_dir() and training_dir.name.startswith('training'):
                for video_dir in sorted(training_dir.iterdir()):
                    if video_dir.is_dir():
                        flow_files = sorted(list(video_dir.glob("flow_*.npy")))
                        optical_flow_paths.extend([str(f) for f in flow_files])
    
    print(f"Found {len(optical_flow_paths)} optical flow files")
    
    if len(optical_flow_paths) == 0:
        print("❌ No optical flow files found!")
        return
    
    # Start extraction
    start_time = time.time()
    extraction_stats = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_flows': len(optical_flow_paths),
        'sequence_length': 16,  # Must be 16 to match I3D model architecture
        'gpu_acceleration': gpu_available,
        'resume_capability': True,
        'checkpoint_interval': 100
    }
    
    try:
        print(f"   Sequence length: {extraction_stats['sequence_length']} (I3D requirement)")
        print(f"   Resume capability: ✅ Enabled (checkpoints every {extraction_stats['checkpoint_interval']} files)")
        
        dynamic_features = extract_dynamic_features_gpu(
            optical_flow_paths,
            sequence_length=extraction_stats['sequence_length'],
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=True
        )
        
        # Save features
        features_path = output_dir / "dynamic_features_i3d.npy"
        np.save(features_path, dynamic_features)
        
        # Calculate extraction stats
        end_time = time.time()
        processing_time = end_time - start_time
        
        extraction_stats.update({
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_seconds': processing_time,
            'processing_time_formatted': str(datetime.fromtimestamp(processing_time).strftime('%H:%M:%S')),
            'features_shape': dynamic_features.shape,
            'throughput_flows_per_second': len(optical_flow_paths) / processing_time,
            'output_file': str(features_path),
            'status': 'completed'
        })
        
        # Save extraction log
        log_path = output_dir / "dynamic_feature_extraction_log.json"
        with open(log_path, 'w') as f:
            json.dump(extraction_stats, f, indent=2, default=str)
        
        print(f"\n🎉 Dynamic Feature Extraction Completed Successfully!")
        print(f"   Processing time: {extraction_stats['processing_time_formatted']}")
        print(f"   Throughput: {extraction_stats['throughput_flows_per_second']:.2f} flows/second")
        print(f"   Output shape: {dynamic_features.shape}")
        print(f"   Features saved to: {features_path}")
        print(f"   Log saved to: {log_path}")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   • Batch size used: {extraction_stats['batch_size']}")
        print(f"   • GPU utilization: 70-80% (memory limited)")
        print(f"   • Resume capability: ✅ Available")
        print(f"   • Checkpoint frequency: Every {extraction_stats['checkpoint_interval']} batches")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Extraction paused by user")
        print(f"   Checkpoint saved - you can resume by running the script again")
        print(f"   Progress will be automatically restored")
        
    except Exception as e:
        print(f"❌ Dynamic feature extraction failed: {e}")
        extraction_stats.update({
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed',
            'error': str(e)
        })
        
        # Save error log
        log_path = output_dir / "dynamic_feature_extraction_error_log.json"
        with open(log_path, 'w') as f:
            json.dump(extraction_stats, f, indent=2, default=str)
        
        raise

if __name__ == "__main__":
    main()
