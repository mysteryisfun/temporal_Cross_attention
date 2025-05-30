#!/usr/bin/env python3
"""
Dynamic Feature Extraction Script - Part 2
Processes the second half of the dataset (e.g., samples starting from 40,001).
"""

from pathlib import Path
import json
import numpy as np
import time
from datetime import datetime
from run_dynamic_feature_extraction import check_and_configure_gpu, extract_dynamic_features_gpu

def main():
    """Main execution function for Part 2 (second half of the dataset)"""
    print("🌊 Starting GPU-Accelerated Dynamic Feature Extraction - Part 2")
    print("=" * 70)

    optical_flow_dir = Path("data/processed/optical_flow")
    output_dir = Path("results/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_available = check_and_configure_gpu()
    print(f"GPU acceleration: {'✅ Enabled' if gpu_available else '❌ Disabled (CPU only)'}")
    print(f"Batch processing: ❌ Disabled (processing one file at a time)")
    print(f"Preprocessing: Optimized for single file, minimal memory use")
    print(f"Checkpointing/Resume: ❌ Disabled (runs in one go)")

    print(f"📋 Collecting optical flow paths from {optical_flow_dir}...")
    optical_flow_paths = []
    if optical_flow_dir.exists():
        for training_dir in sorted(optical_flow_dir.iterdir()):
            if training_dir.is_dir() and training_dir.name.startswith('training'):
                for video_dir in sorted(training_dir.iterdir()):
                    if video_dir.is_dir():
                        flow_files = sorted(list(video_dir.glob("flow_*.npy")))
                        optical_flow_paths.extend([str(f) for f in flow_files])
    print(f"Found {len(optical_flow_paths)} optical flow files\n")

    # Process only the second half of the dataset (starting from 40,001)
    optical_flow_paths = optical_flow_paths[40000:]
    print(f"Processing samples starting from 40,001\n")

    if len(optical_flow_paths) == 0:
        print("❌ No optical flow files found!")
        return

    start_time = time.time()
    extraction_stats = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_flows': len(optical_flow_paths),
        'sequence_length': 16,
        'gpu_acceleration': gpu_available
    }
    try:
        print(f"   Sequence length: {extraction_stats['sequence_length']} (I3D requirement)")
        print(f"   Resume capability: ❌ Disabled (no checkpointing)\n")
        dynamic_features = extract_dynamic_features_gpu(
            optical_flow_paths,
            sequence_length=extraction_stats['sequence_length']
        )
        features_path = output_dir / "dynamic_features_part2.npy"
        np.save(features_path, dynamic_features)
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
        log_path = output_dir / "dynamic_feature_extraction_part2_log.json"
        with open(log_path, 'w') as f:
            json.dump(extraction_stats, f, indent=2, default=str)
        print(f"\n🎉 Dynamic Feature Extraction Part 2 Completed Successfully!")
        print(f"   Processing time: {extraction_stats['processing_time_formatted']}")
        print(f"   Features saved to: {features_path}")
        print(f"   Log saved to: {log_path}\n")
    except Exception as e:
        print(f"❌ Dynamic feature extraction failed: {e}")
        extraction_stats.update({
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed',
            'error': str(e)
        })
        log_path = output_dir / "dynamic_feature_extraction_part2_error_log.json"
        with open(log_path, 'w') as f:
            json.dump(extraction_stats, f, indent=2, default=str)
        raise

if __name__ == "__main__":
    main()
