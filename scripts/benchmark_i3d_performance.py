#!/usr/bin/env python3
"""
I3D Performance Benchmark Script

This script benchmarks the I3D motion extractor with different frame counts
to analyze execution time, memory usage, and determine optimal configuration.

Usage:
    python scripts/benchmark_i3d_performance.py
"""

import torch
import time
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import json
from datetime import datetime
import random

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.motion_extractor.i3d_extractor import I3DMotionExtractor

class I3DPerformanceBenchmark:
    """Benchmark I3D motion extractor performance"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize benchmark"""
        self.device = device
        self.results = {
            'benchmark_time': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'configurations': []
        }
        
        # Find test videos
        self.video_dir = Path("data/raw/videos")
        if self.video_dir.exists():
            self.test_videos = list(self.video_dir.glob("*.webm"))[:5]  # Use 5 test videos
        else:
            print("Warning: Video directory not found, will use synthetic data")
            self.test_videos = []
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            return {'ram_mb': ram_mb, 'gpu_mb': gpu_mb, 'gpu_max_mb': gpu_max_mb}
        
        return {'ram_mb': ram_mb, 'gpu_mb': 0, 'gpu_max_mb': 0}
    
    def benchmark_configuration(self, num_frames, num_runs=10):
        """
        Benchmark I3D with specific frame configuration
        
        Args:
            num_frames: Number of frames to use
            num_runs: Number of benchmark runs
        """
        print(f"\n=== Benchmarking {num_frames} frames configuration ===")
        
        # Initialize extractor
        extractor = I3DMotionExtractor(device=self.device)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        times = []
        memory_usage = []
        feature_stats = []
        
        # Warmup runs
        print("Warming up...")
        for _ in range(3):
            if self.test_videos:
                video_path = str(random.choice(self.test_videos))
                try:
                    synthetic_warmup = torch.randn(1, 3, num_frames, 224, 224).to(self.device)
                    _ = extractor.extract_features(synthetic_warmup)
                except:
                    print(f"Warning: Warmup failed")
                    break
        
        print(f"Running {num_runs} benchmark iterations...")
        
        for i in range(num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Record memory before
            mem_before = self.get_memory_usage()
            
            # Time the extraction
            start_time = time.time()
            
            try:
                if self.test_videos and i % 2 == 0:  # Alternate between real and synthetic
                    video_path = str(random.choice(self.test_videos))
                    features = extractor.extract_features(video_path, num_frames=num_frames)
                else:
                    # Use synthetic video data for consistent benchmarking
                    # Correct tensor shape: (batch, channels, temporal, height, width)
                    synthetic_video = torch.randn(1, 3, num_frames, 224, 224).to(self.device)
                    features = extractor.extract_features(synthetic_video)
                
                # Force GPU synchronization for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Record memory after
                mem_after = self.get_memory_usage()
                
                times.append(execution_time)
                memory_usage.append(mem_after)
                
                # Feature statistics
                if isinstance(features, torch.Tensor):
                    feature_stats.append({
                        'shape': list(features.shape),
                        'mean': float(features.mean().cpu()),
                        'std': float(features.std().cpu()),
                        'min': float(features.min().cpu()),
                        'max': float(features.max().cpu())
                    })
                
                print(f"Run {i+1}: {execution_time:.4f}s, GPU: {mem_after['gpu_mb']:.1f}MB")
                
            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                continue
        
        # Calculate statistics
        if times:
            config_results = {
                'num_frames': num_frames,
                'timing': {
                    'mean_time_s': float(np.mean(times)),
                    'std_time_s': float(np.std(times)),
                    'min_time_s': float(np.min(times)),
                    'max_time_s': float(np.max(times)),
                    'median_time_s': float(np.median(times)),
                    'fps': float(num_frames / np.mean(times)),
                    'throughput_videos_per_second': float(1.0 / np.mean(times))
                },
                'memory': {
                    'peak_gpu_mb': float(max([m['gpu_max_mb'] for m in memory_usage])),
                    'avg_gpu_mb': float(np.mean([m['gpu_mb'] for m in memory_usage])),
                    'peak_ram_mb': float(max([m['ram_mb'] for m in memory_usage]))
                },
                'features': {
                    'output_shape': feature_stats[0]['shape'] if feature_stats else None,
                    'feature_dimensionality': feature_stats[0]['shape'][-1] if feature_stats else None,
                    'avg_feature_stats': {
                        'mean': float(np.mean([f['mean'] for f in feature_stats])) if feature_stats else 0,
                        'std': float(np.mean([f['std'] for f in feature_stats])) if feature_stats else 0
                    }
                },
                'num_runs': len(times),
                'success_rate': len(times) / num_runs
            }
            
            self.results['configurations'].append(config_results)
            
            # Print summary
            print(f"\nResults for {num_frames} frames:")
            print(f"  Average time: {config_results['timing']['mean_time_s']:.4f} ± {config_results['timing']['std_time_s']:.4f}s")
            print(f"  FPS: {config_results['timing']['fps']:.2f}")
            print(f"  Peak GPU memory: {config_results['memory']['peak_gpu_mb']:.1f}MB")
            print(f"  Feature shape: {config_results['features']['output_shape']}")
        else:
            print(f"No successful runs for {num_frames} frames configuration")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("=== I3D Motion Extractor Performance Benchmark ===")
        print(f"Device: {self.device}")
        print(f"Test videos: {len(self.test_videos)}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Test different frame configurations
        frame_configs = [8, 16, 24, 32]  # Test multiple frame counts
        
        for num_frames in frame_configs:
            try:
                self.benchmark_configuration(num_frames, num_runs=5)
            except Exception as e:
                print(f"Failed to benchmark {num_frames} frames: {e}")
                continue
            
            # Small delay between configurations
            time.sleep(2)
        
        # Save results
        self.save_results()
        
        # Generate comparison
        self.generate_comparison()
    
    def save_results(self):
        """Save benchmark results to file"""
        results_dir = Path("experiments/i3d_benchmark")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"i3d_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def generate_comparison(self):
        """Generate comparison plots and summary"""
        if len(self.results['configurations']) < 2:
            print("Need at least 2 configurations for comparison")
            return
        
        configs = self.results['configurations']
        
        # Extract data for plotting
        frame_counts = [c['num_frames'] for c in configs]
        avg_times = [c['timing']['mean_time_s'] for c in configs]
        fps_values = [c['timing']['fps'] for c in configs]
        gpu_memory = [c['memory']['peak_gpu_mb'] for c in configs]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('I3D Motion Extractor Performance Analysis', fontsize=16, fontweight='bold')
        
        # Execution time comparison
        axes[0, 0].bar(frame_counts, avg_times, color='skyblue', alpha=0.8)
        axes[0, 0].set_xlabel('Number of Frames')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Execution Time vs Frame Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add time annotations
        for i, (frames, time_val) in enumerate(zip(frame_counts, avg_times)):
            axes[0, 0].text(frames, time_val + 0.001, f'{time_val:.3f}s', 
                           ha='center', va='bottom', fontweight='bold')
        
        # FPS comparison
        axes[0, 1].bar(frame_counts, fps_values, color='lightcoral', alpha=0.8)
        axes[0, 1].set_xlabel('Number of Frames')
        axes[0, 1].set_ylabel('Effective FPS')
        axes[0, 1].set_title('Processing FPS vs Frame Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory usage comparison
        axes[1, 0].bar(frame_counts, gpu_memory, color='lightgreen', alpha=0.8)
        axes[1, 0].set_xlabel('Number of Frames')
        axes[1, 0].set_ylabel('Peak GPU Memory (MB)')
        axes[1, 0].set_title('GPU Memory Usage vs Frame Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency ratio (FPS per MB)
        efficiency = [fps/mem for fps, mem in zip(fps_values, gpu_memory)]
        axes[1, 1].bar(frame_counts, efficiency, color='gold', alpha=0.8)
        axes[1, 1].set_xlabel('Number of Frames')
        axes[1, 1].set_ylabel('FPS per MB GPU Memory')
        axes[1, 1].set_title('Memory Efficiency vs Frame Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("experiments/i3d_benchmark")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = results_dir / f"i3d_performance_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Frames':<8} {'Time(s)':<10} {'FPS':<8} {'GPU(MB)':<10} {'Efficiency':<12} {'Recommendation'}")
        print("-" * 80)
        
        for config in configs:
            frames = config['num_frames']
            time_s = config['timing']['mean_time_s']
            fps = config['timing']['fps']
            gpu_mb = config['memory']['peak_gpu_mb']
            eff = fps / gpu_mb if gpu_mb > 0 else 0
            
            # Recommendation logic
            if frames == 8:
                rec = "Current (Baseline)"
            elif frames == 16 and time_s < 0.1:  # Less than 100ms
                rec = "Recommended" 
            elif frames > 16 and time_s > 0.2:  # More than 200ms
                rec = "Too Slow"
            else:
                rec = "Consider"
            
            print(f"{frames:<8} {time_s:<10.4f} {fps:<8.1f} {gpu_mb:<10.1f} {eff:<12.3f} {rec}")
        
        print("-" * 80)
        
        # Final recommendation
        best_config = min(configs, key=lambda x: x['timing']['mean_time_s'] if x['num_frames'] <= 16 else float('inf'))
        print(f"\nRECOMMENDATION: Use {best_config['num_frames']} frames")
        print(f"  - Execution time: {best_config['timing']['mean_time_s']:.4f}s")
        print(f"  - Effective FPS: {best_config['timing']['fps']:.1f}")
        print(f"  - GPU memory: {best_config['memory']['peak_gpu_mb']:.1f}MB")
        
        if best_config['num_frames'] > 8:
            speedup = configs[0]['timing']['mean_time_s'] / best_config['timing']['mean_time_s']
            print(f"  - Performance vs 8-frame: {speedup:.1f}x {'slower' if speedup < 1 else 'faster'}")


def main():
    """Main benchmark function"""
    benchmark = I3DPerformanceBenchmark()
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
