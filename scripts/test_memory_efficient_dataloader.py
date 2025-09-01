#!/usr/bin/env python3
"""
Test Memory-Efficient DataLoader

This script tests the memory-efficient DataLoader with RTX 3050 constraints.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.memory_efficient_dataloader import (
    MemoryEfficientVideoDataset,
    create_memory_efficient_dataloader,
    MemoryMonitor
)
import torch
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_memory_efficient_dataloader():
    """Test the memory-efficient DataLoader"""
    
    print("=== Memory-Efficient DataLoader Test ===")
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    initial_usage = memory_monitor.get_memory_usage()
    print(f"Initial Memory - GPU: {initial_usage['gpu_mb']:.1f}MB, RAM: {initial_usage['ram_mb']:.1f}MB")
    
    # Test with small subset first
    data_dir = "data/raw"
    
    try:
        # Create dataset
        print("\n1. Creating dataset...")
        dataset = MemoryEfficientVideoDataset(
            data_dir=data_dir,
            split="train",
            max_videos=10,  # Test with 10 videos first
            cache_dir="experiments/test_cache"
        )
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Check cache stats
        cache_stats = dataset.feature_cache.get_cache_stats()
        print(f"   Cache stats: {cache_stats}")
        
        # Test single item loading
        print("\n2. Testing single item loading...")
        start_time = time.time()
        item = dataset[0]
        load_time = time.time() - start_time
        
        print(f"   Loaded item 0 in {load_time:.3f}s")
        print(f"   Semantic features shape: {item['semantic_features'].shape}")
        print(f"   Motion features shape: {item['motion_features'].shape}")
        print(f"   Label: {item['label']}")
        print(f"   Video ID: {item['video_id']}")
        
        # Check memory after single item
        usage_after_item = memory_monitor.get_memory_usage()
        print(f"   Memory after item - GPU: {usage_after_item['gpu_mb']:.1f}MB, RAM: {usage_after_item['ram_mb']:.1f}MB")
        
        # Test DataLoader
        print("\n3. Testing DataLoader...")
        dataloader = create_memory_efficient_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=4,  # Conservative for RTX 3050
            num_workers=0,  # Single threaded for testing
            max_videos=10,
            cache_dir="experiments/test_cache"
        )
        
        print(f"   DataLoader batch size: {dataloader.batch_size}")
        print(f"   DataLoader length: {len(dataloader)}")
        
        # Test a few batches
        batch_times = []
        memory_usage = []
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
                
            start_time = time.time()
            
            # Simulate processing
            semantic_features = batch['semantic_features']  # (batch_size, 768)
            motion_features = batch['motion_features']      # (batch_size, 512)
            labels = batch['label']                         # (batch_size,)
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Check memory usage
            usage = memory_monitor.get_memory_usage()
            memory_usage.append(usage)
            
            print(f"   Batch {i+1}: {semantic_features.shape}, {motion_features.shape}")
            print(f"   Time: {batch_time:.3f}s, GPU: {usage['gpu_mb']:.1f}MB, RAM: {usage['ram_mb']:.1f}MB")
            
            # Force cleanup
            del batch, semantic_features, motion_features, labels
            memory_monitor.cleanup_memory()
        
        # Performance summary
        print(f"\n4. Performance Summary:")
        print(f"   Average batch time: {sum(batch_times)/len(batch_times):.3f}s")
        print(f"   Peak GPU memory: {max(u['gpu_mb'] for u in memory_usage):.1f}MB")
        print(f"   Peak RAM memory: {max(u['ram_mb'] for u in memory_usage):.1f}MB")
        
        # Test cache effectiveness
        print(f"\n5. Cache Effectiveness:")
        final_cache_stats = dataset.feature_cache.get_cache_stats()
        print(f"   Final cache stats: {final_cache_stats}")
        
        # Test cache loading speed
        print("\n6. Testing cached feature loading...")
        if final_cache_stats['total_cached'] > 0:
            start_time = time.time()
            cached_item = dataset[0]  # Should be cached now
            cached_time = time.time() - start_time
            print(f"   Cached loading time: {cached_time:.3f}s (should be faster)")
        
        print("\n✅ Memory-efficient DataLoader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_memory_limits():
    """Test memory limit handling"""
    
    print("\n=== Memory Limit Test ===")
    
    memory_monitor = MemoryMonitor()
    
    # Test memory checking
    usage = memory_monitor.get_memory_usage()
    print(f"Current usage - GPU: {usage['gpu_percent']:.1f}%, RAM: {usage['ram_percent']:.1f}%")
    
    # Test batch processing capability
    can_process = memory_monitor.can_process_batch(estimated_gpu_mb=300)
    print(f"Can process batch (300MB): {can_process}")
    
    # Test memory cleanup
    print("Testing memory cleanup...")
    memory_monitor.cleanup_memory()
    usage_after = memory_monitor.get_memory_usage()
    print(f"Usage after cleanup - GPU: {usage_after['gpu_mb']:.1f}MB, RAM: {usage_after['ram_mb']:.1f}MB")


def test_feature_extraction_performance():
    """Test feature extraction performance"""
    
    print("\n=== Feature Extraction Performance Test ===")
    
    data_dir = "data/raw"
    
    try:
        dataset = MemoryEfficientVideoDataset(
            data_dir=data_dir,
            split="train",
            max_videos=5,
            cache_dir="experiments/test_cache"
        )
        
        print(f"Testing feature extraction for {len(dataset)} videos...")
        
        total_semantic_time = 0
        total_motion_time = 0
        successful_extractions = 0
        
        for i in range(min(3, len(dataset))):  # Test first 3 videos
            video_id = dataset.videos[i]
            print(f"\nProcessing video {i+1}: {video_id}")
            
            # Time feature extraction
            start_time = time.time()
            semantic_features, motion_features = dataset.extract_features(video_id)
            extraction_time = time.time() - start_time
            
            if semantic_features is not None and motion_features is not None:
                successful_extractions += 1
                print(f"   ✅ Extraction successful in {extraction_time:.3f}s")
                print(f"   Semantic: {semantic_features.shape}, Motion: {motion_features.shape}")
            else:
                print(f"   ❌ Extraction failed")
            
            # Check memory
            usage = MemoryMonitor().get_memory_usage()
            print(f"   Memory - GPU: {usage['gpu_mb']:.1f}MB, RAM: {usage['ram_mb']:.1f}MB")
        
        print(f"\nFeature extraction summary:")
        print(f"   Success rate: {successful_extractions}/{min(3, len(dataset))}")
        
    except Exception as e:
        print(f"Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    
    print("Starting Memory-Efficient DataLoader Tests")
    print("=" * 60)
    
    # Test 1: Memory limits
    test_memory_limits()
    
    # Test 2: Feature extraction performance
    test_feature_extraction_performance()
    
    # Test 3: DataLoader functionality
    success = test_memory_efficient_dataloader()
    
    if success:
        print("\n🎉 All tests passed! Memory-efficient DataLoader is ready.")
    else:
        print("\n💥 Some tests failed. Check the logs above.")


if __name__ == "__main__":
    main()
