"""
Memory-Efficient DataLoader for Something-Something-V2
====================================================

Optimized for RTX 3050 (4GB VRAM) + 16GB RAM:
- Sequential video loading (no batch pre-loading)
- Dynamic batch sizing based on available memory
- Feature caching system
- Memory monitoring and cleanup
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
import os
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Union
import logging
from PIL import Image
import pickle
import hashlib

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, gpu_limit_mb: int = 3500, ram_limit_mb: int = 12000):
        """
        Initialize memory monitor
        
        Args:
            gpu_limit_mb: GPU memory limit in MB (3.5GB for RTX 3050)
            ram_limit_mb: RAM memory limit in MB (12GB of 16GB)
        """
        self.gpu_limit_mb = gpu_limit_mb
        self.ram_limit_mb = ram_limit_mb
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        # RAM usage
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU usage
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_mb = gpu_max_mb = 0
        
        return {
            'ram_mb': ram_mb,
            'gpu_mb': gpu_mb,
            'gpu_max_mb': gpu_max_mb,
            'ram_percent': (ram_mb / self.ram_limit_mb) * 100,
            'gpu_percent': (gpu_mb / self.gpu_limit_mb) * 100
        }
    
    def can_process_batch(self, estimated_gpu_mb: int = 500) -> bool:
        """Check if we can process another batch"""
        usage = self.get_memory_usage()
        return (usage['gpu_mb'] + estimated_gpu_mb) < self.gpu_limit_mb
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VideoFeatureCache:
    """Efficient feature caching system"""
    
    def __init__(self, cache_dir: str = "data/feature_cache"):
        """
        Initialize feature cache
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate directories for different feature types
        self.semantic_cache = self.cache_dir / "semantic"
        self.motion_cache = self.cache_dir / "motion"
        self.semantic_cache.mkdir(exist_ok=True)
        self.motion_cache.mkdir(exist_ok=True)
        
        logger.info(f"Feature cache initialized at {self.cache_dir}")
    
    def get_cache_path(self, video_id: str, feature_type: str) -> Path:
        """Get cache file path for video and feature type"""
        if feature_type == "semantic":
            return self.semantic_cache / f"{video_id}_semantic.npy"
        elif feature_type == "motion":
            return self.motion_cache / f"{video_id}_motion.npy"
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def is_cached(self, video_id: str, feature_type: str) -> bool:
        """Check if features are already cached"""
        cache_path = self.get_cache_path(video_id, feature_type)
        return cache_path.exists()
    
    def save_features(self, video_id: str, feature_type: str, features: np.ndarray):
        """Save features to cache"""
        cache_path = self.get_cache_path(video_id, feature_type)
        np.save(cache_path, features)
        logger.debug(f"Cached {feature_type} features for {video_id}")
    
    def load_features(self, video_id: str, feature_type: str) -> Optional[np.ndarray]:
        """Load features from cache"""
        cache_path = self.get_cache_path(video_id, feature_type)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        semantic_count = len(list(self.semantic_cache.glob("*.npy")))
        motion_count = len(list(self.motion_cache.glob("*.npy")))
        
        return {
            'semantic_cached': semantic_count,
            'motion_cached': motion_count,
            'total_cached': semantic_count + motion_count
        }


class MemoryEfficientVideoDataset(Dataset):
    """
    Memory-efficient dataset for Something-Something-V2
    
    Features:
    - Sequential video loading (no pre-loading)
    - Feature caching system
    - Dynamic batch sizing
    - Memory monitoring
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: str = "data/feature_cache",
        max_videos: Optional[int] = None,
        precompute_features: bool = False
    ):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to Something-Something-V2 dataset
            split: Dataset split ('train', 'val', 'test')
            cache_dir: Directory for feature caching
            max_videos: Maximum number of videos to use (for testing)
            precompute_features: Whether to precompute all features
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_videos = max_videos
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.feature_cache = VideoFeatureCache(cache_dir)
        
        # Load dataset metadata
        self.videos, self.labels, self.label_map = self._load_dataset_info()
        
        logger.info(f"Loaded {len(self.videos)} videos for {split} split")
        
        # Feature extractors (lazy initialization)
        self._semantic_extractor = None
        self._motion_extractor = None
        
        if precompute_features:
            self.precompute_all_features()
    
    def _load_dataset_info(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """Load dataset information from JSON files"""
        # Load labels
        if self.split == "train":
            label_file = self.data_dir / "labels" / "train.json"
        elif self.split == "val":
            label_file = self.data_dir / "labels" / "validation.json"
        elif self.split == "test":
            label_file = self.data_dir / "labels" / "test.json"
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Load category labels
        category_file = self.data_dir / "labels" / "labels.json"
        
        videos = []
        labels = []
        label_map = {}
        
        # Load categories
        if category_file.exists():
            with open(category_file, 'r') as f:
                categories = json.load(f)
                # Handle both list and dict formats
                if isinstance(categories, dict):
                    label_map = categories
                else:
                    label_map = {cat: idx for idx, cat in enumerate(categories)}
        
        # Load video metadata
        if label_file.exists():
            with open(label_file, 'r') as f:
                data = json.load(f)
                
                for item in data:
                    if self.max_videos and len(videos) >= self.max_videos:
                        break
                    
                    # Handle different JSON formats
                    if isinstance(item, dict):
                        video_id = str(item.get('id', item.get('video_id', '')))
                        videos.append(video_id)
                        
                        # Get label
                        if 'template' in item:
                            label_text = item['template'].replace('[', '').replace(']', '')
                        elif 'label' in item:
                            label_text = item['label']
                        elif 'category' in item:
                            label_text = item['category']
                        else:
                            label_text = None
                        
                        if label_text and label_text in label_map:
                            labels.append(label_map[label_text])
                        else:
                            labels.append(0)  # Unknown label
                    else:
                        # Handle simple list format
                        video_id = str(item)
                        videos.append(video_id)
                        labels.append(-1)  # No label info
        else:
            logger.warning(f"Label file not found: {label_file}")
            # Fallback: scan video directory
            video_dir = self.data_dir / "videos"
            if video_dir.exists():
                video_files = list(video_dir.glob("*.webm"))
                for video_file in video_files:
                    if self.max_videos and len(videos) >= self.max_videos:
                        break
                    video_id = video_file.stem
                    videos.append(video_id)
                    labels.append(-1)  # No label
        
        logger.info(f"Loaded {len(videos)} videos, {len(label_map)} categories")
        return videos, labels, label_map
    
    def _get_semantic_extractor(self):
        """Lazy initialization of semantic extractor"""
        if self._semantic_extractor is None:
            from src.models.semantic_extractor.dinov3_extractor import DINOv3SemanticExtractor
            self._semantic_extractor = DINOv3SemanticExtractor(device='cuda')
        return self._semantic_extractor
    
    def _get_motion_extractor(self):
        """Lazy initialization of motion extractor"""
        if self._motion_extractor is None:
            from src.models.motion_extractor.i3d_extractor import I3DMotionExtractor
            self._motion_extractor = I3DMotionExtractor(device='cuda')
        return self._motion_extractor
    
    def _load_video(self, video_id: str) -> Optional[np.ndarray]:
        """Load video frames"""
        video_path = self.data_dir / "videos" / f"{video_id}.webm"
        
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return None
        
        # Load video with memory efficiency
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return None
            
            # Sample 16 frames evenly
            frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
                else:
                    break
            
            if len(frames) == 16:
                return np.array(frames)
            else:
                logger.warning(f"Could not load 16 frames from {video_id}")
                return None
                
        finally:
            cap.release()
    
    def extract_features(self, video_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract or load cached features for a video
        
        Returns:
            Tuple of (semantic_features, motion_features)
        """
        # Check cache first
        semantic_features = self.feature_cache.load_features(video_id, "semantic")
        motion_features = self.feature_cache.load_features(video_id, "motion")
        
        # If both cached, return immediately
        if semantic_features is not None and motion_features is not None:
            return semantic_features, motion_features
        
        # Load video if needed
        video_frames = self._load_video(video_id)
        if video_frames is None:
            return None, None
        
        # Extract semantic features if not cached
        if semantic_features is None:
            try:
                semantic_extractor = self._get_semantic_extractor()
                # Use middle frame for semantic features
                middle_frame = video_frames[8]  # Frame 8 of 16
                
                # Convert numpy to tensor for DINOv3 (expects processed tensor)
                if isinstance(middle_frame, np.ndarray):
                    # Convert to tensor and apply preprocessing
                    frame_tensor = torch.tensor(middle_frame).permute(2, 0, 1).float() / 255.0  # (C, H, W)
                    frame_tensor = frame_tensor.unsqueeze(0).to('cuda')  # (1, C, H, W)
                    
                semantic_features = semantic_extractor.extract_features(frame_tensor)
                
                if semantic_features is not None:
                    if torch.is_tensor(semantic_features):
                        semantic_features = semantic_features.cpu().numpy()
                    self.feature_cache.save_features(video_id, "semantic", semantic_features)
                
            except Exception as e:
                logger.error(f"Error extracting semantic features for {video_id}: {e}")
                semantic_features = None
        
        # Extract motion features if not cached
        if motion_features is None:
            try:
                motion_extractor = self._get_motion_extractor()
                # Convert frames to tensor format: (1, C, T, H, W)
                video_tensor = torch.tensor(video_frames).permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, 3, 16, 224, 224)
                video_tensor = video_tensor.to('cuda')  # Move to GPU
                motion_features = motion_extractor.extract_features(video_tensor)
                
                if motion_features is not None:
                    if torch.is_tensor(motion_features):
                        motion_features = motion_features.cpu().numpy()
                    self.feature_cache.save_features(video_id, "motion", motion_features)
                
            except Exception as e:
                logger.error(f"Error extracting motion features for {video_id}: {e}")
                motion_features = None
        
        # Cleanup video data from memory
        del video_frames
        self.memory_monitor.cleanup_memory()
        
        return semantic_features, motion_features
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get item from dataset
        
        Returns:
            Dict containing semantic_features, motion_features, label, video_id
        """
        video_id = self.videos[idx]
        label = self.labels[idx] if idx < len(self.labels) else -1
        
        # Extract features
        semantic_features, motion_features = self.extract_features(video_id)
        
        # Handle missing features
        if semantic_features is None:
            semantic_features = np.zeros((768,), dtype=np.float32)
        if motion_features is None:
            motion_features = np.zeros((512,), dtype=np.float32)
        
        # Convert to tensors
        semantic_tensor = torch.from_numpy(semantic_features).float()
        motion_tensor = torch.from_numpy(motion_features).float()
        
        return {
            'semantic_features': semantic_tensor,
            'motion_features': motion_tensor,
            'label': label,
            'video_id': video_id
        }
    
    def precompute_all_features(self):
        """Precompute features for entire dataset"""
        logger.info(f"Precomputing features for {len(self.videos)} videos...")
        
        for i, video_id in enumerate(self.videos):
            if i % 100 == 0:
                cache_stats = self.feature_cache.get_cache_stats()
                memory_usage = self.memory_monitor.get_memory_usage()
                logger.info(f"Processed {i}/{len(self.videos)} videos | "
                           f"Cached: {cache_stats['total_cached']} | "
                           f"GPU: {memory_usage['gpu_mb']:.1f}MB | "
                           f"RAM: {memory_usage['ram_mb']:.1f}MB")
            
            # Skip if already cached
            if (self.feature_cache.is_cached(video_id, "semantic") and 
                self.feature_cache.is_cached(video_id, "motion")):
                continue
            
            # Extract features
            self.extract_features(video_id)
        
        logger.info("Feature precomputation complete!")


def create_memory_efficient_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    cache_dir: str = "data/feature_cache",
    max_videos: Optional[int] = None
) -> DataLoader:
    """
    Create memory-efficient DataLoader
    
    Args:
        data_dir: Path to Something-Something-V2 dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size (will be adjusted based on memory)
        num_workers: Number of worker processes
        cache_dir: Feature cache directory
        max_videos: Maximum videos to use (for testing)
    
    Returns:
        DataLoader instance
    """
    dataset = MemoryEfficientVideoDataset(
        data_dir=data_dir,
        split=split,
        cache_dir=cache_dir,
        max_videos=max_videos
    )
    
    # Adjust batch size for RTX 3050 (4GB VRAM)
    memory_monitor = MemoryMonitor()
    usage = memory_monitor.get_memory_usage()
    
    # Conservative batch sizing for 4GB VRAM
    if usage['gpu_percent'] > 70:
        batch_size = min(batch_size, 4)  # Very conservative
    elif usage['gpu_percent'] > 50:
        batch_size = min(batch_size, 6)  # Conservative
    
    logger.info(f"Using batch size: {batch_size} (GPU usage: {usage['gpu_percent']:.1f}%)")
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


# Factory functions for easy access
def get_train_dataloader(data_dir: str, **kwargs) -> DataLoader:
    """Get training DataLoader"""
    return create_memory_efficient_dataloader(data_dir, split="train", **kwargs)

def get_val_dataloader(data_dir: str, **kwargs) -> DataLoader:
    """Get validation DataLoader"""
    return create_memory_efficient_dataloader(data_dir, split="val", **kwargs)

def get_test_dataloader(data_dir: str, **kwargs) -> DataLoader:
    """Get test DataLoader"""
    return create_memory_efficient_dataloader(data_dir, split="test", **kwargs)
