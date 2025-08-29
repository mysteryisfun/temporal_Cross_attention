"""
Dataset class for Something-Something-V2 video action recognition
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from ..utils.constants import FRAME_PER_VIDEO
from ..utils.config import (
    VIDEOS_DIR, LABELS_DIR, TRAIN_JSON, VALIDATION_JSON, TEST_JSON,
    NUM_CLASSES, VIDEO_EXT, FRAME_RATE
)


class SomethingSomethingV2Dataset(Dataset):
    """
    Dataset class for Something-Something-V2 video action recognition dataset.

    This dataset loads videos, samples frames uniformly, and applies augmentations
    while maintaining temporal consistency.
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = FRAME_PER_VIDEO,
        transform: Optional[transforms.Compose] = None,
        temporal_transform: Optional[transforms.Compose] = None,
        spatial_transform: Optional[transforms.Compose] = None,
        augment: bool = True,
        cache_labels: bool = True,
        max_cache_size: int = 1000  # Limit annotation cache size
    ):
        """
        Initialize the dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_frames: Number of frames to sample per video
            transform: Combined transform (if provided, overrides spatial/temporal)
            temporal_transform: Temporal augmentations
            spatial_transform: Spatial augmentations
            augment: Whether to apply augmentations
            cache_labels: Whether to cache label mappings
            max_cache_size: Maximum number of annotations to cache
        """
        self.split = split
        self.num_frames = num_frames
        self.augment = augment and split == "train"  # Only augment training data
        self.max_cache_size = max_cache_size

        # Set data paths
        self.videos_dir = VIDEOS_DIR
        self.labels_dir = LABELS_DIR

        # Load label mappings (lightweight)
        self.label_to_idx = self._load_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Load dataset annotations with memory limits
        self.annotations = self._load_annotations()

        # Initialize video cache (for frequently accessed videos)
        self.video_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.temporal_transform = temporal_transform or self._get_temporal_transform()
            self.spatial_transform = spatial_transform or self._get_spatial_transform()

        print(f"Loaded {len(self.annotations)} videos for {split} split")
        print(f"Number of classes: {len(self.label_to_idx)}")
        print(f"Memory-efficient mode: Videos loaded on-demand only")

    def _load_label_mapping(self) -> Dict[str, int]:
        """Load label to index mapping from labels.json"""
        labels_file = self.labels_dir / "labels.json"

        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with open(labels_file, 'r') as f:
            label_mapping = json.load(f)

        # Convert string values to integers
        return {k: int(v) for k, v in label_mapping.items()}

    def _template_to_label(self, template: str) -> str:
        """Convert template format to label format"""
        # More sophisticated template to label conversion
        
        # Handle specific template patterns
        template_fixes = {
            # Common patterns that need fixing
            '[Something]': 'Something',
            '[something]': 'something', 
            '[part]': 'part',
            '[somewhere]': 'somewhere',
            '[one of many similar things on the table]': 'something',
            '[something in it]': 'something in it'
        }
        
        # Apply fixes
        label = template
        for pattern, replacement in template_fixes.items():
            label = label.replace(pattern, replacement)
            
        return label

    def _load_annotations(self) -> List[Dict]:
        """Load annotations for the specified split"""
        if self.split == "train":
            annotations_file = TRAIN_JSON
        elif self.split == "validation":
            annotations_file = VALIDATION_JSON
        elif self.split == "test":
            annotations_file = TEST_JSON
        else:
            raise ValueError(f"Invalid split: {self.split}")

        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Filter out videos that don't exist
        valid_annotations = []
        for ann in annotations:
            video_path = self.videos_dir / f"{ann['id']}{VIDEO_EXT}"
            if video_path.exists():
                valid_annotations.append(ann)
            else:
                print(f"Warning: Video {ann['id']} not found, skipping")

        return valid_annotations

    def _get_temporal_transform(self) -> transforms.Compose:
        """Get temporal transformations"""
        temporal_transforms = []

        if self.augment:
            # Random temporal cropping/jittering
            temporal_transforms.append(RandomTemporalCrop(size=self.num_frames))
            # Note: TemporalJitter removed as it was causing issues

        # Uniform frame sampling (always applied)
        temporal_transforms.append(UniformTemporalSubsample(num_frames=self.num_frames))

        # Return list of transforms (not torchvision Compose)
        return temporal_transforms

    def _get_spatial_transform(self) -> transforms.Compose:
        """Get spatial transformations"""
        spatial_transforms = []

        # Resize to standard size
        spatial_transforms.append(transforms.Resize((224, 224)))

        if self.augment:
            # Random cropping
            spatial_transforms.append(transforms.RandomCrop(224, padding=4))
            # Random horizontal flip
            spatial_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            # Color jittering
            spatial_transforms.append(transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ))

        # Center crop for validation/test
        if not self.augment:
            spatial_transforms.append(transforms.CenterCrop(224))

        # Convert to tensor and normalize
        spatial_transforms.append(transforms.ToTensor())
        spatial_transforms.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ))

        return transforms.Compose(spatial_transforms)

    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        """Load and decode video frames with optional caching"""
        video_id = video_path.stem  # Extract video ID from path

        # Check cache first
        if video_id in self.video_cache:
            self.cache_hits += 1
            return self.video_cache[video_id].copy()
        else:
            self.cache_misses += 1

        # Load video from disk
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        frames_array = np.array(frames)

        # Cache the video if cache is not full
        if len(self.video_cache) < self.max_cache_size:
            self.video_cache[video_id] = frames_array.copy()

        return frames_array

    def clear_cache(self):
        """Clear video cache to free memory"""
        cache_size = len(self.video_cache)
        self.video_cache.clear()
        print(f"Cleared video cache ({cache_size} videos)")

    def get_cache_stats(self):
        """Get cache hit/miss statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.video_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def _sample_frames(self, frames: np.ndarray, indices: List[int]) -> np.ndarray:
        """Sample frames at given indices"""
        sampled_frames = []
        for idx in indices:
            if idx < len(frames):
                sampled_frames.append(frames[idx])
            else:
                # If index is out of bounds, use the last frame
                sampled_frames.append(frames[-1])

        return np.array(sampled_frames)

    def __len__(self) -> int:
        """Return the number of videos in the dataset"""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a video sample.

        Returns:
            Tuple of (video_frames, label, video_id)
        """
        # Get annotation
        ann = self.annotations[idx]
        video_id = ann['id']
        template = ann['template']
        placeholders = ann['placeholders']

        # Get label index
        template = ann['template']
        label_text = self._template_to_label(template)

        # Map to class index
        if label_text not in self.label_to_idx:
            # Fallback: use first available label
            label_idx = 0
            print(f"Warning: Label '{label_text}' not found in mapping, using label 0")
            print(f"Template was: '{template}'")
        else:
            label_idx = self.label_to_idx[label_text]

        # Load video
        video_path = self.videos_dir / f"{video_id}{VIDEO_EXT}"
        frames = self._load_video_frames(video_path)

        # Apply temporal transform to get frame indices
        if hasattr(self, 'temporal_transform') and self.temporal_transform is not None:
            # Apply list of temporal transforms
            current_frames = frames
            frame_indices = list(range(len(frames)))

            for transform in self.temporal_transform:
                result = transform(current_frames)
                if isinstance(result, tuple):
                    current_frames, frame_indices = result
                else:
                    current_frames = result

            frames = current_frames
        else:
            # Default uniform sampling
            frame_indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int).tolist()
            frames = self._sample_frames(frames, frame_indices)

        # Apply spatial transforms to each frame
        if hasattr(self, 'spatial_transform') and self.spatial_transform is not None:
            transformed_frames = []
            for frame in frames:
                # Convert to PIL Image for torchvision transforms
                pil_frame = F.to_pil_image(frame)
                transformed_frame = self.spatial_transform(pil_frame)
                transformed_frames.append(transformed_frame)

            frames = torch.stack(transformed_frames)

        # If using combined transform
        if hasattr(self, 'transform') and self.transform is not None:
            frames = self.transform(frames)

        return frames, label_idx, video_id


class UniformTemporalSubsample:
    """Uniformly subsample frames from a video"""

    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def __call__(self, frames: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Subsample frames uniformly.

        Args:
            frames: Video frames array of shape (T, H, W, C)

        Returns:
            Tuple of (sampled_frames, frame_indices)
        """
        total_frames = len(frames)

        if total_frames <= self.num_frames:
            # If we have fewer frames than needed, repeat the last frame
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        sampled_frames = frames[indices]
        return sampled_frames, indices


class RandomTemporalCrop:
    """Randomly crop a segment of frames from the video"""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, frames: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Randomly crop temporal segment.

        Args:
            frames: Video frames array

        Returns:
            Tuple of (cropped_frames, frame_indices)
        """
        total_frames = len(frames)

        if total_frames <= self.size:
            return frames, list(range(total_frames))

        # Random start index
        max_start = total_frames - self.size
        start_idx = random.randint(0, max_start)

        cropped_frames = frames[start_idx:start_idx + self.size]
        indices = list(range(start_idx, start_idx + self.size))

        return cropped_frames, indices


class TemporalJitter:
    """Add temporal jittering to frame sampling"""

    def __init__(self, jitter_range: float = 0.1):
        self.jitter_range = jitter_range

    def __call__(self, frames: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Add temporal jitter to frames.

        Args:
            frames: Video frames array

        Returns:
            Tuple of (jittered_frames, frame_indices)
        """
        # For now, just return the original frames
        # This can be enhanced with more sophisticated temporal jittering
        return frames, list(range(len(frames)))
