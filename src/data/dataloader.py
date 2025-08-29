"""
Memory-efficient DataLoader for Something-Something-V2 dataset
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable
import multiprocessing as mp

from .dataset import SomethingSomethingV2Dataset


class MemoryEfficientDataLoader:
    """
    Memory-efficient DataLoader optimized for video datasets.

    Features:
    - Sequential loading (no pre-loading of videos)
    - Optimized batch collation
    - Memory monitoring
    - Configurable prefetching
    """

    def __init__(
        self,
        dataset: SomethingSomethingV2Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        """
        Initialize memory-efficient DataLoader.

        Args:
            dataset: SomethingSomethingV2Dataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Custom collation function
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Use default collate function if none provided
        if collate_fn is None:
            collate_fn = self._default_collate_fn

        # Optimize num_workers based on CPU cores
        if num_workers == 0:
            num_workers = 0  # Explicit sequential processing
        elif num_workers < 0:
            num_workers = max(1, mp.cpu_count() // 2)  # Use half of available cores

        # Create DataLoader with memory-efficient settings
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

        print(f"Created DataLoader with {num_workers} workers, batch_size={batch_size}")
        print(f"Dataset size: {len(dataset)}, Expected batches: {len(dataset) // batch_size}")

    def _default_collate_fn(self, batch):
        """
        Memory-efficient collation function.

        Args:
            batch: List of (frames, label, video_id) tuples

        Returns:
            Tuple of (batched_frames, batched_labels, video_ids)
        """
        if not batch:
            return torch.empty(0), torch.empty(0), []

        # Separate components
        frames_list = []
        labels_list = []
        video_ids_list = []

        for item in batch:
            if len(item) == 3:
                frames, label, video_id = item
                frames_list.append(frames)
                labels_list.append(label)
                video_ids_list.append(video_id)

        # Batch frames efficiently
        if frames_list:
            batched_frames = torch.stack(frames_list, dim=0)
        else:
            batched_frames = torch.empty(0)

        # Batch labels
        if labels_list:
            batched_labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            batched_labels = torch.empty(0, dtype=torch.long)

        return batched_frames, batched_labels, video_ids_list

    def __iter__(self):
        """Iterator for the DataLoader"""
        return iter(self.dataloader)

    def __len__(self):
        """Return number of batches"""
        return len(self.dataloader)

    @property
    def total_samples(self):
        """Return total number of samples in dataset"""
        return len(self.dataset)


def create_efficient_data_loader(
    split: str = "train",
    batch_size: int = 8,
    num_frames: int = 8,
    augment: bool = True,
    num_workers: int = 4,
    **kwargs
) -> MemoryEfficientDataLoader:
    """
    Factory function to create memory-efficient data loader.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        num_frames: Number of frames per video
        augment: Whether to apply augmentations
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader

    Returns:
        MemoryEfficientDataLoader instance
    """
    # Create dataset
    dataset = SomethingSomethingV2Dataset(
        split=split,
        num_frames=num_frames,
        augment=augment
    )

    # Create data loader
    data_loader = MemoryEfficientDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )

    return data_loader


# Memory monitoring utilities
def get_memory_usage():
    """Get current memory usage statistics"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }


def log_memory_usage(stage: str = ""):
    """Log current memory usage"""
    mem_info = get_memory_usage()
    print(f"Memory Usage {stage}: RSS={mem_info['rss']:.1f}MB, "
          f"VMS={mem_info['vms']:.1f}MB, Percent={mem_info['percent']:.1f}%")
