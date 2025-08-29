"""
Example usage of memory-efficient data loading for temporal cross-attention model
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset import SomethingSomethingV2Dataset
from src.data.dataloader import MemoryEfficientDataLoader
from src.utils.constants import FRAME_PER_VIDEO


def create_training_data_loader():
    """Create memory-efficient data loader for training"""

    # Create training dataset
    train_dataset = SomethingSomethingV2Dataset(
        split="train",
        num_frames=FRAME_PER_VIDEO,
        augment=True  # Enable augmentations for training
    )

    # Create memory-efficient data loader
    train_loader = MemoryEfficientDataLoader(
        dataset=train_dataset,
        batch_size=8,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,  # Use multiple workers for parallel loading
        pin_memory=True,  # Enable for faster GPU transfer
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader


def create_validation_data_loader():
    """Create data loader for validation (no augmentations)"""

    val_dataset = SomethingSomethingV2Dataset(
        split="validation",
        num_frames=FRAME_PER_VIDEO,
        augment=False  # No augmentations for validation
    )

    val_loader = MemoryEfficientDataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,  # No shuffling for validation
        num_workers=2,
        pin_memory=True
    )

    return val_loader


def training_loop_example():
    """Example of how to use the data loaders in training"""

    print("🚀 Starting Training Loop Example")
    print("=" * 40)

    # Create data loaders
    train_loader = create_training_data_loader()
    val_loader = create_validation_data_loader()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process a few batches as example
    print("\n📊 Processing sample batches...")

    for epoch in range(1):  # Just one epoch for demonstration
        print(f"\nEpoch {epoch + 1}")

        # Training phase
        train_loader.dataset.augment = True  # Ensure augmentations are enabled
        for batch_idx, (frames, labels, video_ids) in enumerate(train_loader):

            # Move to device
            frames = frames.to(device)
            labels = labels.to(device)

            print(f"  Batch {batch_idx + 1}:")
            print(f"    Frames shape: {frames.shape}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Video IDs: {video_ids[:2]}...")  # Show first 2

            # Here you would:
            # 1. Forward pass through your model
            # 2. Calculate loss
            # 3. Backward pass
            # 4. Optimizer step

            if batch_idx >= 2:  # Only process 3 batches for demo
                break

        # Validation phase
        val_loader.dataset.augment = False  # Disable augmentations for validation
        with torch.no_grad():
            for batch_idx, (frames, labels, video_ids) in enumerate(val_loader):
                frames = frames.to(device)
                labels = labels.to(device)

                print(f"  Val Batch {batch_idx + 1}: {frames.shape}")

                # Here you would:
                # 1. Forward pass through your model
                # 2. Calculate validation metrics

                if batch_idx >= 1:  # Only process 2 batches for demo
                    break

    print("\n✅ Training loop example completed!")
    print("\n💡 Key Points:")
    print("  • Videos are loaded on-demand (memory efficient)")
    print("  • Augmentations enabled for training, disabled for validation")
    print("  • DataLoader handles batching and device transfer automatically")
    print("  • Cache system reduces redundant video loading")


if __name__ == "__main__":
    training_loop_example()
