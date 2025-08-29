"""
Test script for the Something-Something-V2 dataset
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset import SomethingSomethingV2Dataset
from src.utils.constants import FRAME_PER_VIDEO


def test_dataset():
    """Test the dataset loading and basic functionality"""

    print("Testing Something-Something-V2 Dataset...")
    print(f"Frames per video: {FRAME_PER_VIDEO}")

    # Test with small subset first
    try:
        # Create dataset
        dataset = SomethingSomethingV2Dataset(
            split="train",
            num_frames=FRAME_PER_VIDEO,
            augment=False  # Disable augmentations for testing
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(dataset.label_to_idx)}")

        # Test loading a few samples
        print("\nTesting sample loading...")
        for i in range(min(3, len(dataset))):
            frames, label, video_id = dataset[i]

            print(f"Sample {i}:")
            print(f"  Video ID: {video_id}")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Label index: {label}")
            print(f"  Expected frames: {FRAME_PER_VIDEO}")
            print(f"  Actual frames: {frames.shape[0]}")

            # Check tensor properties
            assert isinstance(frames, torch.Tensor), "Frames should be torch.Tensor"
            assert frames.shape[0] == FRAME_PER_VIDEO, f"Should have {FRAME_PER_VIDEO} frames"
            assert frames.shape[1] == 3, "Should have 3 color channels"
            assert frames.shape[2] == 224 and frames.shape[3] == 224, "Should be 224x224"

        print("\n✅ Dataset test passed!")

        # Test DataLoader
        print("\nTesting DataLoader...")

        def collate_fn(batch):
            """Custom collate function for the dataset"""
            frames = []
            labels = []
            video_ids = []

            for item in batch:
                frames.append(item[0])
                labels.append(item[1])
                video_ids.append(item[2])

            return torch.stack(frames), torch.tensor(labels, dtype=torch.long), video_ids

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 for debugging
            collate_fn=collate_fn
        )

        for batch_frames, batch_labels, batch_video_ids in dataloader:
            print(f"Batch frames shape: {batch_frames.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            print(f"Batch labels: {batch_labels}")
            print(f"Batch video IDs: {batch_video_ids}")
            break

        print("✅ DataLoader test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n🎉 All tests passed! Dataset is ready for training.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
