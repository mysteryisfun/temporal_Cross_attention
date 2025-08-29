#!/usr/bin/env python3
"""
Merge Dynamic Feature Extraction Results

This script concatenates the feature arrays from Part 1 and Part 2 into a single dataset.
- Input: dynamic_features_part1.npy, dynamic_features_part2.npy
- Output: dynamic_features_merged.npy

Usage:
    python merge_dynamic_features.py

Requirements:
    - NumPy
    - Both input files must exist in results/features/

Author: GitHub Copilot
"""

import numpy as np
from pathlib import Path
import sys

def main():
    features_dir = Path("results/features")
    part1_path = features_dir / "dynamic_features_part1.npy"
    part2_path = features_dir / "dynamic_features_part2.npy"
    merged_path = features_dir / "dynamic_features_merged.npy"

    # Check if both files exist
    if not part1_path.exists() or not part2_path.exists():
        print(f"❌ One or both input files are missing:")
        print(f"   {part1_path}")
        print(f"   {part2_path}")
        sys.exit(1)

    print(f"🔄 Loading features from Part 1: {part1_path}")
    features1 = np.load(part1_path)
    print(f"   Shape: {features1.shape}")

    print(f"🔄 Loading features from Part 2: {part2_path}")
    features2 = np.load(part2_path)
    print(f"   Shape: {features2.shape}")

    print(f"🔗 Concatenating features...")
    merged_features = np.concatenate([features1, features2], axis=0)
    print(f"   Merged shape: {merged_features.shape}")

    print(f"💾 Saving merged features to: {merged_path}")
    np.save(merged_path, merged_features)
    print(f"✅ Merge complete!")

if __name__ == "__main__":
    main()
