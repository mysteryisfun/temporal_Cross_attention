import os
import pickle
import random
from annotation_parser import load_annotations, parse_annotations

def split_dataset(annotations, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        annotations (list): List of structured annotation data.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.

    Returns:
        dict: Split datasets with keys 'train', 'val', and 'test'.
    """
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.")

    random.shuffle(annotations)
    total = len(annotations)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        'train': annotations[:train_end],
        'val': annotations[train_end:val_end],
        'test': annotations[val_end:]
    }

def save_splits(splits, output_dir):
    """
    Save the dataset splits to files.

    Args:
        splits (dict): Dataset splits with keys 'train', 'val', and 'test'.
        output_dir (str): Directory to save the split files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}_split.pkl")
        with open(output_file, 'wb') as file:
            pickle.dump(split_data, file)

if __name__ == "__main__":
    annotation_file = "c:/Users/ujwal/OneDrive/Documents/GitHub/Cross_Attention_CNN_Research_Execution/data/raw/train-annotation/annotation_training.pkl"
    output_dir = "c:/Users/ujwal/OneDrive/Documents/GitHub/Cross_Attention_CNN_Research_Execution/data/splits"

    annotations = load_annotations(annotation_file)
    if annotations:
        structured_annotations = parse_annotations(annotations)
        splits = split_dataset(structured_annotations)
        save_splits(splits, output_dir)
        print(f"Dataset splits saved to {output_dir}")
