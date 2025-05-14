import numpy as np
from annotation_parser import load_annotations, parse_annotations

def compute_class_distribution(annotations):
    """
    Compute the distribution of classes in the dataset.

    Args:
        annotations (list): List of structured annotation data.

    Returns:
        dict: Class distribution as a dictionary.
    """
    class_counts = {}
    for annotation in annotations:
        for label in annotation['labels']:
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def compute_video_lengths(annotations):
    """
    Compute the lengths of videos in the dataset.

    Args:
        annotations (list): List of structured annotation data.

    Returns:
        list: List of video lengths.
    """
    return [annotation['duration'] for annotation in annotations]

def collect_statistics(annotation_file):
    """
    Collect dataset statistics including class distribution and video lengths.

    Args:
        annotation_file (str): Path to the annotation pickle file.

    Returns:
        dict: Dataset statistics.
    """
    annotations = load_annotations(annotation_file)
    if not annotations:
        return {}

    structured_annotations = parse_annotations(annotations)
    class_distribution = compute_class_distribution(structured_annotations)
    video_lengths = compute_video_lengths(structured_annotations)

    return {
        'class_distribution': class_distribution,
        'video_lengths': {
            'mean': np.mean(video_lengths),
            'std': np.std(video_lengths),
            'min': np.min(video_lengths),
            'max': np.max(video_lengths)
        }
    }

if __name__ == "__main__":
    annotation_file = "c:/Users/ujwal/OneDrive/Documents/GitHub/Cross_Attention_CNN_Research_Execution/data/raw/train-annotation/annotation_training.pkl"
    stats = collect_statistics(annotation_file)
    if stats:
        print("Class Distribution:")
        for cls, count in stats['class_distribution'].items():
            print(f"  {cls}: {count}")

        print("\nVideo Lengths:")
        for key, value in stats['video_lengths'].items():
            print(f"  {key}: {value:.2f}")
