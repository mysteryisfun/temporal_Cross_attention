import os
import pickle

def verify_dataset(annotation_path, dataset_dir):
    """
    Verifies the integrity of the dataset by checking if all annotated files exist in the dataset directory
    and if there are any unannotated files in the dataset directory.

    Parameters:
        annotation_path (str): Path to the annotation file (e.g., annotation_training.pkl).
        dataset_dir (str): Path to the dataset directory containing video files.

    Returns:
        dict: A dictionary containing verification results.
    """
    # Load annotations
    with open(annotation_path, 'rb') as f:
        annotations = pickle.load(f)

    annotated_files = set()
    for trait, videos in annotations.items():
        if trait != 'interview':  # Skip non-trait keys
            annotated_files.update(videos.keys())

    # Get all video files in the dataset directory
    dataset_files = set()
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.mp4'):
                dataset_files.add(file)

    # Check for missing and extra files
    missing_files = annotated_files - dataset_files
    extra_files = dataset_files - annotated_files

    return {
        'missing_files': missing_files,
        'extra_files': extra_files,
        'total_annotated_files': len(annotated_files),
        'total_dataset_files': len(dataset_files)
    }

if __name__ == "__main__":
    annotation_path = "data/raw/train-annotation/annotation_training.pkl"
    dataset_dir = "data/raw/train-1/"

    results = verify_dataset(annotation_path, dataset_dir)

    print("Dataset Verification Results:")
    print(f"Total Annotated Files: {results['total_annotated_files']}")
    print(f"Total Dataset Files: {results['total_dataset_files']}")
    print(f"Missing Files: {results['missing_files']}")
    print(f"Extra Files: {results['extra_files']}")
