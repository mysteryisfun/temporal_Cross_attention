import pickle
import os

file_path = 'data/raw/train-annotation/annotation_training.pkl'
video_dir = 'data/raw/train-1/training80_01'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        print("Top-level keys in the annotation file:", list(data.keys()))

        # Extract filenames from annotations
        annotated_files = set()
        for trait in data:
            annotated_files.update(data[trait].keys())

        # List video files in the directory
        video_files = set(os.listdir(video_dir))

        # Check for discrepancies
        annotated_but_missing = annotated_files - video_files
        present_but_not_annotated = video_files - annotated_files

        print("Annotated but missing in directory:", annotated_but_missing)
        print("Present in directory but not annotated:", present_but_not_annotated)

except Exception as e:
    print("Error loading the annotation file:", e)
