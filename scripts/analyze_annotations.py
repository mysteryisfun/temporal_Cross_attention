"""
Script to analyze the structure of the annotation pickle file.
"""

import pickle
import sys
from pathlib import Path

annotation_file = Path("data/raw/train-annotation/annotation_training.pkl")

try:
    with open(annotation_file, 'rb') as f:
        # Use encoding='latin1' for potential compatibility with Python 2 pickles
        annotation_data = pickle.load(f, encoding='latin1')
    
    # Check the type and top-level structure
    print(f"Type of annotation data: {type(annotation_data)}")
    
    # If it's a dictionary, examine keys
    if isinstance(annotation_data, dict):
        print(f"Top level keys: {list(annotation_data.keys())}")
        
        # Examine the structure of the first item
        first_key = list(annotation_data.keys())[0]
        print(f"\nExample for key '{first_key}':")
        print(f"Type: {type(annotation_data[first_key])}")
        print(f"Content: {annotation_data[first_key]}")
    
    # If it's a list, examine the structure of the first few items
    elif isinstance(annotation_data, list):
        print(f"Number of items: {len(annotation_data)}")
        
        if len(annotation_data) > 0:
            print(f"\nExample for first item:")
            print(f"Type: {type(annotation_data[0])}")
            print(f"Content: {annotation_data[0]}")
    
    # Count the number of annotations that match our train-1 folder
    train1_files = set()
    for subdir in Path("data/raw/train-1").glob("training*"):
        for video_file in subdir.glob("*.mp4"):
            train1_files.add(video_file.name)
    
    print(f"\nTotal number of video files in train-1: {len(train1_files)}")
    
    # Check how many of these files have annotations
    if isinstance(annotation_data, dict):
        annotated_files = set(annotation_data.keys())
        matching_files = train1_files.intersection(annotated_files)
        print(f"Number of train-1 files with annotations: {len(matching_files)}")
    
except Exception as e:
    print(f"Error analyzing annotation file: {str(e)}")
