import pickle

def load_annotations(annotation_file):
    """
    Load and parse annotations from a pickle file.

    Args:
        annotation_file (str): Path to the annotation pickle file.

    Returns:
        dict: Parsed annotations.
    """
    try:
        with open(annotation_file, 'rb') as file:  # Ensure binary mode is used
            annotations = pickle.load(file, encoding='latin1')  # Specify encoding to handle non-ASCII characters
        return annotations
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None

def parse_annotations(annotations):
    """
    Parse the raw annotations into a structured format.

    Args:
        annotations (dict): Raw annotations loaded from the pickle file.

    Returns:
        list: A list of structured annotation data.
    """
    structured_data = []
    for video_id, data in annotations.items():
        structured_data.append({
            'video_id': video_id,
            'labels': data.get('labels', []),
            'duration': data.get('duration', 0),
            'frames': data.get('frames', [])
        })
    return structured_data

if __name__ == "__main__":
    annotation_file = "c:/Users/ujwal/OneDrive/Documents/GitHub/Cross_Attention_CNN_Research_Execution/data/raw/train-annotation/annotation_training.pkl"
    annotations = load_annotations(annotation_file)
    if annotations:
        structured_annotations = parse_annotations(annotations)
        print(f"Parsed {len(structured_annotations)} annotations.")
