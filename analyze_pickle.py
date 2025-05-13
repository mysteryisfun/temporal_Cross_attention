import pickle
import sys

# Load the pickle file
with open('annotation_validation.pkl', 'rb') as f:
    try:
        data = pickle.load(f, encoding='latin1')
        print(f"Type of data: {type(data)}")
        # Dictionary-like object
        if hasattr(data, 'keys'):
            print(f"Keys: {list(data.keys())[:10]}")
            
            # Print a sample item
            sample_key = next(iter(data))
            print(f"Sample for key '{sample_key}': {data[sample_key]}")
        
        # DataFrame-like object
        elif hasattr(data, 'columns'):
            print(f"Columns: {data.columns}")
            print(f"Sample rows:\n{data.head()}")
        
        # List-like object
        elif hasattr(data, '__len__') and not isinstance(data, str):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {str(data[0])[:500]}")
        
        # Other types
        else:
            print(f"Data: {str(data)[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
