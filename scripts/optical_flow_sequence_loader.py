import os
import numpy as np
import cv2
import tensorflow as tf
import sys
from pathlib import Path
import glob

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from scripts.preprocessing.optical_flow_computer import OpticalFlowComputer

class OpticalFlowSequenceLoader:
    """
    Utility class for loading and processing optical flow sequences.
    """
    
    def __init__(self, flow_dir=None, sequence_length=16, resize_shape=(224, 224)):
        """
        Initialize the optical flow sequence loader.
        
        Args:
            flow_dir (str): Directory containing optical flow data.
            sequence_length (int): Length of the flow sequences to generate.
            resize_shape (tuple): Shape to resize flow frames to (height, width).
        """
        self.flow_dir = flow_dir
        self.sequence_length = sequence_length
        self.resize_shape = resize_shape
        self.logger = get_logger(experiment_name="optical_flow_loader")
    
    def find_flow_files(self, directory=None):
        """
        Find all optical flow files in a directory.
        
        Args:
            directory (str): Directory to search in. If None, uses self.flow_dir.
            
        Returns:
            list: List of paths to optical flow files.
        """
        if directory is None:
            directory = self.flow_dir
            
        if directory is None:
            self.logger.logger.error("No directory specified for finding flow files")
            return []
        
        flow_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('_flow_raw.npy'):
                    flow_files.append(os.path.join(root, file))
        
        self.logger.logger.info(f"Found {len(flow_files)} flow files in {directory}")
        
        return flow_files
    
    def load_raw_flow_sequence(self, flow_file):
        """
        Load a raw optical flow sequence from a file.
        
        Args:
            flow_file (str): Path to the flow file.
            
        Returns:
            numpy.ndarray: Optical flow sequence with shape [frames, height, width, channels].
        """
        try:
            flow_sequence = np.load(flow_file)
            self.logger.logger.info(f"Loaded flow sequence with shape {flow_sequence.shape} from {flow_file}")
            return flow_sequence
        except Exception as e:
            self.logger.logger.error(f"Error loading flow file {flow_file}: {str(e)}")
            return None
    
    def preprocess_flow_sequence(self, flow_sequence):
        """
        Preprocess an optical flow sequence for the model.
        
        Args:
            flow_sequence (numpy.ndarray): Optical flow sequence with shape [frames, height, width, channels].
            
        Returns:
            numpy.ndarray: Preprocessed flow sequence with shape [1, frames, height, width, channels].
        """
        # Resize frames if needed
        if flow_sequence.shape[1:3] != self.resize_shape:
            resized_sequence = np.zeros((flow_sequence.shape[0], self.resize_shape[0], self.resize_shape[1], 2))
            
            for i in range(flow_sequence.shape[0]):
                # Resize each channel separately to maintain flow values
                for c in range(2):
                    resized_sequence[i, :, :, c] = cv2.resize(
                        flow_sequence[i, :, :, c], 
                        (self.resize_shape[1], self.resize_shape[0])
                    )
            
            flow_sequence = resized_sequence
        
        # Standardize flow values
        flow_sequence = self._standardize_flow(flow_sequence)
        
        # Add batch dimension
        flow_sequence = np.expand_dims(flow_sequence, 0)
        
        return flow_sequence
    
    def _standardize_flow(self, flow_sequence):
        """
        Standardize optical flow values to a consistent range.
        
        Args:
            flow_sequence (numpy.ndarray): Optical flow sequence.
            
        Returns:
            numpy.ndarray: Standardized flow sequence.
        """
        # Optical flow values are typically in the range [-20, 20]
        # We scale them to [-1, 1] for better conditioning
        return np.clip(flow_sequence / 20.0, -1.0, 1.0)
    
    def adjust_sequence_length(self, flow_sequence, target_length=None):
        """
        Adjust the length of a flow sequence to match the target length.
        
        Args:
            flow_sequence (numpy.ndarray): Optical flow sequence.
            target_length (int): Target sequence length. If None, uses self.sequence_length.
            
        Returns:
            numpy.ndarray: Adjusted flow sequence.
        """
        if target_length is None:
            target_length = self.sequence_length
        
        current_length = flow_sequence.shape[0]
        
        if current_length == target_length:
            return flow_sequence
        
        # If sequence is too short, loop or pad
        if current_length < target_length:
            # If very short, loop the sequence
            if current_length <= target_length // 2:
                repetitions = int(np.ceil(target_length / current_length))
                extended = np.tile(flow_sequence, (repetitions, 1, 1, 1))
                return extended[:target_length]
            # Otherwise, pad with zeros
            else:
                padding = np.zeros((target_length - current_length, *flow_sequence.shape[1:]))
                return np.concatenate([flow_sequence, padding], axis=0)
        
        # If sequence is too long, sample frames uniformly
        else:
            indices = np.linspace(0, current_length - 1, target_length).astype(int)
            return flow_sequence[indices]
    
    def create_sequence_batch(self, flow_files, batch_size=8):
        """
        Create a batch of flow sequences from multiple files.
        
        Args:
            flow_files (list): List of paths to flow files.
            batch_size (int): Batch size.
            
        Returns:
            numpy.ndarray: Batch of flow sequences with shape [batch_size, frames, height, width, channels].
        """
        # Limit to batch_size
        flow_files = flow_files[:batch_size]
        
        # Initialize batch
        batch = []
        
        for flow_file in flow_files:
            # Load flow sequence
            flow_sequence = self.load_raw_flow_sequence(flow_file)
            
            if flow_sequence is None:
                continue
            
            # Adjust sequence length
            flow_sequence = self.adjust_sequence_length(flow_sequence)
            
            # Resize if needed
            if flow_sequence.shape[1:3] != self.resize_shape:
                resized_sequence = np.zeros((flow_sequence.shape[0], self.resize_shape[0], self.resize_shape[1], 2))
                
                for i in range(flow_sequence.shape[0]):
                    for c in range(2):
                        resized_sequence[i, :, :, c] = cv2.resize(
                            flow_sequence[i, :, :, c], 
                            (self.resize_shape[1], self.resize_shape[0])
                        )
                
                flow_sequence = resized_sequence
            
            # Standardize flow values
            flow_sequence = self._standardize_flow(flow_sequence)
            
            batch.append(flow_sequence)
        
        if not batch:
            self.logger.logger.warning("No valid flow sequences found")
            return None
        
        # Stack into a batch
        batch = np.stack(batch, axis=0)
        
        return batch
    
    def generate_flow_data(self, directory=None, batch_size=8, shuffle=True):
        """
        Generate batches of flow sequences for training or evaluation.
        
        Args:
            directory (str): Directory containing flow files. If None, uses self.flow_dir.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the flow files.
            
        Returns:
            generator: Generator yielding batches of flow sequences.
        """
        # Find flow files
        flow_files = self.find_flow_files(directory)
        
        if not flow_files:
            self.logger.logger.error("No flow files found")
            return
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(flow_files)
        
        # Generate batches
        for i in range(0, len(flow_files), batch_size):
            batch_files = flow_files[i:i+batch_size]
            batch = self.create_sequence_batch(batch_files, batch_size=batch_size)
            
            if batch is not None:
                yield batch
    
    def compute_and_save_flow(self, video_path, output_dir, method='farneback', frame_step=1, clip_length=16):
        """
        Compute optical flow from a video and save it as a sequence.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (str): Directory to save the flow sequence.
            method (str): Optical flow method ('farneback' or 'tvl1').
            frame_step (int): Step between frames for flow computation.
            clip_length (int): Length of the flow sequence to save.
            
        Returns:
            str: Path to the saved flow sequence.
        """
        # Create flow computer
        flow_computer = OpticalFlowComputer(method=method)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.logger.error(f"Error opening video file {video_path}")
            return None
        
        # Get video info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.logger.info(f"Processing video with {frame_count} frames, size: {width}x{height}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base output filename
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{video_name}_flow_raw.npy")
        
        # Initialize flow sequence
        flow_sequence = []
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            self.logger.logger.error("Failed to read the first frame")
            cap.release()
            return None
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Process frames
        frame_idx = 0
        
        while True:
            # Skip frames according to frame_step
            for _ in range(frame_step):
                ret, curr_frame = cap.read()
                if not ret:
                    break
            
            if not ret:
                break
            
            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = flow_computer.compute_flow(prev_gray, curr_gray)
            
            # Append to sequence
            flow_sequence.append(flow)
            
            # Update previous frame
            prev_gray = curr_gray
            
            frame_idx += 1
            
            # Stop if we have enough frames
            if len(flow_sequence) >= clip_length:
                break
        
        # Release video
        cap.release()
        
        # Convert to numpy array
        if flow_sequence:
            flow_sequence = np.array(flow_sequence)
            
            # Save flow sequence
            np.save(output_path, flow_sequence)
            
            self.logger.logger.info(f"Saved flow sequence with shape {flow_sequence.shape} to {output_path}")
            
            return output_path
        else:
            self.logger.logger.error("No flow frames were computed")
            return None

def load_and_preprocess_optical_flow_directory(directory, sequence_length=16, resize_shape=(224, 224), batch_size=8):
    """
    Load and preprocess all optical flow sequences in a directory.
    
    Args:
        directory (str): Directory containing optical flow files.
        sequence_length (int): Length of flow sequences.
        resize_shape (tuple): Shape to resize flow frames to.
        batch_size (int): Batch size for processing.
        
    Returns:
        numpy.ndarray: Preprocessed flow sequences.
    """
    loader = OpticalFlowSequenceLoader(
        flow_dir=directory,
        sequence_length=sequence_length,
        resize_shape=resize_shape
    )
    
    # Generate data
    data_generator = loader.generate_flow_data(
        directory=directory,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Process all batches
    all_sequences = []
    
    for batch in data_generator:
        all_sequences.append(batch)
    
    if not all_sequences:
        loader.logger.logger.warning("No sequences were processed")
        return None
    
    # Concatenate all batches
    all_sequences = np.concatenate(all_sequences, axis=0)
    
    loader.logger.logger.info(f"Loaded and preprocessed {all_sequences.shape[0]} sequences")
    
    return all_sequences

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Optical Flow Sequence Loader")
    parser.add_argument("--mode", type=str, choices=["load", "compute"], default="load",
                       help="Operation mode: load existing flows or compute new ones")
    parser.add_argument("--input_dir", type=str, default="data/optical_flow",
                       help="Directory containing optical flow data (for load mode)")
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to the video file (for compute mode)")
    parser.add_argument("--output_dir", type=str, default="data/optical_flow/processed",
                       help="Directory to save processed flow sequences (for compute mode)")
    parser.add_argument("--sequence_length", type=int, default=16,
                       help="Length of flow sequences")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(experiment_name="optical_flow_loader_main")
    
    if args.mode == "load":
        # Load and preprocess existing flows
        flows = load_and_preprocess_optical_flow_directory(
            directory=args.input_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size
        )
        
        if flows is not None:
            logger.logger.info(f"Successfully loaded {flows.shape[0]} flow sequences with shape {flows.shape[1:]}")
    
    elif args.mode == "compute":
        if args.video_path is None:
            logger.logger.error("Video path must be provided for compute mode")
            sys.exit(1)
        
        # Compute new flows
        loader = OpticalFlowSequenceLoader(
            flow_dir=args.input_dir,
            sequence_length=args.sequence_length
        )
        
        output_path = loader.compute_and_save_flow(
            video_path=args.video_path,
            output_dir=args.output_dir,
            clip_length=args.sequence_length
        )
        
        if output_path:
            logger.logger.info(f"Successfully computed and saved flow sequence to {output_path}")
            
            # Load and visualize
            from models.dynamic_feature_extractor.visualization import visualize_optical_flow_sequence
            
            flow_sequence = loader.load_raw_flow_sequence(output_path)
            if flow_sequence is not None:
                viz_path = os.path.join(args.output_dir, f"{os.path.basename(output_path).split('.')[0]}_viz.png")
                visualize_optical_flow_sequence(flow_sequence, save_path=viz_path)
                logger.logger.info(f"Flow visualization saved to {viz_path}")
    
    else:
        logger.logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
