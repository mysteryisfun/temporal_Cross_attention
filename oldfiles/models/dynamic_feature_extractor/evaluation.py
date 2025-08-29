import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
from umap import UMAP

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.visualization import FeatureVisualization, visualize_optical_flow_sequence
from utils.logger import get_logger

class FeatureExtractorEvaluator:
    """
    Evaluator for the dynamic feature extractor model.
    Provides tools for qualitative and quantitative evaluation of temporal features.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to a saved model. If None, creates a new model.
        """
        # Initialize logger
        self.logger = get_logger(experiment_name="dynamic_feature_extractor_evaluation")
        
        # Initialize feature extractor
        self.extractor = DynamicFeatureExtractor()
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            success = self.extractor.load_model(model_path)
            if success:
                self.logger.logger.info(f"Loaded model from {model_path}")
            else:
                self.logger.logger.warning(f"Failed to load model from {model_path}, using new model")
    
    def evaluate_feature_consistency(self, flow_dir, num_sequences=5):
        """
        Evaluate feature consistency across optical flow sequences from the same video.
        
        Args:
            flow_dir (str): Directory containing optical flow sequences from the same video.
            num_sequences (int): Number of sequences to evaluate.
            
        Returns:
            float: Average cosine similarity between feature vectors.
        """
        # Get a list of flow sequence files
        flow_files = []
        for root, _, files in os.walk(flow_dir):
            for file in files:
                if file.endswith('_flow_raw.npy'):
                    flow_files.append(os.path.join(root, file))
        
        if not flow_files:
            self.logger.logger.warning(f"No flow files found in {flow_dir}")
            return None
        
        # Limit the number of sequences
        flow_files = flow_files[:min(num_sequences, len(flow_files))]
        
        # Extract features from all sequences
        all_features = []
        for flow_file in flow_files:
            self.logger.logger.info(f"Processing flow file: {flow_file}")
            
            # Load flow sequence
            try:
                flow_sequence = np.load(flow_file)
                
                # Reshape flow sequence for the model if needed
                # Assuming model expects [batch, frames, height, width, channels]
                if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                    flow_sequence = np.expand_dims(flow_sequence, 0)
                
                # Preprocess flow sequence
                preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
                
                # Extract features
                features = self.extractor.extract_features(preprocessed)
                all_features.append(features[0])
            except Exception as e:
                self.logger.logger.error(f"Error processing {flow_file}: {str(e)}")
                continue
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                # Normalize feature vectors
                f1 = all_features[i] / np.linalg.norm(all_features[i])
                f2 = all_features[j] / np.linalg.norm(all_features[j])
                
                # Compute cosine similarity
                similarity = np.dot(f1, f2)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            self.logger.logger.info(f"Average cosine similarity: {avg_similarity:.4f}")
            return avg_similarity
        else:
            self.logger.logger.warning("No valid comparisons could be made")
            return None
    
    def evaluate_feature_discriminability(self, flow_dirs, num_sequences_per_dir=3):
        """
        Evaluate feature discriminability between optical flow sequences from different videos.
        
        Args:
            flow_dirs (list): List of directories containing optical flow sequences from different videos.
            num_sequences_per_dir (int): Number of sequences to evaluate per directory.
            
        Returns:
            float: Average cosine distance between feature vectors from different videos.
        """
        # Extract features from each directory
        dir_features = []
        
        for flow_dir in flow_dirs:
            if not os.path.isdir(flow_dir):
                continue
                
            # Get flow files
            flow_files = []
            for root, _, files in os.walk(flow_dir):
                for file in files:
                    if file.endswith('_flow_raw.npy'):
                        flow_files.append(os.path.join(root, file))
            
            if not flow_files:
                continue
                
            # Limit the number of sequences
            flow_files = flow_files[:min(num_sequences_per_dir, len(flow_files))]
            
            # Extract features from all sequences in this directory
            video_features = []
            for flow_file in flow_files:
                try:
                    # Load flow sequence
                    flow_sequence = np.load(flow_file)
                    
                    # Reshape if needed
                    if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                        flow_sequence = np.expand_dims(flow_sequence, 0)
                    
                    # Preprocess
                    preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
                    
                    # Extract features
                    features = self.extractor.extract_features(preprocessed)
                    video_features.append(features[0])
                except Exception as e:
                    self.logger.logger.error(f"Error processing {flow_file}: {str(e)}")
                    continue
            
            if video_features:
                # Average the features for this video
                dir_features.append(np.mean(video_features, axis=0))
        
        # Compute inter-video distances
        distances = []
        for i in range(len(dir_features)):
            for j in range(i+1, len(dir_features)):
                # Normalize feature vectors
                f1 = dir_features[i] / np.linalg.norm(dir_features[i])
                f2 = dir_features[j] / np.linalg.norm(dir_features[j])
                
                # Compute cosine distance (1 - similarity)
                distance = 1.0 - np.dot(f1, f2)
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            self.logger.logger.info(f"Average inter-video cosine distance: {avg_distance:.4f}")
            return avg_distance
        else:
            self.logger.logger.warning("No valid video comparisons could be made")
            return None
    
    def visualize_layer_activations(self, flow_sequence_path, layer_name=None, frame_idx=0, num_features=64, save_dir=None):
        """
        Visualize activations of a specific layer for a given optical flow sequence.
        
        Args:
            flow_sequence_path (str): Path to the optical flow sequence.
            layer_name (str): Name of the layer to visualize. If None, uses the final conv layer.
            frame_idx (int): Index of the frame to visualize.
            num_features (int): Number of features to visualize.
            save_dir (str): Directory to save the visualizations.
        """
        # If layer_name is not provided, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.extractor.model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D):
                    layer_name = layer.name
                    break
            self.logger.logger.info(f"Using layer {layer_name}")
        
        # Create a visualization object for this layer
        visualizer = FeatureVisualization(
            model=self.extractor.model,
            layer_name=layer_name
        )
        
        # Load and preprocess the flow sequence
        try:
            flow_sequence = np.load(flow_sequence_path)
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess
            preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
            
            # Get activations
            activations = visualizer.get_intermediate_activations(preprocessed)
            
            # Save path
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.basename(flow_sequence_path).split('.')[0]
                save_path = os.path.join(save_dir, f"{base_name}_{layer_name}_frame{frame_idx}_activations.png")
            else:
                save_path = None
            
            # Visualize activations for the specified frame
            visualizer.visualize_layer_activations(
                activations,
                frame_idx=frame_idx,
                num_features=num_features,
                figsize=(12, 12),
                save_path=save_path
            )
            
            if save_path:
                self.logger.logger.info(f"Saved activation visualization to {save_path}")
            
            return activations
            
        except Exception as e:
            self.logger.logger.error(f"Error visualizing layer activations: {str(e)}")
            return None
    
    def visualize_temporal_activation(self, flow_sequence_path, layer_name=None, feature_idx=0, save_dir=None):
        """
        Visualize how a feature activates over time across the sequence.
        
        Args:
            flow_sequence_path (str): Path to the optical flow sequence.
            layer_name (str): Name of the layer to visualize. If None, uses the final conv layer.
            feature_idx (int): Index of the feature to visualize.
            save_dir (str): Directory to save the visualizations.
        """
        # If layer_name is not provided, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.extractor.model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D):
                    layer_name = layer.name
                    break
            self.logger.logger.info(f"Using layer {layer_name}")
        
        # Create a visualization object for this layer
        visualizer = FeatureVisualization(
            model=self.extractor.model,
            layer_name=layer_name
        )
        
        # Load and preprocess the flow sequence
        try:
            flow_sequence = np.load(flow_sequence_path)
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess
            preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
            
            # Get activations
            activations = visualizer.get_intermediate_activations(preprocessed)
            
            # Save path
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.basename(flow_sequence_path).split('.')[0]
                save_path = os.path.join(save_dir, f"{base_name}_{layer_name}_feature{feature_idx}_temporal.png")
            else:
                save_path = None
            
            # Visualize temporal activation
            visualizer.visualize_temporal_activation(
                activations,
                feature_idx=feature_idx,
                save_path=save_path
            )
            
            if save_path:
                self.logger.logger.info(f"Saved temporal activation visualization to {save_path}")
            
            return activations
            
        except Exception as e:
            self.logger.logger.error(f"Error visualizing temporal activation: {str(e)}")
            return None
    
    def visualize_flow_and_activations(self, flow_sequence_path, layer_name=None, save_dir=None):
        """
        Visualize optical flow and corresponding CNN activations.
        
        Args:
            flow_sequence_path (str): Path to the optical flow sequence.
            layer_name (str): Name of the layer to visualize. If None, uses the final conv layer.
            save_dir (str): Directory to save the visualizations.
        """
        # If layer_name is not provided, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.extractor.model.layers):
                if isinstance(layer, tf.keras.layers.Conv3D):
                    layer_name = layer.name
                    break
            self.logger.logger.info(f"Using layer {layer_name}")
        
        # Create a visualization object for this layer
        visualizer = FeatureVisualization(
            model=self.extractor.model,
            layer_name=layer_name
        )
        
        # Load and preprocess the flow sequence
        try:
            flow_sequence = np.load(flow_sequence_path)
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess
            preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
            
            # Get activations
            activations = visualizer.get_intermediate_activations(preprocessed)
            
            # Save path
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.basename(flow_sequence_path).split('.')[0]
                save_path = os.path.join(save_dir, f"{base_name}_{layer_name}_flow_activations.png")
            else:
                save_path = None
            
            # Visualize flow and activations
            visualizer.visualize_flow_and_activations(
                preprocessed,
                activations,
                save_path=save_path
            )
            
            if save_path:
                self.logger.logger.info(f"Saved flow and activations visualization to {save_path}")
            
            return activations
            
        except Exception as e:
            self.logger.logger.error(f"Error visualizing flow and activations: {str(e)}")
            return None
    
    def plot_feature_dimwise_distribution(self, flow_dirs, num_sequences_per_dir=3, save_path=None):
        """
        Plot the distribution of feature values along each dimension.
        
        Args:
            flow_dirs (list): List of directories containing optical flow sequences.
            num_sequences_per_dir (int): Number of sequences to evaluate per directory.
            save_path (str): Path to save the visualization.
        """
        all_features = []
        
        for flow_dir in flow_dirs:
            if not os.path.isdir(flow_dir):
                continue
            
            # Get flow files
            flow_files = []
            for root, _, files in os.walk(flow_dir):
                for file in files:
                    if file.endswith('_flow_raw.npy'):
                        flow_files.append(os.path.join(root, file))
            
            if not flow_files:
                continue
            
            # Limit the number of sequences
            flow_files = flow_files[:min(num_sequences_per_dir, len(flow_files))]
            
            # Extract features
            for flow_file in flow_files:
                try:
                    flow_sequence = np.load(flow_file)
                    
                    if len(flow_sequence.shape) == 4:
                        flow_sequence = np.expand_dims(flow_sequence, 0)
                    
                    preprocessed = self.extractor.preprocess_flow_sequence(flow_sequence)
                    features = self.extractor.extract_features(preprocessed)
                    all_features.append(features[0])
                except Exception as e:
                    self.logger.logger.error(f"Error processing {flow_file}: {str(e)}")
                    continue
        
        if not all_features:
            self.logger.logger.warning("No features were extracted")
            return
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Compute statistics
        means = np.mean(features_array, axis=0)
        stds = np.std(features_array, axis=0)
        
        # Plot feature statistics
        plt.figure(figsize=(12, 6))
        
        # Plot mean values
        plt.subplot(1, 2, 1)
        plt.plot(means)
        plt.title('Mean Feature Values')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Mean Value')
        plt.grid(alpha=0.3)
        
        # Plot standard deviations
        plt.subplot(1, 2, 2)
        plt.plot(stds)
        plt.title('Feature Standard Deviations')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Standard Deviation')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            self.logger.logger.info(f"Saved feature distribution visualization to {save_path}")
        else:
            plt.show()
    
    def run_comprehensive_evaluation(self, flow_data_dir, output_dir="results/dynamic_evaluations"):
        """
        Run a comprehensive evaluation of the dynamic feature extractor.
        
        Args:
            flow_data_dir (str): Directory containing optical flow data organized by video.
            output_dir (str): Directory to save evaluation results.
            
        Returns:
            dict: Evaluation results summary.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all subdirectories (each representing a different video)
        video_dirs = [os.path.join(flow_data_dir, d) for d in os.listdir(flow_data_dir) 
                    if os.path.isdir(os.path.join(flow_data_dir, d))]
        
        if len(video_dirs) < 2:
            self.logger.logger.error(f"Not enough video directories in {flow_data_dir}")
            return {}
        
        # 1. Evaluate feature consistency within videos
        self.logger.logger.info("Evaluating within-video feature consistency")
        consistency_scores = []
        
        for video_dir in video_dirs[:5]:  # Limit to 5 videos
            score = self.evaluate_feature_consistency(video_dir, num_sequences=3)
            if score is not None:
                consistency_scores.append(score)
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            self.logger.logger.info(f"Average within-video consistency: {avg_consistency:.4f}")
            
            # Save consistency results
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(consistency_scores)), consistency_scores)
            plt.axhline(y=avg_consistency, color='r', linestyle='-', label=f"Average: {avg_consistency:.4f}")
            plt.xlabel("Video")
            plt.ylabel("Feature Consistency (Cosine Similarity)")
            plt.title("Within-Video Feature Consistency")
            plt.legend()
            
            consistency_path = os.path.join(output_dir, "within_video_consistency.png")
            plt.savefig(consistency_path)
            plt.close()
        
        # 2. Evaluate feature discriminability between videos
        self.logger.logger.info("Evaluating between-video feature discriminability")
        discriminability = self.evaluate_feature_discriminability(video_dirs[:10], num_sequences_per_dir=3)
        
        # 3. Visualize feature dimension-wise distribution
        self.logger.logger.info("Generating feature distribution visualization")
        dist_path = os.path.join(output_dir, "feature_distribution.png")
        self.plot_feature_dimwise_distribution(video_dirs[:10], num_sequences_per_dir=2, save_path=dist_path)
        
        # 4. Visualize layer activations and temporal patterns for a sample sequence
        self.logger.logger.info("Visualizing layer activations and temporal patterns")
        
        # Find a sample flow sequence
        sample_flow = None
        for dir_path in video_dirs:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('_flow_raw.npy'):
                        sample_flow = os.path.join(root, file)
                        break
                if sample_flow:
                    break
            if sample_flow:
                break
        
        if sample_flow:
            # Layer activations
            self.visualize_layer_activations(
                sample_flow,
                layer_name=None,  # Use the last convolutional layer
                frame_idx=0,
                save_dir=output_dir
            )
            
            # Temporal activation
            self.visualize_temporal_activation(
                sample_flow,
                layer_name=None,  # Use the last convolutional layer
                feature_idx=0,
                save_dir=output_dir
            )
            
            # Flow and activations
            self.visualize_flow_and_activations(
                sample_flow,
                layer_name=None,  # Use the last convolutional layer
                save_dir=output_dir
            )
        
        self.logger.logger.info("Comprehensive evaluation completed")
        
        # Return a summary of the evaluation
        return {
            "avg_within_video_consistency": float(avg_consistency) if consistency_scores else None,
            "between_video_discriminability": float(discriminability) if discriminability else None
        }
