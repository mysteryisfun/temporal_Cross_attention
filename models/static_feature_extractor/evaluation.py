import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from tensorflow.keras.preprocessing import image
from models.static_feature_extractor.feature_extractor import StaticFeatureExtractor
from models.static_feature_extractor.visualization import FeatureVisualization
from utils.logger import get_logger

class FeatureExtractorEvaluator:
    """
    Evaluator for the static feature extractor model.
    Provides tools for qualitative and quantitative evaluation.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to a saved model. If None, a new model is created.
        """
        self.logger = get_logger(experiment_name="feature_extractor_evaluation")
        
        if model_path and os.path.exists(model_path):
            self.extractor = StaticFeatureExtractor()
            self.extractor.load_model(model_path)
            self.logger.logger.info(f"Loaded model from {model_path}")
        else:
            self.extractor = StaticFeatureExtractor()
            self.logger.logger.info("Created new model")
        
        self.visualizer = FeatureVisualization(model=self.extractor.model)
    
    def evaluate_feature_consistency(self, image_dir, num_images=10):
        """
        Evaluate the consistency of features extracted from similar images.
        
        Args:
            image_dir (str): Directory containing similar images.
            num_images (int): Number of images to evaluate.
            
        Returns:
            float: Average cosine similarity between feature vectors.
        """
        self.logger.logger.info(f"Evaluating feature consistency on {image_dir}")
        
        # List image files
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                     if os.path.isfile(os.path.join(image_dir, f)) and 
                     f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < 2:
            self.logger.logger.warning(f"Not enough images in {image_dir} for consistency evaluation")
            return None
        
        # Limit the number of images
        image_files = image_files[:min(num_images, len(image_files))]
        
        # Extract features from all images
        all_features = []
        for img_path in image_files:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            preprocessed = self.extractor.preprocess_image(img_array)
            features = self.extractor.extract_features(preprocessed)
            all_features.append(features[0])
        
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
        
        avg_similarity = np.mean(similarities)
        self.logger.logger.info(f"Average cosine similarity: {avg_similarity:.4f}")
        
        return avg_similarity
    
    def evaluate_feature_discriminability(self, image_dirs, num_images_per_dir=5):
        """
        Evaluate how well the features can discriminate between different image sets.
        
        Args:
            image_dirs (list): List of directories containing different image classes.
            num_images_per_dir (int): Number of images to evaluate per directory.
            
        Returns:
            float: Average inter-class cosine distance.
        """
        self.logger.logger.info(f"Evaluating feature discriminability across {len(image_dirs)} classes")
        
        class_features = []
        
        # Extract features for each class
        for dir_path in image_dirs:
            if not os.path.isdir(dir_path):
                continue
                
            # List image files
            image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if os.path.isfile(os.path.join(dir_path, f)) and 
                         f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
                
            # Limit the number of images
            image_files = image_files[:min(num_images_per_dir, len(image_files))]
            
            # Extract features from all images in this class
            dir_features = []
            for img_path in image_files:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                preprocessed = self.extractor.preprocess_image(img_array)
                features = self.extractor.extract_features(preprocessed)
                dir_features.append(features[0])
            
            if dir_features:
                # Average the features for this class
                class_features.append(np.mean(dir_features, axis=0))
        
        # Compute inter-class distances
        distances = []
        for i in range(len(class_features)):
            for j in range(i+1, len(class_features)):
                # Normalize feature vectors
                f1 = class_features[i] / np.linalg.norm(class_features[i])
                f2 = class_features[j] / np.linalg.norm(class_features[j])
                
                # Compute cosine distance (1 - similarity)
                distance = 1.0 - np.dot(f1, f2)
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            self.logger.logger.info(f"Average inter-class cosine distance: {avg_distance:.4f}")
            return avg_distance
        else:
            self.logger.logger.warning("No valid class comparisons could be made")
            return None
    
    def visualize_layer_activations(self, image_path, layer_name=None, num_features=64, save_dir=None):
        """
        Visualize activations of a specific layer for a given image.
        
        Args:
            image_path (str): Path to the input image.
            layer_name (str): Name of the layer to visualize. If None, uses the final conv layer.
            num_features (int): Number of features to visualize.
            save_dir (str): Directory to save the visualizations.
        """
        # If layer_name is not provided, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.extractor.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
            self.logger.logger.info(f"Using layer {layer_name}")
        
        # Create a visualization object for this layer
        visualizer = FeatureVisualization(
            model=self.extractor.model,
            layer_name=layer_name
        )
        
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        preprocessed = self.extractor.preprocess_image(img_array)
        
        # Get activations
        activations = visualizer.get_intermediate_activations(preprocessed)
        
        # Save path
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(image_path).split('.')[0]
            save_path = os.path.join(save_dir, f"{base_name}_{layer_name}_activations.png")
        else:
            save_path = None
        
        # Visualize activations
        visualizer.visualize_layer_activations(
            activations,
            num_features=num_features,
            figsize=(12, 12),
            save_path=save_path
        )
        
        if save_path:
            self.logger.logger.info(f"Saved activation visualization to {save_path}")
        
        return activations
    
    def plot_feature_pca(self, image_dirs, num_images_per_dir=5, save_path=None):
        """
        Extract features from multiple image directories and visualize using PCA.
        
        Args:
            image_dirs (dict): Dictionary mapping class names to directories.
            num_images_per_dir (int): Number of images to process per directory.
            save_path (str): Path to save the visualization.
        """
        from sklearn.decomposition import PCA
        
        self.logger.logger.info("Performing PCA visualization of features")
        
        all_features = []
        all_labels = []
        
        # Extract features for each class
        for class_name, dir_path in image_dirs.items():
            if not os.path.isdir(dir_path):
                continue
                
            # List image files
            image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if os.path.isfile(os.path.join(dir_path, f)) and 
                         f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
                
            # Limit the number of images
            image_files = image_files[:min(num_images_per_dir, len(image_files))]
            
            # Extract features from all images in this class
            for img_path in image_files:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                preprocessed = self.extractor.preprocess_image(img_array)
                features = self.extractor.extract_features(preprocessed)
                all_features.append(features[0])
                all_labels.append(class_name)
        
        if len(all_features) < 2:
            self.logger.logger.warning("Not enough samples for PCA visualization")
            return
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Perform PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features_array)
        
        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        
        # Get unique labels and assign colors
        unique_labels = list(set(all_labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            # Plot points for this class
            mask = [l == label for l in all_labels]
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7
            )
        
        # Add labels and legend
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.title("PCA of CNN Features")
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            self.logger.logger.info(f"Saved PCA visualization to {save_path}")
        else:
            plt.show()
            
    def run_comprehensive_evaluation(self, data_dir, output_dir="results/evaluations"):
        """
        Run a comprehensive evaluation of the feature extractor.
        
        Args:
            data_dir (str): Directory containing face images organized by subject.
            output_dir (str): Directory to save evaluation results.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get subdirectories (each representing a different subject)
        subject_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(subject_dirs) < 2:
            self.logger.logger.error(f"Not enough subject directories in {data_dir}")
            return
        
        # 1. Evaluate feature consistency within subjects
        self.logger.logger.info("Evaluating within-subject feature consistency")
        consistency_scores = []
        
        for subject_dir in subject_dirs[:10]:  # Limit to 10 subjects
            score = self.evaluate_feature_consistency(subject_dir, num_images=5)
            if score is not None:
                consistency_scores.append(score)
                
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            self.logger.logger.info(f"Average within-subject consistency: {avg_consistency:.4f}")
            
            # Save consistency results
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(consistency_scores)), consistency_scores)
            plt.axhline(y=avg_consistency, color='r', linestyle='-', label=f"Average: {avg_consistency:.4f}")
            plt.xlabel("Subject")
            plt.ylabel("Feature Consistency (Cosine Similarity)")
            plt.title("Within-Subject Feature Consistency")
            plt.legend()
            
            consistency_path = os.path.join(output_dir, "within_subject_consistency.png")
            plt.savefig(consistency_path)
            plt.close()
            
        # 2. Evaluate feature discriminability between subjects
        self.logger.logger.info("Evaluating between-subject feature discriminability")
        discriminability = self.evaluate_feature_discriminability(subject_dirs[:10], num_images_per_dir=5)
        
        # 3. Visualize feature PCA
        self.logger.logger.info("Generating PCA visualization")
        
        # Create a dictionary of subject dirs
        subject_dict = {}
        for i, dir_path in enumerate(subject_dirs[:10]):
            subject_dict[f"Subject {i+1}"] = dir_path
            
        pca_path = os.path.join(output_dir, "feature_pca.png")
        self.plot_feature_pca(subject_dict, num_images_per_dir=5, save_path=pca_path)
        
        # 4. Visualize layer activations for a sample image
        self.logger.logger.info("Visualizing layer activations")
        
        # Find a sample image
        sample_image = None
        for dir_path in subject_dirs:
            image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if os.path.isfile(os.path.join(dir_path, f)) and 
                         f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                sample_image = image_files[0]
                break
                
        if sample_image:
            self.visualize_layer_activations(
                sample_image,
                layer_name=None,  # Use the last convolutional layer
                save_dir=output_dir
            )
            
        self.logger.logger.info("Comprehensive evaluation completed")
        
        # Return a summary of the evaluation
        return {
            "avg_within_subject_consistency": float(avg_consistency) if consistency_scores else None,
            "between_subject_discriminability": float(discriminability) if discriminability else None
        }
