import os
import argparse
import sys
import tensorflow as tf
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.dynamic_feature_extractor.feature_extractor import DynamicFeatureExtractor
from models.dynamic_feature_extractor.evaluation import FeatureExtractorEvaluator
from scripts.preprocessing.optical_flow_computer import OpticalFlowComputer

def main():
    """
    Main function to implement and run the dynamic feature extractor for Phase 3.2.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dynamic Feature Extractor Implementation")
    parser.add_argument("--data_dir", type=str, default="data/optical_flow",
                       help="Directory containing optical flow data")
    parser.add_argument("--output_dir", type=str, default="results/dynamic_feature_extractor",
                       help="Directory to save model and results")
    parser.add_argument("--config_path", type=str, default="config/model_config/dynamic_model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--mode", type=str, choices=["implement", "test", "evaluate", "all"], default="all",
                       help="Mode to run: implement, test, evaluate, or all")
    parser.add_argument("--num_test_samples", type=int, default=5,
                       help="Number of test samples to process")
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(experiment_name="phase_3_2_dynamic_feature_extractor")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.logger.info("Starting Phase 3.2: Dynamic Feature Extractor Implementation")
    logger.logger.info(f"Configuration: {args}")
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logger.logger.info(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        logger.logger.info(f"GPU: {gpu.name}")
        
    # Record system information
    logger.log_system_info()
    
    # Implement feature extractor
    if args.mode in ["implement", "all"]:
        logger.logger.info("Implementing dynamic feature extractor")
        
        # Initialize the feature extractor
        feature_extractor = DynamicFeatureExtractor(config_path=args.config_path)
        
        # Save the model
        model_path = os.path.join(args.output_dir, "dynamic_feature_extractor_model")
        feature_extractor.save_model(model_path)
        
        logger.logger.info(f"Feature extractor implemented and saved to {model_path}")
    
    # Test feature extractor
    if args.mode in ["test", "all"]:
        logger.logger.info("Testing dynamic feature extractor")
        
        # Run test function
        test_dynamic_feature_extractor(args.data_dir, args.output_dir, args.num_test_samples)
        
        logger.logger.info("Feature extractor testing completed")
    
    # Evaluate feature extractor
    if args.mode in ["evaluate", "all"]:
        logger.logger.info("Evaluating dynamic feature extractor")
        
        # Initialize feature extractor
        model_path = os.path.join(args.output_dir, "dynamic_feature_extractor_model")
        if os.path.exists(model_path):
            evaluator = FeatureExtractorEvaluator(model_path=model_path)
        else:
            evaluator = FeatureExtractorEvaluator()
        
        # Run comprehensive evaluation
        evaluation_results = evaluator.run_comprehensive_evaluation(
            flow_data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "evaluations")
        )
        
        # Log evaluation results
        logger.logger.info(f"Evaluation results: {evaluation_results}")
    
    logger.logger.info("Phase 3.2 completed successfully")

def test_dynamic_feature_extractor(data_dir, output_dir, num_samples=5):
    """
    Test the dynamic feature extractor on sample optical flow sequences.
    
    Args:
        data_dir (str): Directory containing optical flow data.
        output_dir (str): Directory to save results.
        num_samples (int): Number of samples to test.
    """
    # Initialize logger
    logger = get_logger(experiment_name="dynamic_feature_extractor_test")
    logger.logger.info("Starting dynamic feature extractor test")
    
    # Initialize feature extractor
    feature_extractor = DynamicFeatureExtractor()
    logger.logger.info("Feature extractor initialized")
    
    # Create output directories
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Find optical flow sequences
    flow_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_flow_raw.npy'):
                flow_files.append(os.path.join(root, file))
    
    if not flow_files:
        logger.logger.error(f"No optical flow files found in {data_dir}")
        return
    
    # Select a few random flow files for testing
    np.random.seed(42)  # For reproducibility
    selected_files = np.random.choice(flow_files, min(num_samples, len(flow_files)), replace=False)
    
    # Process each selected file
    for flow_file in selected_files:
        logger.logger.info(f"Processing flow file: {flow_file}")
        
        try:
            # Load optical flow sequence
            flow_sequence = np.load(flow_file)
            
            # Log flow sequence shape
            logger.logger.info(f"Flow sequence shape: {flow_sequence.shape}")
            
            # Reshape if needed
            if len(flow_sequence.shape) == 4:  # [frames, height, width, channels]
                flow_sequence = np.expand_dims(flow_sequence, 0)
            
            # Preprocess flow sequence
            preprocessed = feature_extractor.preprocess_flow_sequence(flow_sequence)
            
            # Extract features
            features = feature_extractor.extract_features(preprocessed)
            
            logger.logger.info(f"Extracted features shape: {features.shape}")
            
            # Visualize the optical flow sequence
            from models.dynamic_feature_extractor.visualization import visualize_optical_flow_sequence
            
            viz_path = os.path.join(visualization_dir, f"{os.path.basename(flow_file).split('.')[0]}_flow_viz.png")
            visualize_optical_flow_sequence(flow_sequence, save_path=viz_path)
            
            logger.logger.info(f"Flow visualization saved to {viz_path}")
            
            # Create a feature visualization for the first few dimensions
            plt_features = features[0][:20]  # Take first 20 dimensions for visualization
            
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(plt_features)), plt_features)
            plt.title("Feature Vector Sample (First 20 dimensions)")
            plt.xlabel("Feature Dimension")
            plt.ylabel("Feature Value")
            plt.grid(alpha=0.3)
            
            feature_viz_path = os.path.join(visualization_dir, f"{os.path.basename(flow_file).split('.')[0]}_features.png")
            plt.savefig(feature_viz_path)
            plt.close()
            
            logger.logger.info(f"Feature visualization saved to {feature_viz_path}")
            
        except Exception as e:
            logger.logger.error(f"Error processing {flow_file}: {str(e)}")
    
    logger.logger.info("Dynamic feature extractor test completed")

if __name__ == "__main__":
    # Import matplotlib inside the if block to avoid issues when imported in another module
    import matplotlib.pyplot as plt
    main()
