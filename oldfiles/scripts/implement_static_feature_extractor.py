import os
import argparse
import sys
import tensorflow as tf
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from models.static_feature_extractor.feature_extractor import StaticFeatureExtractor
from models.static_feature_extractor.evaluation import FeatureExtractorEvaluator

def main():
    """
    Main function to implement and run the static feature extractor for Phase 3.1.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Static Feature Extractor Implementation")
    parser.add_argument("--data_dir", type=str, default="data/faces",
                       help="Directory containing face images")
    parser.add_argument("--output_dir", type=str, default="results/static_feature_extractor",
                       help="Directory to save model and results")
    parser.add_argument("--config_path", type=str, default="config/model_config/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--mode", type=str, choices=["implement", "test", "evaluate", "all"], default="all",
                       help="Mode to run: implement, test, evaluate, or all")
    parser.add_argument("--num_test_samples", type=int, default=5,
                       help="Number of test samples to process")
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(experiment_name="phase_3_1_static_feature_extractor")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.logger.info("Starting Phase 3.1: Static Feature Extractor Implementation")
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
        logger.logger.info("Implementing static feature extractor")
        
        # Initialize the feature extractor
        feature_extractor = StaticFeatureExtractor(config_path=args.config_path)
        
        # Save the model
        model_path = os.path.join(args.output_dir, "static_feature_extractor_model")
        feature_extractor.save_model(model_path)
        
        logger.logger.info(f"Feature extractor implemented and saved to {model_path}")
    
    # Test feature extractor
    if args.mode in ["test", "all"]:
        logger.logger.info("Testing static feature extractor")
        
        # Run test script
        import scripts.test_feature_extractor as test_script
        test_script.main()
        
        logger.logger.info("Feature extractor testing completed")
    
    # Evaluate feature extractor
    if args.mode in ["evaluate", "all"]:
        logger.logger.info("Evaluating static feature extractor")
        
        # Initialize feature extractor
        model_path = os.path.join(args.output_dir, "static_feature_extractor_model")
        if os.path.exists(model_path):
            evaluator = FeatureExtractorEvaluator(model_path=model_path)
        else:
            evaluator = FeatureExtractorEvaluator()
        
        # Run comprehensive evaluation
        evaluation_results = evaluator.run_comprehensive_evaluation(
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "evaluations")
        )
        
        # Log evaluation results
        logger.logger.info(f"Evaluation results: {evaluation_results}")
    
    logger.logger.info("Phase 3.1 completed successfully")
    
if __name__ == "__main__":
    main()
