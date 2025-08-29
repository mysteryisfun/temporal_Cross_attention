import os
import logging
import yaml
import datetime
import sys
import numpy as np

# Import TensorFlow within a try block to handle possible import errors
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Some logging features will be disabled.")

# Import Weights & Biases within a try block
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Some logging features will be disabled.")


class ResearchLogger:
    """
    Comprehensive logging system for research experiments.
    Combines file logging, console output, TensorBoard, and optional W&B support.
    """
    
    def __init__(self, config_path=None, experiment_name=None):
        """
        Initialize the logger with configuration.
        
        Args:
            config_path (str): Path to the logging configuration YAML
            experiment_name (str): Name for this experiment run
        """
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name or f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup standard Python logger
        self._setup_file_logger()
        
        # Setup experiment trackers
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if self.config["experiment_tracking"]["tensorboard"]["enabled"] and TENSORFLOW_AVAILABLE:
            self._setup_tensorboard()
            
        if self.config["experiment_tracking"]["wandb"]["enabled"] and WANDB_AVAILABLE:
            self._setup_wandb()
            
        self.logger.info(f"Research logger initialized for experiment: {self.experiment_name}")
        
    def _load_config(self, config_path):
        """Load logging configuration from YAML file"""
        default_config = {
            "general": {
                "log_level": "INFO",
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_to_file": True,
                "log_file_path": "logs/research.log"
            },
            "experiment_tracking": {
                "wandb": {"enabled": False, "project_name": "personality-trait-prediction", "entity": None},
                "tensorboard": {"enabled": False, "log_dir": "logs/tensorboard"},
                "custom_db": {"enabled": False, "db_path": "logs/experiment_tracker.csv"}
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print("Warning: Logging config not found. Using default configuration.")
            return default_config
    
    def _setup_file_logger(self):
        """Configure the Python logging module"""
        self.logger = logging.getLogger(self.experiment_name)
        
        # Get configuration
        log_level_str = self.config["general"]["log_level"]
        log_level = getattr(logging, log_level_str)
        log_format = self.config["general"]["log_format"]
        
        # Configure logger
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if enabled
        if self.config["general"]["log_to_file"]:
            log_file_path = self.config["general"]["log_file_path"]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_tensorboard(self):
        """Set up TensorBoard logging"""
        try:
            log_dir = self.config["experiment_tracking"]["tensorboard"]["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            
            full_log_dir = os.path.join(log_dir, self.experiment_name)
            self.tensorboard_writer = tf.summary.create_file_writer(full_log_dir)
            self.logger.info(f"TensorBoard logging enabled at {full_log_dir}")
        except Exception as e:
            self.logger.warning(f"TensorBoard setup failed: {str(e)}")
    
    def _setup_wandb(self):
        """Set up Weights & Biases logging"""
        if WANDB_AVAILABLE:
            wandb_config = self.config["experiment_tracking"]["wandb"]
            self.wandb_run = wandb.init(
                project=wandb_config["project_name"],
                entity=wandb_config["entity"],
                name=self.experiment_name,
                config=self.config,
                reinit=True
            )
            self.logger.info(f"Weights & Biases logging enabled for project {wandb_config['project_name']}")
        else:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to all enabled tracking systems
        
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int): Current step (epoch or iteration)
        """
        # Log to Python logger at INFO level
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics at step {step}: {metrics_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer and TENSORFLOW_AVAILABLE:
            with self.tensorboard_writer.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(name, value, step=step)
                self.tensorboard_writer.flush()
        
        # Log to W&B
        if self.wandb_run and WANDB_AVAILABLE:
            self.wandb_run.log(metrics, step=step)
    
    def log_model_architecture(self, model):
        """
        Log model architecture summary
        
        Args:
            model (tf.keras.Model): TensorFlow/Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Cannot log model architecture.")
            return
            
        self.logger.info(f"Model Architecture: {model.__class__.__name__}")
        
        # Count trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        total_params = model.count_params()
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Model summary as string
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        summary_str = "\n".join(model_summary)
        self.logger.info(f"Model Summary:\n{summary_str}")
        
        # Log to W&B if enabled
        if self.wandb_run and WANDB_AVAILABLE and hasattr(wandb, "watch"):
            self.wandb_run.watch(model)
    
    def log_dataset_stats(self, dataset_info):
        """
        Log dataset statistics
        
        Args:
            dataset_info (dict): Dictionary containing dataset statistics
        """
        self.logger.info("Dataset Statistics:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")
              # Log to experiment trackers
        if self.wandb_run and WANDB_AVAILABLE:
            self.wandb_run.config.update({"dataset": dataset_info})
    
    def log_hyperparameters(self, hparams, step=0):
        """
        Log hyperparameters
        
        Args:
            hparams (dict): Dictionary of hyperparameters
            step (int): Current step (epoch or iteration)
        """
        self.logger.info("Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
            
        # Log to experiment trackers
        if self.tensorboard_writer and TENSORFLOW_AVAILABLE:
            with self.tensorboard_writer.as_default():
                # TensorBoard hparams can be complex to set up properly in TF
                # Here we just log them as text for simplicity
                tf.summary.text(
                    "hyperparameters", 
                    tf.constant("\n".join([f"{k}: {v}" for k, v in hparams.items()])),
                    step=step
                )
                self.tensorboard_writer.flush()
            
        if self.wandb_run and WANDB_AVAILABLE:
            self.wandb_run.config.update(hparams)
            
    def log_image(self, name, image, step=None):
        """
        Log image to experiment trackers
        
        Args:
            name (str): Image name
            image (numpy array): Image data
            step (int): Current step
        """
        # Log to TensorBoard
        if self.tensorboard_writer and TENSORFLOW_AVAILABLE:
            # Convert to tensor if needed
            if not isinstance(image, tf.Tensor):
                image_tensor = tf.convert_to_tensor(image)
            else:
                image_tensor = image
                
            # Add batch dimension if needed
            if len(image_tensor.shape) == 3:
                image_tensor = tf.expand_dims(image_tensor, 0)
                
            with self.tensorboard_writer.as_default():
                tf.summary.image(name, image_tensor, step=step)
                self.tensorboard_writer.flush()
              # Log to W&B
        if self.wandb_run and WANDB_AVAILABLE:
            self.wandb_run.log({name: wandb.Image(image)}, step=step)
    
    def log_attention_map(self, name, attention_map, original_image=None, step=None):
        """
        Log attention map visualization
        
        Args:
            name (str): Name for the attention map
            attention_map (numpy array): Attention weights
            original_image (numpy array): Original image for overlay
            step (int): Current step
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            
            # Create figure
            fig = plt.figure(figsize=(10, 5))
            canvas = FigureCanvas(fig)
            
            if original_image is not None:
                # Show original image
                plt.subplot(1, 2, 1)
                plt.imshow(original_image)
                plt.title("Original Image")
                plt.axis('off')
                
                # Show attention map overlay
                plt.subplot(1, 2, 2)
                plt.imshow(original_image)
                plt.imshow(attention_map, alpha=0.5, cmap='jet')
                plt.title("Attention Map Overlay")
                plt.axis('off')
            else:
                # Just show attention map
                plt.imshow(attention_map, cmap='jet')
                plt.title("Attention Map")
                plt.colorbar()
                plt.axis('off')
            
            plt.tight_layout()
            
            # Convert figure to numpy array
            canvas.draw()
            attention_img = np.array(canvas.buffer_rgba())
            
            plt.close(fig)
            
            # Log the visualization
            self.log_image(name, attention_img, step)
            
            return True
        except Exception as e:
            self.logger.warning(f"Could not create attention map visualization: {str(e)}")
            return False
        
    def log_system_info(self):
        """Log basic system information"""
        self.logger.info("System Information:")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Operating system: {os.name}")
        
        if TENSORFLOW_AVAILABLE:
            self.logger.info(f"TensorFlow version: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            self.logger.info(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                self.logger.info(f"  {gpu.name}")
                
    def close(self):
        """Clean up resources"""
        if self.tensorboard_writer and TENSORFLOW_AVAILABLE:
            self.tensorboard_writer.flush()
            
        if self.wandb_run and WANDB_AVAILABLE:
            self.wandb_run.finish()
            
        self.logger.info(f"Experiment {self.experiment_name} logging completed")
        
        # Remove handlers to avoid duplicate logs
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Create a global logger instance with default configuration
def get_logger(config_path=None, experiment_name=None):
    """
    Get a logger instance.
    
    Args:
        config_path (str): Path to the logging configuration YAML
        experiment_name (str): Name for this experiment run
        
    Returns:
        ResearchLogger: Logger instance
    """
    return ResearchLogger(config_path, experiment_name)
