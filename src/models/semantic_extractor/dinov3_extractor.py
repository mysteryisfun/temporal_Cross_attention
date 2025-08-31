"""
DINOv3 Semantic Feature Extractor Implementation
===============================================

This module implements the semantic feature extraction using DINOv3 ViT-B/16 distilled model.
The model is used for extracting rich semantic features from individual video frames.

Model: facebook/dinov3-vitb16-pretrain-lvd1689m (86M parameters)
Output: 768-dimensional semantic embeddings per frame
Usage: Inference only (frozen weights)
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class DINOv3SemanticExtractor(nn.Module):
    """
    DINOv3 ViT-B/16 distilled semantic feature extractor.

    This class handles:
    - Loading the pretrained DINOv3 model
    - Preprocessing input images
    - Extracting semantic features
    - Post-processing features for downstream tasks
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "auto",
        freeze_weights: bool = True
    ):
        """
        Initialize the DINOv3 semantic extractor.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('auto', 'cpu', 'cuda')
            freeze_weights: Whether to freeze model parameters
        """
        super().__init__()

        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading DINOv3 model: {model_name}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except ValueError:
            # Fallback for models not yet supported by transformers
            logger.warning(f"AutoImageProcessor failed for {model_name}, using manual processing")
            self.processor = None
            # Manual image processing parameters for DINOv3
            self.image_size = 224
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        self.model = AutoModel.from_pretrained(model_name)

        # Move to device
        self.model.to(self.device)

        # Freeze parameters for inference
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("Model weights frozen for inference")

        # Model configuration
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Register tokens: {self.num_register_tokens}")
        logger.info(f"  - Device: {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess a single image for DINOv3.

        Args:
            image: Input image (PIL Image, numpy array, or file path)

        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image, numpy array, or file path")

        # Process with DINOv3 processor or manual processing
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        else:
            # Manual preprocessing
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std)
            ])
            if isinstance(image, Image.Image):
                tensor = transform(image).unsqueeze(0)
            else:
                # Convert numpy array to PIL first
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                tensor = transform(image).unsqueeze(0)
            return tensor.to(self.device)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic features from preprocessed pixel values.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W]

        Returns:
            Semantic features [B, num_patches, hidden_size]
        """
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            last_hidden_states = outputs.last_hidden_state

        # Extract patch features (exclude CLS token and register tokens)
        batch_size = pixel_values.shape[0]
        _, img_height, img_width = pixel_values.shape[-3:]

        num_patches_height = img_height // self.patch_size
        num_patches_width = img_width // self.patch_size
        num_patches_flat = num_patches_height * num_patches_width

        # Remove CLS token and register tokens
        patch_features_flat = last_hidden_states[:, 1 + self.num_register_tokens:, :]

        # Reshape to spatial grid
        patch_features = patch_features_flat.unflatten(
            1, (num_patches_height, num_patches_width)
        )

        return patch_features

    def forward(self, images: Union[List[Union[Image.Image, np.ndarray, str]], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the semantic extractor.

        Args:
            images: List of images or preprocessed tensor

        Returns:
            Semantic features [B, H, W, hidden_size] or [B, num_patches, hidden_size]
        """
        if isinstance(images, torch.Tensor):
            # Already preprocessed
            return self.extract_features(images)
        else:
            # Preprocess images
            if not isinstance(images, list):
                images = [images]

            pixel_values_list = []
            for img in images:
                pixel_values = self.preprocess_image(img)
                pixel_values_list.append(pixel_values)

            pixel_values_batch = torch.cat(pixel_values_list, dim=0)
            return self.extract_features(pixel_values_batch)

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        return self.hidden_size

    def get_patch_info(self, image_size: tuple) -> dict:
        """
        Get patch information for a given image size.

        Args:
            image_size: (height, width) of input image

        Returns:
            Dictionary with patch information
        """
        height, width = image_size
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        num_patches_total = num_patches_height * num_patches_width

        return {
            'patch_size': self.patch_size,
            'num_patches_height': num_patches_height,
            'num_patches_width': num_patches_width,
            'num_patches_total': num_patches_total,
            'feature_dim': self.hidden_size
        }


def create_semantic_extractor(
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    device: str = "auto",
    freeze_weights: bool = True
) -> DINOv3SemanticExtractor:
    """
    Factory function to create a DINOv3 semantic extractor.

    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on
        freeze_weights: Whether to freeze model parameters

    Returns:
        Configured DINOv3SemanticExtractor instance
    """
    return DINOv3SemanticExtractor(
        model_name=model_name,
        device=device,
        freeze_weights=freeze_weights
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the semantic extractor
    extractor = create_semantic_extractor()

    # Test with a sample image
    from transformers.image_utils import load_image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)

    print(f"Original image size: {image.size}")

    # Extract features
    features = extractor([image])
    print(f"Extracted features shape: {features.shape}")

    # Get patch information
    patch_info = extractor.get_patch_info(image.size)
    print(f"Patch information: {patch_info}")
