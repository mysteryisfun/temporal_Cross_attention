"""
I3D Motion Feature Extractor Implementation
==========================================

This module implements the motion feature extraction using I3D (Inflated 3D ConvNet).
The model is used for extracting temporal motion features from video sequences.

Model: I3D with ResNet-50 backbone pretrained on Kinetics-400
Output: 2048-dimensional motion features
Usage: Inference only (frozen weights)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import numpy as np
from typing import List, Union, Optional
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class I3DMotionExtractor(nn.Module):
    """
    I3D motion feature extractor for temporal modeling.

    This class handles:
    - Loading pretrained I3D/R3D model
    - Processing video sequences (16 frames)
    - Extracting motion features
    - Temporal feature aggregation
    """

    def __init__(
        self,
        model_type: str = "r3d_18",
        device: str = "auto",
        freeze_weights: bool = True,
        num_frames: int = 16
    ):
        """
        Initialize the I3D motion extractor.

        Args:
            model_type: Type of 3D CNN model ('r3d_18', 'mc3_18', 's3d')
            device: Device to run model on ('auto', 'cpu', 'cuda')
            freeze_weights: Whether to freeze model parameters
            num_frames: Number of frames to process
        """
        super().__init__()

        self.model_type = model_type
        self.device = self._get_device(device)
        self.num_frames = num_frames
        self.input_size = (224, 224)

        logger.info(f"Loading I3D model: {model_type}")
        
        # Load pretrained model
        if model_type == "r3d_18":
            from torchvision.models.video import r3d_18
            self.model = r3d_18(pretrained=True)
            self.feature_dim = 512  # R3D-18 feature dimension
        elif model_type == "mc3_18":
            from torchvision.models.video import mc3_18
            self.model = mc3_18(pretrained=True)
            self.feature_dim = 512  # MC3-18 feature dimension
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Remove classification head to get features
        self.model.fc = nn.Identity()

        # Move to device
        self.model.to(self.device)

        # Freeze parameters for inference
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info("Model weights frozen for inference")

        # Video preprocessing
        # R3D models expect normalized input per channel across the entire video
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Model type: {model_type}")
        logger.info(f"  - Feature dimension: {self.feature_dim}")
        logger.info(f"  - Number of frames: {num_frames}")
        logger.info(f"  - Input size: {self.input_size}")
        logger.info(f"  - Device: {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video frames.

        Args:
            video_path: Path to video file

        Returns:
            Preprocessed video tensor [1, 3, T, H, W]
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame {frame_idx}")
                continue
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for transforms
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        
        cap.release()
        
        if len(frames) != self.num_frames:
            logger.warning(f"Expected {self.num_frames} frames, got {len(frames)}")
            # Pad with last frame if needed
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        
        return frames

    def preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess frames for I3D model.

        Args:
            frames: List of PIL Images

        Returns:
            Preprocessed tensor [1, 3, T, H, W]
        """
        # Convert frames to tensors and resize
        frame_tensors = []
        for frame in frames:
            # Resize and convert to tensor
            frame_resized = frame.resize(self.input_size)
            frame_tensor = transforms.ToTensor()(frame_resized)
            frame_tensors.append(frame_tensor)
        
        # Stack frames: [T, 3, H, W]
        video_tensor = torch.stack(frame_tensors, dim=0)
        
        # Rearrange to [3, T, H, W] for 3D CNN
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # Add batch dimension: [1, 3, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        
        # Move to device first
        video_tensor = video_tensor.to(self.device)
        
        # Manual normalization for video data
        mean = self.mean.to(self.device)
        std = self.std.to(self.device)
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor

    def extract_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from video tensor.

        Args:
            video_tensor: Preprocessed video tensor [B, 3, T, H, W]

        Returns:
            Motion features [B, feature_dim]
        """
        with torch.inference_mode():
            features = self.model(video_tensor)
        
        return features

    def forward(self, video_input: Union[str, torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Forward pass through the motion extractor.

        Args:
            video_input: Video path, preprocessed tensor, or list of frames

        Returns:
            Motion features [B, feature_dim]
        """
        if isinstance(video_input, str):
            # Load from video path
            frames = self.load_video_frames(video_input)
            video_tensor = self.preprocess_frames(frames)
            return self.extract_features(video_tensor), frames
        elif isinstance(video_input, list):
            # List of PIL Images
            video_tensor = self.preprocess_frames(video_input)
            return self.extract_features(video_tensor), video_input
        elif isinstance(video_input, torch.Tensor):
            # Already preprocessed
            return self.extract_features(video_input), None
        else:
            raise ValueError("Input must be video path, tensor, or list of PIL Images")

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        return self.feature_dim


def create_motion_extractor(
    model_type: str = "r3d_18",
    device: str = "auto",
    freeze_weights: bool = True,
    num_frames: int = 16
) -> I3DMotionExtractor:
    """
    Factory function to create an I3D motion extractor.

    Args:
        model_type: Type of 3D CNN model
        device: Device to run model on
        freeze_weights: Whether to freeze model parameters
        num_frames: Number of frames to process

    Returns:
        Configured I3DMotionExtractor instance
    """
    return I3DMotionExtractor(
        model_type=model_type,
        device=device,
        freeze_weights=freeze_weights,
        num_frames=num_frames
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the motion extractor
    extractor = create_motion_extractor()

    print(f"Motion extractor created successfully")
    print(f"Model type: {extractor.model_type}")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"Number of frames: {extractor.num_frames}")
    print(f"Device: {extractor.device}")
