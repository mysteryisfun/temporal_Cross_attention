import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from src.models.cross_attention import CrossAttention

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

def test_cross_attention():
    # Define dimensions
    batch_size = 8
    static_dim = 512
    dynamic_dim = 256
    fusion_dim = 128
    num_heads = 4

    # Create random input tensors
    static_features = tf.convert_to_tensor(np.random.rand(batch_size, static_dim), dtype=tf.float32)
    dynamic_features = tf.convert_to_tensor(np.random.rand(batch_size, dynamic_dim), dtype=tf.float32)

    # Initialize CrossAttention module
    cross_attention = CrossAttention(static_dim, dynamic_dim, fusion_dim, num_heads)

    # Pass inputs through the module
    fused_features = cross_attention(static_features, dynamic_features)

    # Print output shape
    print("Output shape:", fused_features.shape)

    # Check if the output shape is correct
    assert fused_features.shape == (batch_size, fusion_dim), "Output shape mismatch!"
    print("CrossAttention module test passed.")

if __name__ == "__main__":
    test_cross_attention()
