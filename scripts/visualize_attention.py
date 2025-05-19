"""
Attention Visualization Tool for CrossAttention Module

This script provides functions to extract and visualize attention weights from the CrossAttention module.
- Plots attention maps for a given input batch
- Saves attention visualizations for analysis

Usage:
    # Activate virtual environment
    if (Test-Path .\env\Scripts\activate.ps1) {
        .\env\Scripts\activate.ps1
        python .\scripts\visualize_attention.py
    }
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.models.cross_attention import CrossAttention

def get_attention_weights(cross_attention_layer, static_features, dynamic_features):
    """
    Returns the attention weights from the MultiHeadAttention layer for given inputs.
    """
    # Project features
    static_proj = cross_attention_layer.static_proj(static_features)
    dynamic_proj = cross_attention_layer.dynamic_proj(dynamic_features)
    static_proj = tf.expand_dims(static_proj, axis=1)
    dynamic_proj = tf.expand_dims(dynamic_proj, axis=1)
    # Get attention output and weights
    _, attn_weights = cross_attention_layer.attention(
        query=static_proj, value=dynamic_proj, key=dynamic_proj, return_attention_scores=True
    )
    return attn_weights.numpy()

def plot_attention_map(attn_weights, save_path=None):
    """
    Plots and optionally saves the attention map for a batch.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_weights[0, 0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Cross-Attention Map (Head 0, Sample 0)')
    plt.xlabel('Key/Value Sequence')
    plt.ylabel('Query Sequence')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Example usage with random data
    batch_size = 4
    static_dim = 512
    dynamic_dim = 256
    fusion_dim = 128
    num_heads = 4
    static_features = tf.convert_to_tensor(np.random.rand(batch_size, static_dim), dtype=tf.float32)
    dynamic_features = tf.convert_to_tensor(np.random.rand(batch_size, dynamic_dim), dtype=tf.float32)
    cross_attention = CrossAttention(static_dim, dynamic_dim, fusion_dim, num_heads)
    attn_weights = get_attention_weights(cross_attention, static_features, dynamic_features)
    print('Attention weights shape:', attn_weights.shape)
    plot_attention_map(attn_weights)

if __name__ == "__main__":
    main()
