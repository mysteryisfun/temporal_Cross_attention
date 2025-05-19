# filepath: c:\Users\ujwal\OneDrive\Documents\GitHub\Cross_Attention_CNN_Research_Execution\src\models\cross_attention.py.new
import tensorflow as tf
from keras.layers import Dense, LayerNormalization, MultiHeadAttention

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, static_dim, dynamic_dim, fusion_dim, num_heads):
        super(CrossAttention, self).__init__()
        
        # Linear projections to a common dimension
        self.static_proj = Dense(fusion_dim)
        self.dynamic_proj = Dense(fusion_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=fusion_dim)
        
        # Layer normalization
        self.norm = LayerNormalization(epsilon=1e-6)
        
    def call(self, static_features, dynamic_features, return_attention_weights=False):
        # Project features to a common dimension
        static_proj = self.static_proj(static_features)
        dynamic_proj = self.dynamic_proj(dynamic_features)

        # Reshape inputs to 3D (batch_size, 1, feature_dim) for MultiHeadAttention
        static_proj = tf.expand_dims(static_proj, axis=1)
        dynamic_proj = tf.expand_dims(dynamic_proj, axis=1)

        # Compute cross-attention
        if return_attention_weights:
            attn_output, attention_weights = self.attention(
                query=static_proj, 
                value=dynamic_proj, 
                key=dynamic_proj,
                return_attention_scores=True
            )
        else:
            attn_output = self.attention(query=static_proj, value=dynamic_proj, key=dynamic_proj)

        # Squeeze the output back to 2D (batch_size, feature_dim)
        attn_output = tf.squeeze(attn_output, axis=1)
        # Add residual connection and normalize
        fused_features = self.norm(attn_output + tf.squeeze(static_proj, axis=1))
        
        if return_attention_weights:
            return fused_features, attention_weights
        return fused_features
