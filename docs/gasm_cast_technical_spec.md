# GASM-CAST Technical Implementation Specification

## Architecture Overview

**GASM-CAST** (Graph-Aware Semantic-Motion Cross-Attention with Temporal Structure) is a sophisticated fusion architecture that goes beyond simple cross-attention to achieve:
- 🧠 **Intra-modal understanding** through graph attention networks
- 🔄 **Efficient cross-modal fusion** through bottleneck attention (B-CAST)
- 📈 **Hierarchical learning** through progressive 3-layer architecture

## Implementation Roadmap (Week 5-8)

### Week 5: Metadata Generation & Graph Construction

#### 5.1 Spatial Position Generation
**File**: `src/utils/metadata_generator.py`
```python
def generate_spatial_positions():
    """Generate 14x14 spatial positions for DINOv3 patches"""
    # Create grid coordinates for 196 patches
    # Output: [196, 2] tensor with (x, y) coordinates
```

#### 5.2 Temporal Position Generation  
```python
def generate_temporal_positions():
    """Generate temporal positions for 8 I3D segments from 16 frames"""
    # Map 16 frames to 8 segments with temporal indexing
    # Output: [8, 1] tensor with temporal indices
```

#### 5.3 Graph Construction Utilities
```python
def build_spatial_graph(positions, similarity_threshold=0.8):
    """Build adaptive graph for semantic patches"""
    # Connect spatially adjacent + semantically similar patches
    
def build_temporal_graph(positions, adjacency_window=2):
    """Build adaptive graph for motion segments"""
    # Connect temporally adjacent + motion-similar segments
```

### Week 5-6: Intra-Modal Graph Attention

#### 6.1 Semantic Graph Attention Module
**File**: `src/models/graph_attention/semantic_graph_attention.py`
```python
class SemanticGraphAttention(nn.Module):
    def __init__(self, input_dim=768, num_heads=8):
        self.graph_attention = GraphAttentionLayer(input_dim, num_heads)
        
    def forward(self, semantic_features, spatial_positions):
        # Input: [B, 196, 768] semantic features
        # Build adaptive spatial graph based on positions + similarities
        # Apply graph attention to refine patch representations
        # Output: [B, 196, 768] context-aware semantic features
```

#### 6.2 Motion Graph Attention Module
**File**: `src/models/graph_attention/motion_graph_attention.py`
```python
class MotionGraphAttention(nn.Module):
    def __init__(self, input_dim=512, num_heads=8):
        self.graph_attention = GraphAttentionLayer(input_dim, num_heads)
        
    def forward(self, motion_features, temporal_positions):
        # Input: [B, 8, 512] motion features
        # Build adaptive temporal graph based on positions + similarities
        # Apply graph attention to refine segment representations
        # Output: [B, 8, 512] context-aware motion features
```

### Week 6-7: Progressive B-CAST Implementation

#### 7.1 Bottleneck Cross-Attention Layer
**File**: `src/models/cross_attention/bottleneck_attention.py`
```python
class BottleneckCrossAttention(nn.Module):
    def __init__(self, sem_dim=768, mot_dim=512, bottleneck_dim=256):
        # Compress both modalities to bottleneck space
        self.sem_compress = nn.Linear(sem_dim, bottleneck_dim)
        self.mot_compress = nn.Linear(mot_dim, bottleneck_dim)
        
        # Cross-attention in compressed space
        self.cross_attention = MultiHeadAttention(bottleneck_dim, num_heads=8)
        
        # Residual integration
        self.sem_expand = nn.Linear(bottleneck_dim, sem_dim)
        self.mot_expand = nn.Linear(bottleneck_dim, mot_dim)
        
    def forward(self, semantic_features, motion_features):
        # Compress: [B, 196, 768] → [B, 196, 256], [B, 8, 512] → [B, 8, 256]
        # Cross-attend in 256D space (efficient)
        # Expand back and add residual connections
        # Return: Enhanced semantic + motion features
```

#### 7.2 Progressive 3-Layer Architecture
**File**: `src/models/cross_attention/gasm_cast.py`
```python
class GASMCAST(nn.Module):
    def __init__(self):
        # Graph attention modules
        self.semantic_graph = SemanticGraphAttention()
        self.motion_graph = MotionGraphAttention()
        
        # Progressive B-CAST layers
        self.layer1_basic = BottleneckCrossAttention()      # Basic correspondences
        self.layer2_pattern = BottleneckCrossAttention()    # Pattern correspondences  
        self.layer3_action = BottleneckCrossAttention()     # Action correspondences
        
    def forward(self, raw_semantic, raw_motion, spatial_pos, temporal_pos):
        # Step 1: Intra-modal graph refinement
        semantic = self.semantic_graph(raw_semantic, spatial_pos)
        motion = self.motion_graph(raw_motion, temporal_pos)
        
        # Step 2: Progressive cross-modal fusion
        sem_1, mot_1 = self.layer1_basic(semantic, motion)      # Hand patches ↔ Contact motions
        sem_2, mot_2 = self.layer2_pattern(sem_1, mot_1)        # Object shapes ↔ Trajectories
        sem_3, mot_3 = self.layer3_action(sem_2, mot_2)         # Context + Motion → Action
        
        return sem_3, mot_3
```

### Week 7-8: Integration & Analysis

#### 8.1 Complete Model Integration
**File**: `src/models/complete_model.py`
```python
class TemporalCrossAttentionModel(nn.Module):
    def __init__(self, num_classes=174):
        self.semantic_extractor = DINOv3SemanticExtractor()
        self.motion_extractor = I3DMotionExtractor()
        self.gasm_cast = GASMCAST()
        self.classifier = nn.Linear(768 + 512, num_classes)
        
    def forward(self, video_frames):
        # Extract features
        semantic = self.semantic_extractor(video_frames)  # [B, 196, 768]
        motion = self.motion_extractor(video_frames)      # [B, 8, 512]
        
        # Generate metadata (universal)
        spatial_pos = generate_spatial_positions()
        temporal_pos = generate_temporal_positions()
        
        # GASM-CAST fusion
        fused_semantic, fused_motion = self.gasm_cast(
            semantic, motion, spatial_pos, temporal_pos
        )
        
        # Global pooling + classification
        semantic_global = fused_semantic.mean(dim=1)  # [B, 768]
        motion_global = fused_motion.mean(dim=1)      # [B, 512]
        combined = torch.cat([semantic_global, motion_global], dim=1)
        
        return self.classifier(combined)
```

#### 8.2 Attention Visualization Tools
**File**: `scripts/visualize_gasm_cast_attention.py`
```python
def visualize_graph_attention(model, video, save_path):
    """Visualize graph attention weights for interpretability"""
    # Show which patches/segments attend to each other
    
def visualize_cross_attention_layers(model, video, save_path):
    """Visualize progressive cross-attention learning"""
    # Layer 1: Basic correspondences
    # Layer 2: Pattern correspondences
    # Layer 3: Action correspondences
```

## Performance Targets & Validation

### Computational Efficiency
- **Target**: 50% reduction in cross-attention computation vs vanilla approach
- **Method**: Bottleneck compression (768D/512D → 256D)
- **Expected Memory**: 60-80% of vanilla cross-attention
- **Training Time**: 10-30% faster than full cross-attention

### Accuracy Targets
- **Something-Something-V2**: 76-80% Top-1 accuracy
- **Improvement**: +3-5% over simple cross-attention baseline
- **Parameters**: <20M trainable (graph attention + B-CAST + classifier)

### Research Novelty
- **First combination** of graph attention + bottleneck cross-attention for video action recognition
- **Progressive architecture** for hierarchical semantic-motion understanding
- **Efficiency gains** while maintaining/improving performance

## Testing & Validation Strategy

### Week 5 Testing: Metadata & Graph Construction
- [ ] Verify spatial position generation (14×14 = 196 patches)
- [ ] Validate temporal position mapping (16 frames → 8 segments)
- [ ] Test graph construction algorithms (spatial + temporal)

### Week 6 Testing: Graph Attention Modules
- [ ] Validate semantic graph attention improves patch representations
- [ ] Test motion graph attention enhances temporal understanding
- [ ] Benchmark computational overhead of graph operations

### Week 7 Testing: B-CAST Implementation
- [ ] Verify bottleneck compression maintains information
- [ ] Test progressive learning through 3 layers
- [ ] Validate residual connections prevent information loss

### Week 8 Testing: End-to-End Integration
- [ ] Complete pipeline functionality testing
- [ ] Performance benchmarking vs simple cross-attention
- [ ] Attention visualization and interpretability analysis
- [ ] Memory efficiency and training stability validation

## Expected Outcomes

### Technical Achievements
1. **Novel Architecture**: Graph-aware progressive cross-attention for video understanding
2. **Efficiency Gains**: 50% computational reduction with maintained/improved accuracy
3. **Interpretability**: Clear visualization of attention patterns at different levels
4. **Scalability**: Modular design adaptable to other multi-modal tasks

### Research Impact
1. **Publication Readiness**: Strong novelty for top-tier CV conference (CVPR, ICCV, ECCV)
2. **Community Contribution**: Open-source implementation for reproducibility
3. **Methodological Advance**: Progressive learning paradigm for multi-modal fusion
4. **Practical Application**: Efficient architecture suitable for resource-constrained deployment

---

This specification provides a clear roadmap for implementing the GASM-CAST architecture, moving significantly beyond simple cross-attention to create a sophisticated, efficient, and interpretable video action recognition system.
