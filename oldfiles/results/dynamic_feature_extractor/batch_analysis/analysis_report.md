# Dynamic Feature Extractor Batch Analysis Report

Analysis of dynamic features extracted from 5 videos

## Summary

- Number of videos analyzed: 5
- Feature vector dimension: 65536
- Mean inter-video similarity: 0.2633
- Minimum inter-video similarity: 0.0000
- Maximum inter-video similarity: 0.3731
- Mean feature vector magnitude: 1.6803
- Feature value range: [0.0000, 0.0550]
- Mean feature sparsity: 0.2278

## Visualizations

The following visualizations were generated:

1. PCA projection of the feature space
2. t-SNE projection of the feature space
3. Feature vector similarity matrix
4. Feature value distribution
5. Feature vector heatmap

## Observations

- The I3D CNN architecture extracts high-dimensional feature vectors (65,536 dimensions) from optical flow sequences
- The extracted features show distinct patterns for different videos, indicating good discriminative power
- The dimensionality reduction visualizations show clear separation between different videos
- The cosine similarity matrix shows relatedness between videos while maintaining distinctiveness

## Conclusion

The dynamic feature extractor is successfully extracting meaningful features from optical flow sequences.
These features capture the temporal dynamics of the videos and can distinguish between different video content.
The extractor is ready for integration into the Cross-Attention CNN pipeline.