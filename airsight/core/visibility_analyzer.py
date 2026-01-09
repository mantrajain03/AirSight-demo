"""
Main Visibility Analyzer

Combines all feature extractors to compute a unified visibility score.
"""

import numpy as np
import cv2
from .feature_extractors import (
    ContrastAnalyzer,
    EdgeAnalyzer,
    ColorAnalyzer,
    DarkChannelPrior,
    StructuralAnalyzer
)


class VisibilityAnalyzer:
    """
    Main analyzer that combines all visibility degradation features.
    
    Computes a unified visibility score (0-100) from:
    - Contrast loss
    - Edge attenuation
    - Color distortion
    - Haze density
    - Structural decay
    """
    
    def __init__(self, weights=None):
        """
        Initialize analyzer with optional feature weights.
        
        Args:
            weights: dict with keys: contrast, edge, color, haze, structure
                    If None, uses default balanced weights
        """
        if weights is None:
            self.weights = {
                'contrast': 0.20,
                'edge': 0.25,
                'color': 0.15,
                'haze': 0.25,
                'structure': 0.15
            }
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def analyze(self, image):
        """
        Complete visibility analysis pipeline.
        
        Args:
            image: numpy array (BGR format from OpenCV)
        
        Returns:
            dict: Complete analysis results including:
                - visibility_score: 0-100 (higher = better visibility)
                - feature_scores: Individual feature scores
                - feature_maps: Visual maps for each feature
                - raw_features: Raw feature extraction results
        """
        # Extract all features
        contrast_results = ContrastAnalyzer.analyze(image)
        edge_results = EdgeAnalyzer.analyze(image)
        color_results = ColorAnalyzer.analyze(image)
        haze_results = DarkChannelPrior.analyze(image)
        structure_results = StructuralAnalyzer.analyze(image)
        
        # Collect degradation scores (all 0-1, higher = worse)
        degradation_scores = {
            'contrast': contrast_results['contrast_loss_index'],
            'edge': 1.0 - edge_results['edge_strength_score'],  # Convert strength to loss
            'color': color_results['color_degradation_score'],
            'haze': haze_results['average_haze_density'],
            'structure': structure_results['structural_decay_score']
        }
        
        # Compute weighted degradation score
        weighted_degradation = sum(
            degradation_scores[k] * self.weights[k] 
            for k in self.weights.keys()
        )
        
        # Convert to visibility score (0-100, higher = better)
        visibility_score = (1.0 - weighted_degradation) * 100.0
        visibility_score = max(0.0, min(100.0, visibility_score))  # Clamp to 0-100
        
        # Collect feature maps for visualization
        feature_maps = {
            'haze_density': haze_results['haze_density_map'],
            'edge_magnitude': edge_results['sobel_magnitude'],
            'edge_canny': edge_results['canny_edges'],
            'texture_variance': structure_results['texture_variance_map']
        }
        
        # Collect individual feature scores (0-100, higher = better)
        feature_scores = {
            'contrast': (1.0 - degradation_scores['contrast']) * 100.0,
            'edge': (1.0 - degradation_scores['edge']) * 100.0,
            'color': (1.0 - degradation_scores['color']) * 100.0,
            'haze': (1.0 - degradation_scores['haze']) * 100.0,
            'structure': (1.0 - degradation_scores['structure']) * 100.0
        }
        
        return {
            'visibility_score': visibility_score,
            'feature_scores': feature_scores,
            'degradation_scores': degradation_scores,
            'feature_maps': feature_maps,
            'raw_features': {
                'contrast': contrast_results,
                'edge': edge_results,
                'color': color_results,
                'haze': haze_results,
                'structure': structure_results
            },
            'weights': self.weights
        }
    
    def analyze_batch(self, images):
        """
        Analyze multiple images.
        
        Args:
            images: list of numpy arrays
        
        Returns:
            list: Analysis results for each image
        """
        return [self.analyze(img) for img in images]
