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
    
    @staticmethod
    def estimate_aqi(visibility_score, haze_density, degradation_scores):
        """
        Estimate Air Quality Index (AQI) based on visibility and pollution indicators.
        
        AQI ranges:
        - 0-50: Good (Green)
        - 51-100: Moderate (Yellow)
        - 101-150: Unhealthy for Sensitive Groups (Orange)
        - 151-200: Unhealthy (Red)
        - 201-300: Very Unhealthy (Purple)
        - 301-500: Hazardous (Maroon)
        
        Args:
            visibility_score: Visibility score (0-100, higher = better)
            haze_density: Average haze density (0-1, higher = more haze)
            degradation_scores: Dict of degradation scores (0-1, higher = worse)
        
        Returns:
            dict: AQI information including value, category, color, and health message
        """
        # Base AQI calculation: inverse relationship with visibility
        # Lower visibility = higher AQI
        base_aqi = 500 * (1.0 - visibility_score / 100.0)
        
        # Adjust based on haze density (strong indicator of pollution)
        haze_factor = haze_density * 150  # Can add up to 150 points
        
        # Adjust based on overall degradation
        avg_degradation = sum(degradation_scores.values()) / len(degradation_scores)
        degradation_factor = avg_degradation * 100  # Can add up to 100 points
        
        # Combine factors (weighted)
        estimated_aqi = base_aqi * 0.5 + haze_factor * 0.3 + degradation_factor * 0.2
        
        # Clamp to valid AQI range (0-500)
        estimated_aqi = max(0, min(500, estimated_aqi))
        estimated_aqi = int(round(estimated_aqi))
        
        # Determine category and color
        if estimated_aqi <= 50:
            category = "Good"
            color = "#00E400"  # Green
            health_message = "Air quality is satisfactory, and air pollution poses little or no risk."
        elif estimated_aqi <= 100:
            category = "Moderate"
            color = "#FFFF00"  # Yellow
            health_message = "Air quality is acceptable. Sensitive individuals may experience minor breathing discomfort."
        elif estimated_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "#FF7E00"  # Orange
            health_message = "Members of sensitive groups may experience health effects. General public is less likely to be affected."
        elif estimated_aqi <= 200:
            category = "Unhealthy"
            color = "#FF0000"  # Red
            health_message = "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."
        elif estimated_aqi <= 300:
            category = "Very Unhealthy"
            color = "#8F3F97"  # Purple
            health_message = "Health alert: everyone may experience more serious health effects."
        else:
            category = "Hazardous"
            color = "#7E0023"  # Maroon
            health_message = "Health warning of emergency conditions. Entire population is likely to be affected."
        
        return {
            'aqi': estimated_aqi,
            'category': category,
            'color': color,
            'health_message': health_message,
            'visibility_score': visibility_score,
            'haze_density': haze_density
        }
    
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
        
        # Estimate AQI based on visibility and pollution indicators
        aqi_info = VisibilityAnalyzer.estimate_aqi(
            visibility_score,
            haze_results['average_haze_density'],
            degradation_scores
        )
        
        return {
            'visibility_score': visibility_score,
            'feature_scores': feature_scores,
            'degradation_scores': degradation_scores,
            'feature_maps': feature_maps,
            'aqi': aqi_info,
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
