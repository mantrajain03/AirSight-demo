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
    def extract_hsv_features(image):
        """
        Extract HSV histogram features for AQI estimation.
        
        Returns:
            dict: HSV histogram statistics
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Compute histograms
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h / (hist_h.sum() + 1e-10)
        hist_s = hist_s / (hist_s.sum() + 1e-10)
        hist_v = hist_v / (hist_v.sum() + 1e-10)
        
        # Extract features: mean, std, entropy for each channel
        features = {}
        for name, hist in [('hue', hist_h), ('saturation', hist_s), ('value', hist_v)]:
            features[f'{name}_mean'] = np.mean(hist)
            features[f'{name}_std'] = np.std(hist)
            features[f'{name}_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        # Value channel brightness (indicator of haze/fog)
        features['brightness'] = np.mean(v) / 255.0
        
        return features
    
    @staticmethod
    def detect_image_quality_issues(image, raw_features):
        """
        Detect image quality issues that affect AQI estimation reliability.
        
        Returns:
            dict: Reliability flags and warnings
        """
        issues = {
            'is_night': False,
            'is_fog': False,
            'is_rain': False,
            'is_indoor': False,
            'is_distorted': False,
            'reliability_score': 1.0,
            'warnings': []
        }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Night detection: low average brightness
        avg_brightness = np.mean(gray) / 255.0
        if avg_brightness < 0.2:
            issues['is_night'] = True
            issues['reliability_score'] *= 0.3
            issues['warnings'].append("Night image detected - AQI estimation may be unreliable")
        
        # Fog detection: high haze density + low contrast + high brightness
        haze_density = raw_features.get('haze', {}).get('average_haze_density', 0)
        contrast_loss = raw_features.get('contrast', {}).get('contrast_loss_index', 0)
        if haze_density > 0.4 and contrast_loss > 0.5 and avg_brightness > 0.6:
            issues['is_fog'] = True
            issues['reliability_score'] *= 0.7
            issues['warnings'].append("Fog detected - may confuse AQI estimation")
        
        # Rain detection: high edge density + high saturation variance
        edge_density = raw_features.get('edge', {}).get('edge_density', 0)
        if edge_density > 0.3:
            # Check for rain-like patterns (many small edges)
            issues['reliability_score'] *= 0.9
        
        # Indoor detection: low dynamic range + balanced colors
        dynamic_range = raw_features.get('contrast', {}).get('dynamic_range', 1.0)
        color_balance = raw_features.get('color', {}).get('channel_balance', 0)
        if dynamic_range < 0.3 and color_balance > 0.9:
            issues['is_indoor'] = True
            issues['reliability_score'] *= 0.5
            issues['warnings'].append("Indoor image detected - AQI estimation not applicable")
        
        # Distorted image: very low entropy or very high degradation
        entropy = raw_features.get('contrast', {}).get('entropy', 8.0)
        if entropy < 2.0:
            issues['is_distorted'] = True
            issues['reliability_score'] *= 0.4
            issues['warnings'].append("Image appears distorted - low reliability")
        
        issues['reliability_score'] = max(0.0, min(1.0, issues['reliability_score']))
        
        return issues
    
    @staticmethod
    def estimate_aqi(image, visibility_score, haze_density, degradation_scores, raw_features):
        """
        Enhanced AQI estimation using comprehensive feature engineering.
        
        Based on project prompt requirements:
        - Uses baseline features: contrast, edge density, HSV histograms, dark channel prior, visibility index
        - Outputs: Estimated AQI, confidence score, reliability flag
        
        AQI ranges:
        - 0-50: Good (Green)
        - 51-100: Moderate (Yellow)
        - 101-150: Unhealthy for Sensitive Groups (Orange)
        - 151-200: Unhealthy (Red)
        - 201-300: Very Unhealthy (Purple)
        - 301-500: Hazardous (Maroon)
        
        Args:
            image: Original image (BGR format)
            visibility_score: Visibility score (0-100, higher = better)
            haze_density: Average haze density (0-1, higher = more haze)
            degradation_scores: Dict of degradation scores (0-1, higher = worse)
            raw_features: Dict containing all raw feature extraction results
        
        Returns:
            dict: AQI information including value, category, color, confidence, and reliability
        """
        # Extract HSV histogram features
        hsv_features = VisibilityAnalyzer.extract_hsv_features(image)
        
        # Detect image quality issues
        quality_issues = VisibilityAnalyzer.detect_image_quality_issues(image, raw_features)
        
        # Feature engineering for AQI regression
        # 1. Visibility Index (inverse relationship with AQI)
        visibility_index = visibility_score / 100.0  # 0-1, higher = better
        
        # 2. Dark Channel Prior (haze density) - strong pollution indicator
        dcp_score = haze_density  # Already 0-1
        
        # 3. Contrast features
        contrast_loss = degradation_scores.get('contrast', 0.5)
        contrast_entropy = raw_features.get('contrast', {}).get('entropy', 4.0) / 8.0  # Normalize
        
        # 4. Edge density
        edge_loss = degradation_scores.get('edge', 0.5)
        edge_density = raw_features.get('edge', {}).get('edge_density', 0.1)
        
        # 5. HSV features (saturation and value are indicators of haze)
        saturation_mean = hsv_features.get('saturation_mean', 0.5)
        value_mean = hsv_features.get('value_mean', 0.5)
        brightness = hsv_features.get('brightness', 0.5)
        
        # 6. Color degradation
        color_degradation = degradation_scores.get('color', 0.3)
        
        # 7. Structural decay
        structure_decay = degradation_scores.get('structure', 0.4)
        
        # Multi-feature AQI estimation (regression-like approach)
        # Base AQI from visibility (inverse relationship with non-linear scaling)
        # Use exponential scaling for better sensitivity at high pollution levels
        visibility_factor = 1.0 - visibility_index
        base_aqi = 500 * (visibility_factor ** 1.2)  # Non-linear scaling
        
        # Dark Channel Prior contribution (strongest indicator)
        # Enhanced with saturation consideration (high saturation + high DCP = pollution)
        saturation_factor = 1.0 + (1.0 - saturation_mean) * 0.3  # Low saturation = more haze
        dcp_contribution = dcp_score * 200 * saturation_factor
        
        # Contrast loss contribution (enhanced)
        # Combine contrast loss with entropy for better accuracy
        contrast_factor = contrast_loss * (1.0 + (1.0 - contrast_entropy) * 0.5)
        contrast_contribution = contrast_factor * 100
        
        # Edge density contribution (low edge density = more pollution)
        # Enhanced with edge strength consideration
        edge_strength = raw_features.get('edge', {}).get('edge_strength_score', 0.5)
        edge_factor = (1.0 - edge_density) * (1.0 + (1.0 - edge_strength) * 0.3)
        edge_contribution = edge_factor * 80
        
        # HSV brightness contribution (high brightness with low contrast = haze)
        # Enhanced brightness factor with value channel analysis
        value_factor = value_mean / 255.0 if value_mean > 0 else 0.5
        brightness_factor = brightness * (1.0 - contrast_entropy) * value_factor * 60
        
        # Color degradation contribution (enhanced)
        # Color shift toward blue/gray indicates pollution
        channel_means = raw_features.get('color', {}).get('channel_means', {})
        if channel_means:
            b_mean = channel_means.get('b', 128) / 255.0
            r_mean = channel_means.get('r', 128) / 255.0
            # Blue shift (b > r) indicates haze
            blue_shift = max(0, (b_mean - r_mean)) * 0.5
            color_factor = color_degradation * (1.0 + blue_shift)
        else:
            color_factor = color_degradation
        color_contribution = color_factor * 70
        
        # Structural decay contribution (enhanced)
        # Combine with texture variance for better accuracy
        texture_var = raw_features.get('structure', {}).get('texture_variance', 500)
        texture_factor = min(texture_var / 1000.0, 1.0)  # Normalize
        structure_factor = structure_decay * (1.0 + (1.0 - texture_factor) * 0.4)
        structure_contribution = structure_factor * 50
        
        # Weighted combination (emphasizing DCP and visibility)
        estimated_aqi = (
            base_aqi * 0.28 +           # Visibility index: 28% (increased)
            dcp_contribution * 0.32 +   # Dark Channel Prior: 32% (strongest, increased)
            contrast_contribution * 0.16 +  # Contrast: 16% (increased)
            edge_contribution * 0.10 +   # Edge density: 10%
            brightness_factor * 0.07 +   # Brightness factor: 7% (decreased)
            color_contribution * 0.05 +  # Color: 5% (decreased)
            structure_contribution * 0.02  # Structure: 2% (decreased)
        )
        
        # Clamp to valid AQI range (0-500)
        estimated_aqi = max(0, min(500, estimated_aqi))
        estimated_aqi = int(round(estimated_aqi))
        
        # Calculate confidence score based on feature consistency
        # Higher confidence when features agree, lower when they disagree
        feature_scores = [
            1.0 - contrast_loss,
            1.0 - edge_loss,
            1.0 - color_degradation,
            1.0 - dcp_score,
            1.0 - structure_decay
        ]
        feature_std = np.std(feature_scores)
        feature_mean = np.mean(feature_scores)
        
        # Confidence based on consistency (lower std = higher confidence)
        consistency_score = max(0.3, 1.0 - feature_std * 1.5)
        
        # Confidence based on feature quality (higher mean = more reliable)
        quality_score = feature_mean
        
        # Combined confidence (weighted average)
        confidence_score = (consistency_score * 0.6 + quality_score * 0.4)
        
        # Adjust confidence based on reliability issues
        confidence_score *= quality_issues['reliability_score']
        confidence_score = max(0.0, min(1.0, confidence_score))
        
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
        
        # Reliability flag
        is_reliable = quality_issues['reliability_score'] > 0.6 and not quality_issues['is_night'] and not quality_issues['is_indoor']
        
        return {
            'aqi': estimated_aqi,
            'category': category,
            'color': color,
            'health_message': health_message,
            'confidence_score': confidence_score,
            'reliability_flag': is_reliable,
            'reliability_score': quality_issues['reliability_score'],
            'quality_issues': quality_issues,
            'visibility_score': visibility_score,
            'haze_density': haze_density,
            'feature_contributions': {
                'visibility': base_aqi * 0.25,
                'dark_channel_prior': dcp_contribution * 0.30,
                'contrast': contrast_contribution * 0.15,
                'edge_density': edge_contribution * 0.10,
                'brightness': brightness_factor * 0.08,
                'color': color_contribution * 0.07,
                'structure': structure_contribution * 0.05
            }
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
        
        # Collect raw features for AQI estimation
        raw_features_dict = {
            'contrast': contrast_results,
            'edge': edge_results,
            'color': color_results,
            'haze': haze_results,
            'structure': structure_results
        }
        
        # Estimate AQI based on visibility and pollution indicators
        aqi_info = VisibilityAnalyzer.estimate_aqi(
            image,
            visibility_score,
            haze_results['average_haze_density'],
            degradation_scores,
            raw_features_dict
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
