"""
Feature Extraction Modules for AirSight

Implements classical image processing techniques to extract visibility degradation features:
1. Contrast Degradation Analysis
2. Edge Attenuation Measurement
3. Color Channel Distortion
4. Dark Channel Prior (Haze Density)
5. Structural Visibility Decay
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy


class ContrastAnalyzer:
    """Analyzes contrast degradation through histogram spread, entropy, and dynamic range."""
    
    @staticmethod
    def compute_histogram_spread(image):
        """
        Measures histogram spread as an indicator of contrast.
        
        Formula: std(histogram) / mean(histogram)
        Higher spread = better contrast
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist += 1e-10  # Avoid division by zero
        
        mean_hist = np.mean(hist)
        std_hist = np.std(hist)
        
        spread = std_hist / mean_hist if mean_hist > 0 else 0
        return spread
    
    @staticmethod
    def compute_entropy(image):
        """
        Computes image entropy as a measure of information content.
        
        Higher entropy = more information = better visibility
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-10)  # Normalize
        
        img_entropy = entropy(hist, base=2)
        return img_entropy
    
    @staticmethod
    def compute_dynamic_range(image):
        """
        Measures dynamic range compression.
        
        Formula: (max - min) / 255
        Higher value = better dynamic range
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        dynamic_range = (max_val - min_val) / 255.0
        return dynamic_range
    
    @staticmethod
    def analyze(image):
        """
        Comprehensive contrast degradation analysis.
        
        Returns:
            dict: Contains contrast_loss_index (0-1, higher = worse) and individual metrics
        """
        spread = ContrastAnalyzer.compute_histogram_spread(image)
        img_entropy = ContrastAnalyzer.compute_entropy(image)
        dynamic_range = ContrastAnalyzer.compute_dynamic_range(image)
        
        # Normalize metrics (higher values = better contrast)
        # Normalize spread (typical range: 0-5, normalize to 0-1)
        normalized_spread = min(spread / 5.0, 1.0)
        
        # Normalize entropy (max theoretical: 8 bits, normalize to 0-1)
        normalized_entropy = img_entropy / 8.0
        
        # Dynamic range is already 0-1
        
        # Combine metrics (weighted average)
        contrast_quality = (normalized_spread * 0.3 + 
                           normalized_entropy * 0.4 + 
                           dynamic_range * 0.3)
        
        # Convert to loss index (1 - quality)
        contrast_loss_index = 1.0 - contrast_quality
        
        return {
            'contrast_loss_index': contrast_loss_index,
            'histogram_spread': spread,
            'entropy': img_entropy,
            'dynamic_range': dynamic_range,
            'contrast_quality': contrast_quality
        }


class EdgeAnalyzer:
    """Measures edge attenuation using Sobel and Canny edge detection."""
    
    @staticmethod
    def compute_sobel_edges(image):
        """Computes Sobel edge magnitude."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return magnitude
    
    @staticmethod
    def compute_canny_edges(image, low_threshold=50, high_threshold=150):
        """Computes Canny edge map."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    @staticmethod
    def analyze(image):
        """
        Comprehensive edge attenuation analysis.
        
        Returns:
            dict: Contains edge_strength_score (0-1, higher = better) and edge maps
        """
        sobel_magnitude = EdgeAnalyzer.compute_sobel_edges(image)
        canny_edges = EdgeAnalyzer.compute_canny_edges(image)
        
        # Edge strength metrics
        mean_magnitude = np.mean(sobel_magnitude)
        max_magnitude = np.max(sobel_magnitude)
        edge_density = np.sum(canny_edges > 0) / (canny_edges.shape[0] * canny_edges.shape[1])
        
        # Normalize mean magnitude (typical range: 0-100, normalize to 0-1)
        normalized_mean_mag = min(mean_magnitude / 100.0, 1.0)
        
        # Normalize max magnitude (typical range: 0-500, normalize to 0-1)
        normalized_max_mag = min(max_magnitude / 500.0, 1.0)
        
        # Edge density is already 0-1
        
        # Combine metrics
        edge_strength_score = (normalized_mean_mag * 0.4 + 
                              normalized_max_mag * 0.3 + 
                              edge_density * 0.3)
        
        return {
            'edge_strength_score': edge_strength_score,
            'mean_magnitude': mean_magnitude,
            'max_magnitude': max_magnitude,
            'edge_density': edge_density,
            'sobel_magnitude': sobel_magnitude,
            'canny_edges': canny_edges
        }


class ColorAnalyzer:
    """Analyzes color channel distortion through RGB separation and variance analysis."""
    
    @staticmethod
    def analyze(image):
        """
        Comprehensive color channel distortion analysis.
        
        Returns:
            dict: Contains color_degradation_score (0-1, higher = worse) and channel metrics
        """
        if len(image.shape) != 3:
            # Grayscale image - no color distortion
            return {
                'color_degradation_score': 0.0,
                'channel_mean_shift': 0.0,
                'inter_channel_variance': 0.0,
                'channel_balance': 1.0
            }
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Channel means
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        
        # Channel mean shift (deviation from balanced state)
        # In a balanced image, channels should be similar
        channel_means = np.array([mean_b, mean_g, mean_r])
        mean_all = np.mean(channel_means)
        
        # Mean shift as coefficient of variation
        mean_shift = np.std(channel_means) / (mean_all + 1e-10)
        
        # Inter-channel variance (how different channels are from each other)
        inter_channel_var = np.var([mean_b, mean_g, mean_r])
        normalized_inter_var = min(inter_channel_var / (128**2), 1.0)  # Normalize
        
        # Channel balance (inverse of mean shift)
        channel_balance = 1.0 / (1.0 + mean_shift)
        
        # Color degradation score (higher = more degradation)
        # Haze typically causes color shift toward a dominant channel
        color_degradation_score = min(mean_shift * 0.5 + normalized_inter_var * 0.5, 1.0)
        
        return {
            'color_degradation_score': color_degradation_score,
            'channel_mean_shift': mean_shift,
            'inter_channel_variance': normalized_inter_var,
            'channel_balance': channel_balance,
            'channel_means': {'b': mean_b, 'g': mean_g, 'r': mean_r}
        }


class DarkChannelPrior:
    """Implements Dark Channel Prior for haze density estimation."""
    
    @staticmethod
    def compute_dark_channel(image, patch_size=15):
        """
        Computes dark channel prior map.
        
        Dark channel: min(min(I^c)) over local patch
        Haze regions have higher dark channel values.
        """
        if len(image.shape) == 3:
            # Get minimum across color channels
            min_channel = np.min(image, axis=2)
        else:
            min_channel = image.copy()
        
        # Local minimum over patch
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel.astype(float)
    
    @staticmethod
    def analyze(image, patch_size=15):
        """
        Comprehensive haze density analysis using Dark Channel Prior.
        
        Returns:
            dict: Contains haze_density_map and average_haze_density
        """
        dark_channel = DarkChannelPrior.compute_dark_channel(image, patch_size)
        
        # Normalize to 0-1
        dark_channel_norm = dark_channel / 255.0
        
        # Average haze density
        avg_haze_density = np.mean(dark_channel_norm)
        
        # Higher dark channel = more haze
        return {
            'haze_density_map': dark_channel_norm,
            'average_haze_density': avg_haze_density,
            'max_haze_density': np.max(dark_channel_norm),
            'min_haze_density': np.min(dark_channel_norm)
        }


class StructuralAnalyzer:
    """Analyzes structural visibility decay through texture variance and local structure."""
    
    @staticmethod
    def compute_texture_variance(image, window_size=5):
        """
        Computes local texture variance as a measure of structural information.
        
        Lower variance = less structural detail = more degradation
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Local variance using standard deviation filter
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        mean_local = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Compute local variance
        local_var = cv2.filter2D((gray.astype(np.float32) - mean_local)**2, -1, kernel)
        
        # Ensure non-negative and handle edge cases
        local_var = np.maximum(local_var, 0)
        
        return local_var
    
    @staticmethod
    def compute_laplacian_variance(image):
        """
        Laplacian variance as a measure of image sharpness.
        
        Higher variance = sharper image = better visibility
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        return laplacian_var
    
    @staticmethod
    def analyze(image, window_size=5):
        """
        Comprehensive structural visibility decay analysis.
        
        Returns:
            dict: Contains structural_decay_score (0-1, higher = worse) and metrics
        """
        texture_var_map = StructuralAnalyzer.compute_texture_variance(image, window_size)
        laplacian_var = StructuralAnalyzer.compute_laplacian_variance(image)
        
        # Average texture variance
        avg_texture_var = np.mean(texture_var_map)
        
        # Normalize texture variance (typical range: 0-1000, normalize to 0-1)
        # Handle edge case where variance is 0
        if avg_texture_var > 0:
            normalized_texture_var = min(avg_texture_var / 1000.0, 1.0)
        else:
            normalized_texture_var = 0.0
        
        # Normalize Laplacian variance (typical range: 0-10000, normalize to 0-1)
        # Handle edge case where variance is 0
        if laplacian_var > 0:
            normalized_laplacian_var = min(laplacian_var / 10000.0, 1.0)
        else:
            normalized_laplacian_var = 0.0
        
        # Structural quality (higher = better)
        structural_quality = (normalized_texture_var * 0.5 + normalized_laplacian_var * 0.5)
        
        # Structural decay score (1 - quality)
        structural_decay_score = 1.0 - structural_quality
        
        return {
            'structural_decay_score': structural_decay_score,
            'texture_variance': avg_texture_var,
            'laplacian_variance': laplacian_var,
            'structural_quality': structural_quality,
            'texture_variance_map': texture_var_map
        }
