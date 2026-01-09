"""
Visualization Module

Creates visual outputs including:
- Haze heatmaps
- Edge loss overlays
- Histogram comparisons
- Temporal visibility graphs
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    """Creates visual outputs for visibility analysis."""
    
    @staticmethod
    def create_haze_heatmap(image, haze_map, alpha=0.6):
        """
        Overlay haze density map on original image.
        
        Args:
            image: Original image (BGR)
            haze_map: Haze density map (0-1)
            alpha: Overlay transparency
        
        Returns:
            numpy array: Overlaid image (BGR)
        """
        # Normalize haze map to 0-255
        haze_uint8 = (haze_map * 255).astype(np.uint8)
        
        # Apply colormap (red = high haze, blue = low haze)
        heatmap = cv2.applyColorMap(haze_uint8, cv2.COLORMAP_HOT)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    @staticmethod
    def create_edge_overlay(image, edge_map, color=(0, 255, 0), thickness=1):
        """
        Overlay edge map on original image.
        
        Args:
            image: Original image (BGR)
            edge_map: Edge map (binary or magnitude)
            color: Edge color (BGR tuple)
            thickness: Edge line thickness
        
        Returns:
            numpy array: Image with edges overlaid
        """
        overlay = image.copy()
        
        # Normalize edge map if needed
        if edge_map.dtype != np.uint8:
            if edge_map.max() > 1.0:
                edge_map = (edge_map / edge_map.max() * 255).astype(np.uint8)
            else:
                edge_map = (edge_map * 255).astype(np.uint8)
        
        # Create colored edge overlay
        edge_colored = np.zeros_like(image)
        edge_colored[:, :, 0] = color[0]  # B
        edge_colored[:, :, 1] = color[1]  # G
        edge_colored[:, :, 2] = color[2]  # R
        
        # Mask edges
        mask = edge_map > (np.max(edge_map) * 0.3)  # Threshold at 30% of max
        overlay[mask] = edge_colored[mask]
        
        return overlay
    
    @staticmethod
    def plot_histogram_comparison(image, title="Image Histogram"):
        """
        Plot RGB and grayscale histograms.
        
        Args:
            image: Image (BGR format)
            title: Plot title
        
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # RGB histograms
        colors = ('b', 'g', 'r')
        channel_names = ('Blue', 'Green', 'Red')
        
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            axes[0].plot(hist, color=color, label=name, alpha=0.7)
        
        axes[0].set_title('RGB Channel Histograms')
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Grayscale histogram
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axes[1].plot(hist_gray, color='black', alpha=0.7)
        axes[1].set_title('Grayscale Histogram')
        axes[1].set_xlabel('Pixel Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_visibility_temporal(frame_times, visibility_scores, smoothed_scores=None,
                                clarity_drops=None, title="Visibility Over Time"):
        """
        Plot temporal visibility graph for video.
        
        Args:
            frame_times: List of frame times (seconds)
            visibility_scores: Raw visibility scores
            smoothed_scores: Smoothed scores (optional)
            clarity_drops: List of clarity drop events (optional)
            title: Plot title
        
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw scores
        ax.plot(frame_times, visibility_scores, 'b-', alpha=0.3, 
                label='Raw Visibility', linewidth=1)
        
        # Plot smoothed scores
        if smoothed_scores:
            ax.plot(frame_times, smoothed_scores, 'r-', 
                   label='Smoothed Visibility', linewidth=2)
        
        # Mark clarity drops
        if clarity_drops:
            for drop in clarity_drops:
                idx = drop['index']
                if idx < len(frame_times):
                    ax.axvline(x=frame_times[idx], color='orange', 
                             linestyle='--', alpha=0.7, linewidth=1)
                    ax.text(frame_times[idx], ax.get_ylim()[1] * 0.95,
                           f"Drop: {drop['drop_magnitude']:.1f}",
                           rotation=90, ha='right', va='top', fontsize=8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Visibility Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_analysis_dashboard(image, analysis_result, save_path=None):
        """
        Create comprehensive analysis dashboard.
        
        Args:
            image: Original image (BGR)
            analysis_result: Result from VisibilityAnalyzer.analyze()
            save_path: Optional path to save figure
        
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Visibility score display
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        score = analysis_result['visibility_score']
        color = 'green' if score > 70 else 'orange' if score > 40 else 'red'
        ax2.text(0.5, 0.5, f'Visibility Score\n{score:.1f}/100', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                color=color, transform=ax2.transAxes)
        ax2.text(0.5, 0.2, f"Quality: {'Excellent' if score > 70 else 'Good' if score > 40 else 'Poor'}",
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        
        # Feature scores bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        feature_scores = analysis_result['feature_scores']
        features = list(feature_scores.keys())
        scores = [feature_scores[f] for f in features]
        colors_map = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax3.barh(features, scores, color=colors_map)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Score')
        ax3.set_title('Feature Scores', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax3.text(score + 2, i, f'{score:.1f}', va='center', fontsize=10)
        
        # Haze heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        haze_map = analysis_result['feature_maps']['haze_density']
        haze_overlay = Visualizer.create_haze_heatmap(image, haze_map)
        ax4.imshow(cv2.cvtColor(haze_overlay, cv2.COLOR_BGR2RGB))
        ax4.set_title('Haze Density Map', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Edge overlay
        ax5 = fig.add_subplot(gs[1, 1])
        edge_map = analysis_result['feature_maps']['edge_canny']
        edge_overlay = Visualizer.create_edge_overlay(image, edge_map)
        ax5.imshow(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB))
        ax5.set_title('Edge Detection', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Texture variance map
        ax6 = fig.add_subplot(gs[1, 2])
        texture_map = analysis_result['feature_maps']['texture_variance']
        texture_norm = (texture_map / texture_map.max() * 255).astype(np.uint8)
        texture_colored = cv2.applyColorMap(texture_norm, cv2.COLORMAP_VIRIDIS)
        ax6.imshow(cv2.cvtColor(texture_colored, cv2.COLOR_BGR2RGB))
        ax6.set_title('Texture Variance Map', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Histogram
        ax7 = fig.add_subplot(gs[2, :])
        Visualizer.plot_histogram_comparison(image, title="")
        ax7 = fig.gca()
        ax7.set_title('RGB and Grayscale Histograms', fontsize=12, fontweight='bold')
        
        plt.suptitle('AirSight Visibility Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_video_summary(video_result, save_path=None):
        """
        Create summary visualization for video analysis.
        
        Args:
            video_result: Result from VideoProcessor.process_video()
            save_path: Optional path to save figure
        
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Temporal visibility graph
        ax1 = axes[0, 0]
        Visualizer.plot_visibility_temporal(
            video_result['frame_times'],
            video_result['visibility_scores'],
            video_result['smoothed_scores'],
            video_result['clarity_drops'],
            title=""
        )
        ax1 = fig.gca()
        ax1.set_title('Visibility Over Time', fontsize=12, fontweight='bold')
        
        # Statistics
        ax2 = axes[0, 1]
        ax2.axis('off')
        stats = video_result['statistics']
        worst = video_result['worst_frame']
        
        stats_text = f"""
        Video Statistics
        
        Mean Visibility: {stats['mean_visibility']:.2f}
        Std Deviation: {stats['std_visibility']:.2f}
        Min Visibility: {stats['min_visibility']:.2f}
        Max Visibility: {stats['max_visibility']:.2f}
        Visibility Range: {stats['visibility_range']:.2f}
        
        Worst Frame:
        Index: {worst['index']}
        Time: {worst['time']:.2f}s
        Score: {worst['score']:.2f}
        
        Clarity Drops: {len(video_result['clarity_drops'])}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, 
                family='monospace', va='center', transform=ax2.transAxes)
        
        # Visibility distribution
        ax3 = axes[1, 0]
        ax3.hist(video_result['smoothed_scores'], bins=30, 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(stats['mean_visibility'], color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {stats['mean_visibility']:.2f}")
        ax3.set_xlabel('Visibility Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Visibility Score Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature scores over time (if available)
        ax4 = axes[1, 1]
        if video_result['frame_results']:
            # Plot average feature scores
            feature_names = list(video_result['frame_results'][0]['feature_scores'].keys())
            for feature in feature_names:
                scores = [r['feature_scores'][feature] for r in video_result['frame_results']]
                ax4.plot(video_result['frame_times'], scores, 
                        label=feature.capitalize(), alpha=0.7, linewidth=1.5)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Feature Score')
            ax4.set_title('Feature Scores Over Time', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
        
        plt.suptitle('AirSight Video Analysis Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
