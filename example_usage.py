"""
Example Usage Scripts for AirSight

Demonstrates how to use the AirSight system for image and video analysis.
"""

import cv2
import numpy as np
from airsight.core.visibility_analyzer import VisibilityAnalyzer
from airsight.core.video_processor import VideoProcessor
from airsight.visualization.visualizer import Visualizer
from airsight.pipeline import process_image, process_video, quick_analyze


def example_image_analysis():
    """Example: Analyze a single image."""
    print("=" * 60)
    print("Example 1: Image Analysis")
    print("=" * 60)
    
    # Replace with your image path
    image_path = "sample_image.jpg"
    
    try:
        # Quick analysis
        score = quick_analyze(image_path)
        print(f"\nQuick Analysis - Visibility Score: {score:.2f}/100")
        
        # Full analysis
        result = process_image(image_path, output_path="analysis_dashboard.png", show_dashboard=False)
        
        print(f"\nFull Analysis Results:")
        print(f"  Overall Visibility: {result['visibility_score']:.2f}/100")
        print(f"\n  Feature Scores:")
        for feature, score_val in result['feature_scores'].items():
            print(f"    {feature.capitalize()}: {score_val:.2f}/100")
        
        print(f"\n  Degradation Metrics:")
        for feature, degradation in result['degradation_scores'].items():
            print(f"    {feature.capitalize()} Loss: {degradation:.3f}")
        
        print("\nDashboard saved to: analysis_dashboard.png")
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please provide a valid image path.")


def example_custom_weights():
    """Example: Use custom feature weights."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Feature Weights")
    print("=" * 60)
    
    # Custom weights (emphasize haze and edge detection)
    custom_weights = {
        'contrast': 0.15,
        'edge': 0.30,      # Increased weight
        'color': 0.10,
        'haze': 0.35,      # Increased weight
        'structure': 0.10
    }
    
    analyzer = VisibilityAnalyzer(weights=custom_weights)
    
    # Replace with your image path
    image_path = "sample_image.jpg"
    image = cv2.imread(image_path)
    
    if image is not None:
        result = analyzer.analyze(image)
        print(f"\nVisibility Score (Custom Weights): {result['visibility_score']:.2f}/100")
        print(f"\nWeights Used:")
        for feature, weight in result['weights'].items():
            print(f"  {feature.capitalize()}: {weight:.2%}")
    else:
        print(f"Error: Could not load image '{image_path}'")


def example_video_analysis():
    """Example: Analyze a video file."""
    print("\n" + "=" * 60)
    print("Example 3: Video Analysis")
    print("=" * 60)
    
    # Replace with your video path
    video_path = "sample_video.mp4"
    
    try:
        result = process_video(
            video_path,
            max_frames=50,      # Process first 50 frames
            sample_rate=2,      # Process every 2nd frame
            output_path="video_analysis.png",
            show_summary=False
        )
        
        stats = result['statistics']
        print(f"\nVideo Analysis Results:")
        print(f"  Processed Frames: {result['processed_frames']}")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"\n  Visibility Statistics:")
        print(f"    Mean: {stats['mean_visibility']:.2f}")
        print(f"    Std Dev: {stats['std_visibility']:.2f}")
        print(f"    Min: {stats['min_visibility']:.2f}")
        print(f"    Max: {stats['max_visibility']:.2f}")
        print(f"    Range: {stats['visibility_range']:.2f}")
        
        worst = result['worst_frame']
        print(f"\n  Worst Frame:")
        print(f"    Index: {worst['index']}")
        print(f"    Time: {worst['time']:.2f}s")
        print(f"    Score: {worst['score']:.2f}")
        
        print(f"\n  Clarity Drops Detected: {len(result['clarity_drops'])}")
        if result['clarity_drops']:
            print("    First 3 drops:")
            for drop in result['clarity_drops'][:3]:
                print(f"      Frame {drop['index']}: Drop of {drop['drop_magnitude']:.2f}")
        
        print("\nVideo analysis summary saved to: video_analysis.png")
        
    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found.")
        print("Please provide a valid video path.")


def example_individual_features():
    """Example: Access individual feature extractors."""
    print("\n" + "=" * 60)
    print("Example 4: Individual Feature Extraction")
    print("=" * 60)
    
    from airsight.core.feature_extractors import (
        ContrastAnalyzer,
        EdgeAnalyzer,
        ColorAnalyzer,
        DarkChannelPrior,
        StructuralAnalyzer
    )
    
    # Replace with your image path
    image_path = "sample_image.jpg"
    image = cv2.imread(image_path)
    
    if image is not None:
        # Contrast analysis
        contrast_result = ContrastAnalyzer.analyze(image)
        print(f"\nContrast Analysis:")
        print(f"  Loss Index: {contrast_result['contrast_loss_index']:.3f}")
        print(f"  Entropy: {contrast_result['entropy']:.3f}")
        print(f"  Dynamic Range: {contrast_result['dynamic_range']:.3f}")
        
        # Edge analysis
        edge_result = EdgeAnalyzer.analyze(image)
        print(f"\nEdge Analysis:")
        print(f"  Strength Score: {edge_result['edge_strength_score']:.3f}")
        print(f"  Mean Magnitude: {edge_result['mean_magnitude']:.2f}")
        print(f"  Edge Density: {edge_result['edge_density']:.3f}")
        
        # Color analysis
        color_result = ColorAnalyzer.analyze(image)
        print(f"\nColor Analysis:")
        print(f"  Degradation Score: {color_result['color_degradation_score']:.3f}")
        print(f"  Channel Balance: {color_result['channel_balance']:.3f}")
        
        # Haze analysis
        haze_result = DarkChannelPrior.analyze(image)
        print(f"\nHaze Analysis:")
        print(f"  Average Haze Density: {haze_result['average_haze_density']:.3f}")
        print(f"  Max Haze Density: {haze_result['max_haze_density']:.3f}")
        
        # Structure analysis
        structure_result = StructuralAnalyzer.analyze(image)
        print(f"\nStructure Analysis:")
        print(f"  Decay Score: {structure_result['structural_decay_score']:.3f}")
        print(f"  Texture Variance: {structure_result['texture_variance']:.2f}")
        print(f"  Laplacian Variance: {structure_result['laplacian_variance']:.2f}")
    else:
        print(f"Error: Could not load image '{image_path}'")


def example_visualization():
    """Example: Create custom visualizations."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Visualizations")
    print("=" * 60)
    
    # Replace with your image path
    image_path = "sample_image.jpg"
    image = cv2.imread(image_path)
    
    if image is not None:
        analyzer = VisibilityAnalyzer()
        result = analyzer.analyze(image)
        
        # Create haze heatmap
        haze_map = result['feature_maps']['haze_density']
        haze_overlay = Visualizer.create_haze_heatmap(image, haze_map)
        cv2.imwrite("haze_heatmap.jpg", haze_overlay)
        print("Haze heatmap saved to: haze_heatmap.jpg")
        
        # Create edge overlay
        edge_map = result['feature_maps']['edge_canny']
        edge_overlay = Visualizer.create_edge_overlay(image, edge_map)
        cv2.imwrite("edge_overlay.jpg", edge_overlay)
        print("Edge overlay saved to: edge_overlay.jpg")
        
        # Create histogram
        fig = Visualizer.plot_histogram_comparison(image)
        fig.savefig("histogram.png", dpi=150, bbox_inches='tight')
        print("Histogram saved to: histogram.png")
        
    else:
        print(f"Error: Could not load image '{image_path}'")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AirSight - Example Usage Scripts")
    print("=" * 60)
    print("\nNote: Update image/video paths in the examples before running.")
    print("\nRunning examples...\n")
    
    # Run examples (comment out ones you don't want to run)
    # example_image_analysis()
    # example_custom_weights()
    # example_video_analysis()
    # example_individual_features()
    # example_visualization()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo run examples, uncomment the function calls in the main block.")
    print("Make sure to update the image/video paths first.")
