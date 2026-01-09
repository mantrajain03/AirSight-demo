"""
Main Pipeline for AirSight

Simple interface for processing images and videos.
"""

import cv2
import numpy as np
from .core.visibility_analyzer import VisibilityAnalyzer
from .core.video_processor import VideoProcessor
from .visualization.visualizer import Visualizer


def process_image(image_path, output_path=None, show_dashboard=True):
    """
    Process a single image.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save dashboard
        show_dashboard: Whether to display dashboard
    
    Returns:
        dict: Analysis results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Analyze
    analyzer = VisibilityAnalyzer()
    result = analyzer.analyze(image)
    
    # Create visualization
    if show_dashboard or output_path:
        fig = Visualizer.create_analysis_dashboard(image, result, save_path=output_path)
        if show_dashboard:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            plt.close(fig)
    
    return result


def process_video(video_path, max_frames=None, sample_rate=1, 
                 output_path=None, show_summary=True):
    """
    Process a video file.
    
    Args:
        video_path: Path to input video
        max_frames: Maximum frames to process
        sample_rate: Process every Nth frame
        output_path: Optional path to save summary
        show_summary: Whether to display summary
    
    Returns:
        dict: Video analysis results
    """
    # Process video
    processor = VideoProcessor()
    result = processor.process_video(video_path, max_frames, sample_rate)
    
    # Create visualization
    if show_summary or output_path:
        fig = Visualizer.create_video_summary(result, save_path=output_path)
        if show_summary:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            plt.close(fig)
    
    return result


def quick_analyze(image_path):
    """
    Quick analysis - returns just the visibility score.
    
    Args:
        image_path: Path to input image
    
    Returns:
        float: Visibility score (0-100)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    analyzer = VisibilityAnalyzer()
    result = analyzer.analyze(image)
    
    return result['visibility_score']
