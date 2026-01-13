"""
Video Processing Module

Handles frame-wise visibility estimation, temporal smoothing, and video analysis.
"""

import numpy as np
import cv2
from .visibility_analyzer import VisibilityAnalyzer
from collections import deque


class VideoProcessor:
    """
    Processes video files for temporal visibility analysis.
    """
    
    def __init__(self, visibility_analyzer=None, temporal_window=5):
        """
        Initialize video processor.
        
        Args:
            visibility_analyzer: VisibilityAnalyzer instance (creates new if None)
            temporal_window: Size of temporal smoothing window
        """
        self.analyzer = visibility_analyzer or VisibilityAnalyzer()
        self.temporal_window = temporal_window
    
    def process_video(self, video_path, max_frames=None, sample_rate=1):
        """
        Process video file frame by frame.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all)
            sample_rate: Process every Nth frame (1 = all frames)
        
        Returns:
            dict: Video analysis results with temporal data
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_results = []
        frame_times = []
        frame_indices = []
        
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                try:
                    # Analyze frame
                    result = self.analyzer.analyze(frame)
                    result['frame_index'] = frame_idx
                    result['frame_time'] = frame_idx / fps if fps > 0 else frame_idx
                    
                    frame_results.append(result)
                    frame_times.append(result['frame_time'])
                    frame_indices.append(frame_idx)
                    
                    processed_count += 1
                    if max_frames and processed_count >= max_frames:
                        break
                except Exception as e:
                    # Skip frames that fail to analyze
                    print(f"Warning: Failed to analyze frame {frame_idx}: {str(e)}")
                    continue
            
            frame_idx += 1
        
        cap.release()
        
        # Extract visibility scores
        visibility_scores = [r['visibility_score'] for r in frame_results]
        
        # Extract AQI values if available
        aqi_values = []
        if frame_results and 'aqi' in frame_results[0]:
            aqi_values = [r['aqi']['aqi'] for r in frame_results]
        
        # Temporal smoothing
        smoothed_scores = self._temporal_smooth(visibility_scores)
        
        # Detect sudden clarity drops
        clarity_drops = self._detect_clarity_drops(smoothed_scores, threshold=10.0)
        
        # Find worst visibility frame
        worst_frame_idx = np.argmin(smoothed_scores)
        worst_frame_result = frame_results[worst_frame_idx]
        
        # Compute statistics
        stats = {
            'mean_visibility': np.mean(smoothed_scores),
            'std_visibility': np.std(smoothed_scores),
            'min_visibility': np.min(smoothed_scores),
            'max_visibility': np.max(smoothed_scores),
            'visibility_range': np.max(smoothed_scores) - np.min(smoothed_scores)
        }
        
        # AQI statistics if available
        aqi_stats = None
        if aqi_values:
            aqi_stats = {
                'mean_aqi': np.mean(aqi_values),
                'min_aqi': np.min(aqi_values),
                'max_aqi': np.max(aqi_values),
                'aqi_values': aqi_values
            }
        
        return {
            'frame_results': frame_results,
            'visibility_scores': visibility_scores,
            'smoothed_scores': smoothed_scores,
            'frame_times': frame_times,
            'frame_indices': frame_indices,
            'clarity_drops': clarity_drops,
            'worst_frame': {
                'index': worst_frame_idx,
                'time': frame_times[worst_frame_idx],
                'score': smoothed_scores[worst_frame_idx],
                'result': worst_frame_result
            },
            'statistics': stats,
            'aqi_stats': aqi_stats,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': len(frame_results)
        }
    
    def _temporal_smooth(self, scores, window_size=None):
        """
        Apply temporal smoothing to visibility scores.
        
        Uses moving average with specified window size.
        """
        if window_size is None:
            window_size = self.temporal_window
        
        if len(scores) < window_size:
            return scores
        
        smoothed = []
        window = deque(maxlen=window_size)
        
        for score in scores:
            window.append(score)
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _detect_clarity_drops(self, scores, threshold=10.0):
        """
        Detect sudden drops in visibility (clarity drops).
        
        Args:
            scores: List of visibility scores
            threshold: Minimum drop magnitude to consider significant
        
        Returns:
            list: List of (index, drop_magnitude) tuples
        """
        drops = []
        
        for i in range(1, len(scores)):
            drop = scores[i-1] - scores[i]
            if drop > threshold:
                drops.append({
                    'index': i,
                    'drop_magnitude': drop,
                    'before_score': scores[i-1],
                    'after_score': scores[i]
                })
        
        return drops
    
    def extract_frame(self, video_path, frame_index):
        """
        Extract a specific frame from video.
        
        Args:
            video_path: Path to video file
            frame_index: Frame index to extract
        
        Returns:
            numpy array: Frame image
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not extract frame {frame_index}")
        
        return frame
