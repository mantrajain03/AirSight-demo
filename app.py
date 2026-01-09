"""
AirSight Streamlit Web Interface

Interactive web application for visibility analysis.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

from airsight.core.visibility_analyzer import VisibilityAnalyzer
from airsight.core.video_processor import VideoProcessor
from airsight.visualization.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="AirSight - Visual Visibility & Haze Intelligence",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">👁️ AirSight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Visual Visibility & Haze Intelligence System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Configuration")
input_type = st.sidebar.radio(
    "Input Type",
    ["Image", "Video"],
    help="Select whether to analyze an image or video"
)

# Initialize analyzer
analyzer = VisibilityAnalyzer()

if input_type == "Image":
    st.header("Image Visibility Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file to analyze visibility degradation"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Analyze
            with st.spinner("Analyzing visibility..."):
                result = analyzer.analyze(image)
            
            # Display results
            with col2:
                st.subheader("Analysis Results")
                
                # Visibility score
                score = result['visibility_score']
                score_color = "🟢" if score > 70 else "🟡" if score > 40 else "🔴"
                st.metric(
                    "Overall Visibility Score",
                    f"{score:.2f}/100",
                    delta=f"{score_color} {'Excellent' if score > 70 else 'Good' if score > 40 else 'Poor'}"
                )
                
                # Feature scores
                st.subheader("Feature Breakdown")
                feature_scores = result['feature_scores']
                
                for feature, score_val in feature_scores.items():
                    st.progress(score_val / 100.0, text=f"{feature.capitalize()}: {score_val:.2f}/100")
            
            # Detailed visualizations
            st.markdown("---")
            st.subheader("Detailed Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "Dashboard", "Haze Map", "Edge Detection", "Histograms"
            ])
            
            with tab1:
                fig = Visualizer.create_analysis_dashboard(image, result)
                st.pyplot(fig)
                plt.close(fig)
            
            with tab2:
                col1, col2 = st.columns([1, 1])
                haze_map = result['feature_maps']['haze_density']
                haze_overlay = Visualizer.create_haze_heatmap(image, haze_map)
                
                with col1:
                    st.image(cv2.cvtColor(haze_overlay, cv2.COLOR_BGR2RGB), 
                            caption="Haze Density Overlay", use_container_width=True)
                
                with col2:
                    st.metric("Average Haze Density", 
                             f"{result['raw_features']['haze']['average_haze_density']:.3f}")
                    st.metric("Max Haze Density", 
                             f"{result['raw_features']['haze']['max_haze_density']:.3f}")
            
            with tab3:
                col1, col2 = st.columns([1, 1])
                edge_map = result['feature_maps']['edge_canny']
                edge_overlay = Visualizer.create_edge_overlay(image, edge_map)
                
                with col1:
                    st.image(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB), 
                            caption="Edge Detection Overlay", use_container_width=True)
                
                with col2:
                    st.metric("Edge Strength Score", 
                             f"{result['feature_scores']['edge']:.2f}/100")
                    st.metric("Edge Density", 
                             f"{result['raw_features']['edge']['edge_density']:.3f}")
            
            with tab4:
                fig = Visualizer.plot_histogram_comparison(image)
                st.pyplot(fig)
                plt.close(fig)
            
            # Raw metrics
            with st.expander("View Raw Metrics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Contrast Metrics**")
                    st.json({
                        "Loss Index": f"{result['raw_features']['contrast']['contrast_loss_index']:.3f}",
                        "Entropy": f"{result['raw_features']['contrast']['entropy']:.3f}",
                        "Dynamic Range": f"{result['raw_features']['contrast']['dynamic_range']:.3f}"
                    })
                
                with col2:
                    st.write("**Color Metrics**")
                    st.json({
                        "Degradation Score": f"{result['raw_features']['color']['color_degradation_score']:.3f}",
                        "Channel Balance": f"{result['raw_features']['color']['channel_balance']:.3f}"
                    })
                
                with col3:
                    st.write("**Structure Metrics**")
                    st.json({
                        "Decay Score": f"{result['raw_features']['structure']['structural_decay_score']:.3f}",
                        "Texture Variance": f"{result['raw_features']['structure']['texture_variance']:.2f}"
                    })

else:  # Video
    st.header("Video Visibility Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze temporal visibility"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            max_frames = st.number_input("Max Frames to Process", min_value=1, value=100, step=10)
        with col2:
            sample_rate = st.number_input("Sample Rate (process every Nth frame)", min_value=1, value=1, step=1)
        
        # Process video
        if st.button("Analyze Video", type="primary"):
            with st.spinner("Processing video frames..."):
                processor = VideoProcessor()
                result = processor.process_video(tfile.name, max_frames, sample_rate)
            
            # Display summary
            st.subheader("Video Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            stats = result['statistics']
            
            with col1:
                st.metric("Mean Visibility", f"{stats['mean_visibility']:.2f}")
            with col2:
                st.metric("Min Visibility", f"{stats['min_visibility']:.2f}")
            with col3:
                st.metric("Max Visibility", f"{stats['max_visibility']:.2f}")
            with col4:
                st.metric("Clarity Drops", len(result['clarity_drops']))
            
            # Temporal graph
            st.subheader("Visibility Over Time")
            fig = Visualizer.plot_visibility_temporal(
                result['frame_times'],
                result['visibility_scores'],
                result['smoothed_scores'],
                result['clarity_drops']
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Worst frame
            st.subheader("Worst Visibility Frame")
            worst = result['worst_frame']
            st.write(f"Frame Index: {worst['index']}, Time: {worst['time']:.2f}s, Score: {worst['score']:.2f}")
            
            # Clarity drops
            if result['clarity_drops']:
                st.subheader("Detected Clarity Drops")
                drops_data = []
                for drop in result['clarity_drops']:
                    drops_data.append({
                        "Frame": drop['index'],
                        "Time (s)": f"{result['frame_times'][drop['index']]:.2f}",
                        "Drop Magnitude": f"{drop['drop_magnitude']:.2f}",
                        "Before": f"{drop['before_score']:.2f}",
                        "After": f"{drop['after_score']:.2f}"
                    })
                st.table(drops_data)
            
            # Full summary
            st.subheader("Complete Analysis")
            fig = Visualizer.create_video_summary(result)
            st.pyplot(fig)
            plt.close(fig)
            
            # Cleanup
            os.unlink(tfile.name)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>AirSight</strong> - Transforming invisible visibility loss into measurable visual intelligence</p>
        <p>Built with classical digital image and video processing techniques</p>
    </div>
""", unsafe_allow_html=True)
