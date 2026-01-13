"""
AirSight Streamlit Web Interface

Interactive web application for visibility analysis with AQI estimation.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import time

from airsight.core.visibility_analyzer import VisibilityAnalyzer
from airsight.core.video_processor import VideoProcessor
from airsight.visualization.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="AirSight - Visual Visibility & Haze Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* AQI Card Styles */
    .aqi-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
        border: 2px solid;
        transition: transform 0.3s ease;
    }
    
    .aqi-card:hover {
        transform: translateY(-5px);
    }
    
    .aqi-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .aqi-category {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .aqi-message {
        font-size: 1rem;
        color: #475569;
        margin-top: 1rem;
        line-height: 1.6;
        font-style: italic;
    }
    
    /* Metric Card Styles */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateX(5px);
    }
    
    /* Feature Score Bars */
    .feature-bar-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        border-bottom: 3px solid #667eea;
    }
    
    /* Upload Area */
    .uploadedFile {
        border: 2px dashed #cbd5e1;
        border-radius: 0.75rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .aqi-value {
            font-size: 3rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌫️ AirSight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Visual Visibility & Haze Intelligence System | Image-Based AQI Detection</div>', unsafe_allow_html=True)

# Sidebar with improved styling
st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.markdown("---")

input_type = st.sidebar.radio(
    "**Input Type**",
    ["Image", "Video", "Live Camera"],
    help="Select whether to analyze an image, video, or live camera feed",
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.markdown("""
**AirSight** analyzes visual visibility degradation using:
- Contrast collapse detection
- Edge weakening analysis
- Color distortion measurement
- Haze density estimation
- Structural decay assessment

**AQI Estimation** is calculated purely from image analysis without requiring location or weather APIs.
""")

# Initialize analyzer
analyzer = VisibilityAnalyzer()

if input_type == "Image":
    st.markdown("### 📸 Image Visibility & Air Quality Analysis")
    
    # File uploader with better styling
    uploaded_file = st.file_uploader(
        "**Upload an image to analyze**",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file to analyze visibility degradation and estimate AQI",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Analyze
            with st.spinner("🔍 Analyzing visibility and estimating AQI..."):
                result = analyzer.analyze(image)
            
            # Get AQI information
            aqi_info = result['aqi']
            
            # Display AQI prominently at the top
            st.markdown("---")
            st.markdown("### 🌍 Air Quality Index (AQI) Estimation")
            
            # Create AQI card with dynamic color
            aqi_color = aqi_info['color']
            confidence_score = aqi_info.get('confidence_score', 0.8)
            reliability_flag = aqi_info.get('reliability_flag', True)
            reliability_score = aqi_info.get('reliability_score', 1.0)
            
            # Confidence indicator
            confidence_color = "#10b981" if confidence_score > 0.7 else "#f59e0b" if confidence_score > 0.5 else "#ef4444"
            confidence_label = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.5 else "Low"
            
            # Reliability indicator
            reliability_icon = "✅" if reliability_flag else "⚠️"
            reliability_text = "Reliable" if reliability_flag else "Low Reliability"
            
            aqi_html = f"""
            <div class="aqi-card" style="border-color: {aqi_color};">
                <div class="aqi-value" style="color: {aqi_color};">
                    {aqi_info['aqi']}
                </div>
                <div class="aqi-category" style="color: {aqi_color};">
                    {aqi_info['category']}
                </div>
                <div style="margin: 1rem 0; padding: 0.75rem; background: rgba(255,255,255,0.7); border-radius: 0.5rem;">
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div>
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Confidence</div>
                            <div style="font-weight: 600; color: {confidence_color};">
                                {confidence_score*100:.0f}% ({confidence_label})
                            </div>
                        </div>
                        <div style="border-left: 1px solid #cbd5e1; padding-left: 1rem;">
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Reliability</div>
                            <div style="font-weight: 600; color: {'#10b981' if reliability_flag else '#f59e0b'};">
                                {reliability_icon} {reliability_text}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="aqi-message">
                    {aqi_info['health_message']}
                </div>
            </div>
            """
            st.markdown(aqi_html, unsafe_allow_html=True)
            
            # Display warnings if any
            quality_issues = aqi_info.get('quality_issues', {})
            warnings = quality_issues.get('warnings', [])
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            
            # Disclaimer
            st.info("ℹ️ **Disclaimer**: This is a visual estimation based on image analysis only. It does not replace certified AQI sensors and should not be used for legal or medical purposes.")
            
            # Feature contributions (optional expander)
            with st.expander("📊 View Feature Contributions to AQI"):
                contributions = aqi_info.get('feature_contributions', {})
                st.markdown("**How each feature contributes to the AQI estimate:**")
                col1, col2 = st.columns(2)
                with col1:
                    for feature, value in list(contributions.items())[:4]:
                        st.metric(
                            feature.replace('_', ' ').title(),
                            f"{value:.1f}",
                            help=f"Contribution of {feature} to AQI"
                        )
                with col2:
                    for feature, value in list(contributions.items())[4:]:
                        st.metric(
                            feature.replace('_', ' ').title(),
                            f"{value:.1f}",
                            help=f"Contribution of {feature} to AQI"
                        )
            
            # Main content area
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📷 Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Visibility Analysis")
                
                # Visibility score with better display
                score = result['visibility_score']
                score_color = "#10b981" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
                score_emoji = "🟢" if score > 70 else "🟡" if score > 40 else "🔴"
                score_label = "Excellent" if score > 70 else "Good" if score > 40 else "Poor"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #64748b; font-size: 0.9rem;">Overall Visibility Score</h3>
                    <h2 style="margin: 0.5rem 0; color: {score_color}; font-size: 2.5rem; font-weight: 700;">
                        {score:.1f}<span style="font-size: 1.2rem; color: #94a3b8;">/100</span>
                    </h2>
                    <p style="margin: 0; color: {score_color}; font-weight: 600;">
                        {score_emoji} {score_label}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature breakdown
                st.markdown("#### 🔬 Feature Breakdown")
                feature_scores = result['feature_scores']
                
                for feature, score_val in feature_scores.items():
                    # Color based on score
                    bar_color = "#10b981" if score_val > 70 else "#f59e0b" if score_val > 40 else "#ef4444"
                    st.markdown(f"""
                    <div class="feature-bar-container">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #334155;">{feature.capitalize()}</span>
                            <span style="font-weight: 700; color: {bar_color};">{score_val:.1f}/100</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(score_val / 100.0)
            
            # Detailed visualizations
            st.markdown("---")
            st.markdown("### 📈 Detailed Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Dashboard", "🌫️ Haze Map", "🔍 Edge Detection", "📉 Histograms", "ℹ️ Raw Metrics"
            ])
            
            with tab1:
                st.markdown("#### Comprehensive Analysis Dashboard")
                fig = Visualizer.create_analysis_dashboard(image, result)
                st.pyplot(fig)
                plt.close(fig)
            
            with tab2:
                st.markdown("#### Haze Density Analysis")
                col1, col2 = st.columns([1, 1])
                haze_map = result['feature_maps']['haze_density']
                haze_overlay = Visualizer.create_haze_heatmap(image, haze_map)
                
                with col1:
                    st.image(cv2.cvtColor(haze_overlay, cv2.COLOR_BGR2RGB), 
                            caption="Haze Density Heatmap Overlay", use_container_width=True)
                
                with col2:
                    st.markdown("##### Haze Metrics")
                    st.metric("Average Haze Density", 
                             f"{result['raw_features']['haze']['average_haze_density']:.3f}",
                             help="Higher values indicate more haze/pollution")
                    st.metric("Max Haze Density", 
                             f"{result['raw_features']['haze']['max_haze_density']:.3f}")
                    st.metric("Min Haze Density", 
                             f"{result['raw_features']['haze']['min_haze_density']:.3f}")
                    
                    st.info("💡 **Tip**: Red areas indicate high haze density, which correlates with poor air quality.")
            
            with tab3:
                st.markdown("#### Edge Detection Analysis")
                col1, col2 = st.columns([1, 1])
                edge_map = result['feature_maps']['edge_canny']
                edge_overlay = Visualizer.create_edge_overlay(image, edge_map)
                
                with col1:
                    st.image(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB), 
                            caption="Edge Detection Overlay (Green = Detected Edges)", use_container_width=True)
                
                with col2:
                    st.markdown("##### Edge Metrics")
                    st.metric("Edge Strength Score", 
                             f"{result['feature_scores']['edge']:.2f}/100",
                             help="Higher scores indicate better edge preservation")
                    st.metric("Edge Density", 
                             f"{result['raw_features']['edge']['edge_density']:.3f}",
                             help="Proportion of pixels identified as edges")
                    st.metric("Mean Magnitude", 
                             f"{result['raw_features']['edge']['mean_magnitude']:.2f}")
                    
                    st.info("💡 **Tip**: Clear images have strong, well-defined edges. Haze reduces edge clarity.")
            
            with tab4:
                st.markdown("#### Histogram Analysis")
                fig = Visualizer.plot_histogram_comparison(image)
                st.pyplot(fig)
                plt.close(fig)
                st.info("💡 **Tip**: Histograms show the distribution of pixel intensities. Narrow distributions indicate reduced contrast.")
            
            with tab5:
                st.markdown("#### Raw Technical Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### Contrast Metrics")
                    st.json({
                        "Loss Index": f"{result['raw_features']['contrast']['contrast_loss_index']:.3f}",
                        "Entropy": f"{result['raw_features']['contrast']['entropy']:.3f}",
                        "Dynamic Range": f"{result['raw_features']['contrast']['dynamic_range']:.3f}",
                        "Histogram Spread": f"{result['raw_features']['contrast']['histogram_spread']:.3f}"
                    })
                
                with col2:
                    st.markdown("##### Color Metrics")
                    st.json({
                        "Degradation Score": f"{result['raw_features']['color']['color_degradation_score']:.3f}",
                        "Channel Balance": f"{result['raw_features']['color']['channel_balance']:.3f}",
                        "Mean Shift": f"{result['raw_features']['color']['channel_mean_shift']:.3f}",
                        "Inter-Channel Variance": f"{result['raw_features']['color']['inter_channel_variance']:.3f}"
                    })
                
                with col3:
                    st.markdown("##### Structure Metrics")
                    st.json({
                        "Decay Score": f"{result['raw_features']['structure']['structural_decay_score']:.3f}",
                        "Texture Variance": f"{result['raw_features']['structure']['texture_variance']:.2f}",
                        "Laplacian Variance": f"{result['raw_features']['structure']['laplacian_variance']:.2f}"
                    })
                
                st.markdown("##### AQI Calculation Details")
                st.json({
                    "Visibility Score": f"{aqi_info['visibility_score']:.2f}",
                    "Haze Density": f"{aqi_info['haze_density']:.3f}",
                    "Estimated AQI": aqi_info['aqi'],
                    "Category": aqi_info['category']
                })

else:  # Video
    st.markdown("### 🎥 Video Visibility Analysis")
    
    uploaded_file = st.file_uploader(
        "**Upload a video to analyze**",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze temporal visibility changes",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Configuration
        st.markdown("#### ⚙️ Processing Configuration")
        col1, col2 = st.columns(2)
        with col1:
            max_frames = st.number_input("Max Frames to Process", min_value=1, value=100, step=10,
                                         help="Limit the number of frames to process for faster analysis")
        with col2:
            sample_rate = st.number_input("Sample Rate (process every Nth frame)", min_value=1, value=1, step=1,
                                         help="Process every Nth frame to speed up analysis")
        
        # Process video
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎬 Processing video frames..."):
                processor = VideoProcessor()
                result = processor.process_video(tfile.name, max_frames, sample_rate)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display summary with better layout
            st.markdown("---")
            st.markdown("### 📊 Video Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            stats = result['statistics']
            
            with col1:
                st.metric("Mean Visibility", f"{stats['mean_visibility']:.2f}",
                         help="Average visibility across all processed frames")
            with col2:
                st.metric("Min Visibility", f"{stats['min_visibility']:.2f}",
                         help="Lowest visibility score detected")
            with col3:
                st.metric("Max Visibility", f"{stats['max_visibility']:.2f}",
                         help="Highest visibility score detected")
            with col4:
                st.metric("Clarity Drops", len(result['clarity_drops']),
                         help="Number of sudden visibility drops detected")
            
            # Display AQI statistics if available
            if result.get('aqi_stats'):
                st.markdown("---")
                st.markdown("### 🌍 Air Quality Index (AQI) Statistics")
                aqi_stats = result['aqi_stats']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean AQI", f"{aqi_stats['mean_aqi']:.0f}",
                             help="Average AQI across all processed frames")
                with col2:
                    st.metric("Min AQI", f"{aqi_stats['min_aqi']:.0f}",
                             help="Best (lowest) AQI detected")
                with col3:
                    st.metric("Max AQI", f"{aqi_stats['max_aqi']:.0f}",
                             help="Worst (highest) AQI detected")
                
                # Show AQI category for mean
                mean_aqi = int(aqi_stats['mean_aqi'])
                if mean_aqi <= 50:
                    aqi_category = "Good"
                    aqi_color = "#00E400"
                elif mean_aqi <= 100:
                    aqi_category = "Moderate"
                    aqi_color = "#FFFF00"
                elif mean_aqi <= 150:
                    aqi_category = "Unhealthy for Sensitive Groups"
                    aqi_color = "#FF7E00"
                elif mean_aqi <= 200:
                    aqi_category = "Unhealthy"
                    aqi_color = "#FF0000"
                elif mean_aqi <= 300:
                    aqi_category = "Very Unhealthy"
                    aqi_color = "#8F3F97"
                else:
                    aqi_category = "Hazardous"
                    aqi_color = "#7E0023"
                
                st.markdown(f"""
                <div class="info-box">
                    <p style="margin: 0; font-size: 1.1rem;">
                        <strong>Average Air Quality:</strong> 
                        <span style="color: {aqi_color}; font-weight: 700;">{aqi_category}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Temporal graph
            st.markdown("---")
            st.markdown("### 📈 Visibility Over Time")
            fig = Visualizer.plot_visibility_temporal(
                result['frame_times'],
                result['visibility_scores'],
                result['smoothed_scores'],
                result['clarity_drops']
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Worst frame
            st.markdown("---")
            st.markdown("### ⚠️ Worst Visibility Frame")
            worst = result['worst_frame']
            st.warning(f"**Frame Index:** {worst['index']} | **Time:** {worst['time']:.2f}s | **Score:** {worst['score']:.2f}/100")
            
            # Clarity drops
            if result['clarity_drops']:
                st.markdown("---")
                st.markdown("### 📉 Detected Clarity Drops")
                drops_data = []
                for drop in result['clarity_drops']:
                    drops_data.append({
                        "Frame": drop['index'],
                        "Time (s)": f"{result['frame_times'][drop['index']]:.2f}",
                        "Drop Magnitude": f"{drop['drop_magnitude']:.2f}",
                        "Before": f"{drop['before_score']:.2f}",
                        "After": f"{drop['after_score']:.2f}"
                    })
                st.dataframe(drops_data, use_container_width=True)
            
            # Full summary
            st.markdown("---")
            st.markdown("### 📋 Complete Analysis")
            fig = Visualizer.create_video_summary(result)
            st.pyplot(fig)
            plt.close(fig)
            
            # Cleanup
            os.unlink(tfile.name)

elif input_type == "Live Camera":
    st.markdown("### 📹 Live Camera Visibility & Air Quality Analysis")
    
    # Analysis settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Analysis Settings")
    show_metrics = st.sidebar.checkbox("Show Real-time Metrics", value=True)
    max_history = st.sidebar.slider("Max History Points", min_value=10, max_value=200, value=100,
                                   help="Maximum number of data points to keep in history")
    
    # Initialize session state
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    if 'history' not in st.session_state:
        st.session_state.history = {
            'visibility_scores': [],
            'aqi_values': [],
            'timestamps': []
        }
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Use Streamlit's camera input
    st.markdown("#### 📷 Capture Image from Camera")
    camera_image = st.camera_input("Take a picture with your camera", 
                                   help="Click the button to capture an image from your camera for analysis")
    
    if camera_image is not None:
        # Convert PIL image to OpenCV format
        image_array = np.array(camera_image)
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Analyze the captured image
        with st.spinner("🔍 Analyzing visibility and estimating AQI..."):
            try:
                result = analyzer.analyze(image_bgr)
                st.session_state.latest_result = result
                st.session_state.frame_count += 1
                
                # Update history
                current_time = time.time()
                st.session_state.history['visibility_scores'].append(result['visibility_score'])
                st.session_state.history['aqi_values'].append(result['aqi']['aqi'])
                st.session_state.history['timestamps'].append(current_time)
                
                # Keep only last max_history readings
                if len(st.session_state.history['visibility_scores']) > max_history:
                    st.session_state.history['visibility_scores'].pop(0)
                    st.session_state.history['aqi_values'].pop(0)
                    st.session_state.history['timestamps'].pop(0)
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
        
        # Display results
        if st.session_state.latest_result:
            result = st.session_state.latest_result
            aqi_info = result['aqi']
            
            # Display AQI prominently
            st.markdown("---")
            st.markdown("### 🌍 Air Quality Index (AQI) Estimation")
            
            aqi_color = aqi_info['color']
            confidence_score = aqi_info.get('confidence_score', 0.8)
            reliability_flag = aqi_info.get('reliability_flag', True)
            
            confidence_color = "#10b981" if confidence_score > 0.7 else "#f59e0b" if confidence_score > 0.5 else "#ef4444"
            confidence_label = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.5 else "Low"
            reliability_icon = "✅" if reliability_flag else "⚠️"
            reliability_text = "Reliable" if reliability_flag else "Low Reliability"
            
            aqi_html = f"""
            <div class="aqi-card" style="border-color: {aqi_color};">
                <div class="aqi-value" style="color: {aqi_color};">
                    {aqi_info['aqi']}
                </div>
                <div class="aqi-category" style="color: {aqi_color};">
                    {aqi_info['category']}
                </div>
                <div style="margin: 1rem 0; padding: 0.75rem; background: rgba(255,255,255,0.7); border-radius: 0.5rem;">
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div>
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Confidence</div>
                            <div style="font-weight: 600; color: {confidence_color};">
                                {confidence_score*100:.0f}% ({confidence_label})
                            </div>
                        </div>
                        <div style="border-left: 1px solid #cbd5e1; padding-left: 1rem;">
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.25rem;">Reliability</div>
                            <div style="font-weight: 600; color: {'#10b981' if reliability_flag else '#f59e0b'};">
                                {reliability_icon} {reliability_text}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="aqi-message">
                    {aqi_info['health_message']}
                </div>
            </div>
            """
            st.markdown(aqi_html, unsafe_allow_html=True)
            
            # Display warnings if any
            quality_issues = aqi_info.get('quality_issues', {})
            warnings = quality_issues.get('warnings', [])
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            
            # Display metrics
            if show_metrics:
                st.markdown("---")
                st.markdown("#### 📊 Analysis Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Visibility Score", f"{result['visibility_score']:.1f}/100")
                with col2:
                    st.metric("Contrast", f"{result['feature_scores']['contrast']:.1f}/100")
                with col3:
                    st.metric("Edge", f"{result['feature_scores']['edge']:.1f}/100")
                with col4:
                    st.metric("Color", f"{result['feature_scores']['color']:.1f}/100")
                with col5:
                    st.metric("Structure", f"{result['feature_scores']['structure']:.1f}/100")
            
            # Display history graph if we have enough data
            if len(st.session_state.history['visibility_scores']) > 1:
                st.markdown("---")
                st.markdown("### 📈 Analysis History")
                try:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # Visibility over time
                    times = [(t - st.session_state.history['timestamps'][0]) 
                            for t in st.session_state.history['timestamps']]
                    axes[0].plot(times, st.session_state.history['visibility_scores'], 
                                'b-', linewidth=2, label='Visibility Score', marker='o', markersize=4)
                    axes[0].set_xlabel('Time (seconds)', fontsize=12)
                    axes[0].set_ylabel('Visibility Score', fontsize=12)
                    axes[0].set_title('Visibility Score Over Time', fontsize=14, fontweight='bold')
                    axes[0].set_ylim(0, 100)
                    axes[0].grid(True, alpha=0.3)
                    axes[0].legend()
                    
                    # AQI over time
                    axes[1].plot(times, st.session_state.history['aqi_values'], 
                                'r-', linewidth=2, label='AQI', alpha=0.7, marker='s', markersize=4)
                    axes[1].axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Good (50)')
                    axes[1].axhline(y=100, color='yellow', linestyle='--', alpha=0.5, label='Moderate (100)')
                    axes[1].axhline(y=150, color='orange', linestyle='--', alpha=0.5, label='Unhealthy for Sensitive (150)')
                    axes[1].set_xlabel('Time (seconds)', fontsize=12)
                    axes[1].set_ylabel('AQI', fontsize=12)
                    axes[1].set_title('Air Quality Index Over Time', fontsize=14, fontweight='bold')
                    axes[1].set_ylim(0, 500)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"⚠️ Could not display history graph: {str(e)}")
            
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = {
                    'visibility_scores': [],
                    'aqi_values': [],
                    'timestamps': []
                }
                st.session_state.frame_count = 0
                st.rerun()
    
    else:
        st.info("👆 Use the camera button above to capture an image for analysis")
        
        # Show instructions
        with st.expander("📖 How to use Live Camera"):
            st.markdown("""
            **Live Camera Analysis Instructions:**
            
            1. **Click the camera button** - This will activate your webcam
            2. **Position your camera** - Point it at the scene you want to analyze
            3. **Capture image** - Click the capture button to take a snapshot
            4. **View results** - The analysis will appear automatically
            5. **Track history** - Multiple captures create a history graph showing trends
            
            **Tips:**
            - Point camera at outdoor scenes for best AQI estimation
            - Ensure good lighting for accurate analysis
            - Take multiple captures over time to see trends
            - History is automatically limited to the configured maximum
            - Each capture is analyzed independently
            
            **Note:** This uses Streamlit's camera input which requires browser permissions.
            """)
    
    # Show current history stats
    if len(st.session_state.history['visibility_scores']) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 History Stats")
        st.sidebar.metric("Total Captures", st.session_state.frame_count)
        if len(st.session_state.history['visibility_scores']) > 0:
            avg_visibility = np.mean(st.session_state.history['visibility_scores'])
            avg_aqi = np.mean(st.session_state.history['aqi_values'])
            st.sidebar.metric("Avg Visibility", f"{avg_visibility:.1f}")
            st.sidebar.metric("Avg AQI", f"{avg_aqi:.0f}")
    
    else:
        st.info("👆 Click 'Start Camera' to begin live analysis")
        
        # Show instructions
        with st.expander("📖 How to use Live Camera"):
            st.markdown("""
            **Live Camera Analysis Instructions:**
            
            1. **Connect your camera** - Make sure your webcam or external camera is connected
            2. **Select camera index** - Use 0 for default camera, or try 1, 2, etc. if you have multiple cameras
            3. **Click 'Start Camera'** - Begin live analysis
            4. **Adjust settings** - Use the sidebar to configure:
               - Analysis interval: Process every Nth frame (higher = faster but less frequent updates)
               - Show live feed: Display the camera feed with overlaid metrics
               - Show real-time metrics: Display current visibility and AQI values
            5. **Monitor graphs** - Real-time graphs show visibility and AQI trends over time
            6. **Stop when done** - Click 'Stop Camera' to end the session
            
            **Tips:**
            - Point camera at outdoor scenes for best AQI estimation
            - Ensure good lighting for accurate analysis
            - Higher analysis intervals reduce CPU usage
            - History is automatically limited to last 100 readings
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; background: #f8fafc; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
            <strong>🌫️ AirSight</strong>
        </p>
        <p style='margin: 0.25rem 0;'>
            Transforming invisible visibility loss into measurable visual intelligence
        </p>
        <p style='margin: 0.25rem 0; color: #94a3b8; font-size: 0.9rem;'>
            Built with classical digital image and video processing techniques
        </p>
        <p style='margin-top: 1rem; color: #94a3b8; font-size: 0.85rem;'>
            AQI estimation is image-based and does not require location or weather API access
        </p>
    </div>
""", unsafe_allow_html=True)
