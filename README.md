# AirSight — Visual Visibility & Haze Intelligence System

**"Transforming invisible visibility loss into measurable visual intelligence using digital image and video processing."**

## Overview

AirSight is a sensor-free image and video processing system that estimates visual visibility degradation by measuring:
- **Contrast collapse** — Histogram spread, entropy, and dynamic range compression
- **Edge weakening** — Sobel and Canny edge detection with gradient magnitude analysis
- **Color distortion** — RGB channel separation and inter-channel variance
- **Haze density** — Dark Channel Prior (DCP) implementation
- **Structural decay** — Texture variance and local structural degradation

The system produces interpretable numeric scores (0-100) and visual maps without relying on object detection or domain-specific machine learning.

## Key Features

✅ **Domain-Agnostic** — Works on any image or video, independent of content  
✅ **Sensor-Free** — No external sensors required  
✅ **Image-Based AQI Estimation** — Estimates Air Quality Index purely from image analysis  
✅ **Confidence Scoring** — Provides confidence scores and reliability flags for AQI estimates  
✅ **Explainable** — Fully interpretable feature extraction and scoring  
✅ **Classical Methods** — Based on proven digital image processing techniques  
✅ **Modular Design** — Clean, extensible architecture  
✅ **Hackathon-Ready** — Quick setup and demo-friendly interface

## AQI Estimation

The system estimates Air Quality Index (AQI) using comprehensive feature engineering:

**Baseline Features:**
- **Contrast Analysis** — Histogram spread, entropy, dynamic range
- **Edge Density** — Sobel and Canny edge detection
- **HSV Histograms** — Hue, Saturation, Value channel analysis
- **Dark Channel Prior** — Haze density estimation
- **Visibility Index** — Combined visibility score

**Output Format:**
- **Estimated AQI** (0-500) with standard categories
- **Confidence Score** (0-1) based on feature consistency
- **Reliability Flag** indicating if estimation is reliable
- **Quality Warnings** for night images, fog, indoor scenes, etc.

**Important Disclaimer:**
This is a visual estimation only and does not replace certified AQI sensors. Not intended for legal or medical use.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Test installation: `python test_installation.py`
3. Launch web interface: `streamlit run app.py`
4. After the web interface is up and running, upload the sample image in \Sample Image\ directory to test the project.
5. Alternatively, you may upload any image and use the project to analyze it.

