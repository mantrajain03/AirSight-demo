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
✅ **Explainable** — Fully interpretable feature extraction and scoring  
✅ **Classical Methods** — Based on proven digital image processing techniques  
✅ **Modular Design** — Clean, extensible architecture  
✅ **Hackathon-Ready** — Quick setup and demo-friendly interface

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Test installation: `python test_installation.py`
3. Launch web interface: `streamlit run app.py`

For detailed documentation, see the full README.md file.
