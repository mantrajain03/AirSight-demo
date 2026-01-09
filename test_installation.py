"""
Quick installation test script for AirSight.

Run this to verify that all dependencies are installed correctly.
"""

import sys

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  Version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  Version: {np.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import scipy
        print("✓ SciPy imported successfully")
        print(f"  Version: {scipy.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import SciPy: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
        print(f"  Version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Matplotlib: {e}")
        return False
    
    try:
        import streamlit
        print("✓ Streamlit imported successfully")
        print(f"  Version: {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import Streamlit: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pillow: {e}")
        return False
    
    return True


def test_airsight_modules():
    """Test AirSight module imports."""
    print("\nTesting AirSight modules...")
    
    try:
        from airsight.core.feature_extractors import (
            ContrastAnalyzer,
            EdgeAnalyzer,
            ColorAnalyzer,
            DarkChannelPrior,
            StructuralAnalyzer
        )
        print("✓ Feature extractors imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import feature extractors: {e}")
        return False
    
    try:
        from airsight.core.visibility_analyzer import VisibilityAnalyzer
        print("✓ Visibility analyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visibility analyzer: {e}")
        return False
    
    try:
        from airsight.core.video_processor import VideoProcessor
        print("✓ Video processor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import video processor: {e}")
        return False
    
    try:
        from airsight.visualization.visualizer import Visualizer
        print("✓ Visualizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualizer: {e}")
        return False
    
    try:
        from airsight.pipeline import process_image, process_video, quick_analyze
        print("✓ Pipeline functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pipeline: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with a dummy image."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        import cv2
        from airsight.core.visibility_analyzer import VisibilityAnalyzer
        
        # Create a dummy test image (random noise)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        analyzer = VisibilityAnalyzer()
        result = analyzer.analyze(test_image)
        
        assert 'visibility_score' in result
        assert 0 <= result['visibility_score'] <= 100
        assert 'feature_scores' in result
        assert 'feature_maps' in result
        
        print("✓ Visibility analyzer works correctly")
        print(f"  Test image visibility score: {result['visibility_score']:.2f}/100")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AirSight Installation Test")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Test AirSight modules
    if not test_airsight_modules():
        all_passed = False
        print("\n⚠ AirSight modules not found. Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
        print("\n⚠ Basic functionality test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! AirSight is ready to use.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run 'streamlit run app.py' to launch the web interface")
    print("  2. Or use the Python API as shown in example_usage.py")
    print()
