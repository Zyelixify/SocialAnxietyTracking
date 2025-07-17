"""
Test script to verify GazeTracking installation and basic functionality
"""

import sys
import cv2
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import dlib
        print("✓ dlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dlib: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        from gaze_tracking import GazeTracking
        print("✓ GazeTracking imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GazeTracking: {e}")
        return False
    
    return True

def test_camera():
    """Test if camera can be accessed"""
    print("\nTesting camera access...")
    
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("✗ Could not open webcam")
        return False
    
    ret, frame = webcam.read()
    webcam.release()
    
    if not ret:
        print("✗ Could not capture frame from webcam")
        return False
    
    print("✓ Camera access successful")
    print(f"  Frame shape: {frame.shape}")
    return True

def test_gaze_tracking():
    """Test basic GazeTracking functionality"""
    print("\nTesting GazeTracking initialization...")
    
    try:
        from gaze_tracking import GazeTracking
        gaze = GazeTracking()
        print("✓ GazeTracking object created successfully")
        
        # Test with a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gaze.refresh(dummy_frame)
        print("✓ Frame refresh method works")
        
        # Test method calls (should not crash even with no face detected)
        _ = gaze.pupils_located
        _ = gaze.is_blinking()
        _ = gaze.is_left()
        _ = gaze.is_right()
        _ = gaze.is_center()
        _ = gaze.horizontal_ratio()
        _ = gaze.vertical_ratio()
        _ = gaze.pupil_left_coords()
        _ = gaze.pupil_right_coords()
        _ = gaze.annotated_frame()
        
        print("✓ All GazeTracking methods accessible")
        return True
        
    except Exception as e:
        print(f"✗ GazeTracking test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("GazeTracking Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test camera
    if not test_camera():
        all_passed = False
        print("  Note: Camera test failed. This might be okay if no camera is connected.")
    
    # Test GazeTracking
    if not test_gaze_tracking():
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' for the basic demo")
        print("2. Run 'python data_collector.py' for data collection")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
