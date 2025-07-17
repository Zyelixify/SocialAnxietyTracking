"""
Social Anxiety Tracking - Main Demo
Based on the GazeTracking library by Antoine Lamé

This script demonstrates real-time eye tracking and gaze direction detection,
which can be used for social anxiety research and tracking applications.
"""

import cv2
import time
import datetime
from gaze_tracking import GazeTracking

def main():
    """
    Main function that runs the gaze tracking demo
    """
    # Initialize the gaze tracking object
    gaze = GazeTracking()
    
    # Initialize webcam (0 for default camera)
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    
    print("Social Anxiety Tracking - Gaze Analysis")
    print("Press ESC to exit")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    # Variables for tracking
    start_time = time.time()
    
    try:
        while True:
            # Capture frame from webcam
            ret, frame = webcam.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Analyze the frame for gaze tracking
            gaze.refresh(frame)
            
            # Get the annotated frame with pupils highlighted
            frame = gaze.annotated_frame()
            
            # Initialize display text
            text = ""
            text_color = (147, 58, 31)  # Blue-ish color
            
            # Determine gaze direction and display appropriate text
            if gaze.is_blinking():
                text = "Blinking"
                text_color = (0, 255, 255)  # Yellow for blinking
            elif gaze.is_right():
                text = "Looking right"
                text_color = (0, 0, 255)  # Red for right
            elif gaze.is_left():
                text = "Looking left"
                text_color = (255, 0, 0)  # Blue for left
            elif gaze.is_center():
                text = "Looking center"
                text_color = (0, 255, 0)  # Green for center
            else:
                text = "No gaze detected"
                text_color = (128, 128, 128)  # Gray for no detection
            
            # Add main gaze direction text
            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, text_color, 2)
            
            # Get pupil coordinates
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            
            # Display pupil coordinates
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            
            # Get gaze ratios for more detailed analysis
            horizontal_ratio = gaze.horizontal_ratio()
            vertical_ratio = gaze.vertical_ratio()
            
            if horizontal_ratio is not None and vertical_ratio is not None:
                cv2.putText(frame, f"H Ratio: {horizontal_ratio:.2f}", (90, 200), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 100, 255), 1)
                cv2.putText(frame, f"V Ratio: {vertical_ratio:.2f}", (90, 230), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 100, 255), 1)
            
            # Calculate session time
            session_time = int(time.time() - start_time)
            cv2.putText(frame, f"Session: {session_time}s", (90, 270), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            # Add title to the frame
            cv2.putText(frame, "Social Anxiety Tracking - Gaze Analysis", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Social Anxiety Tracking", frame)
            
            # Check for ESC key to exit
            if cv2.waitKey(1) == 27:  # ESC key
                break
                
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    
    finally:
        # Clean up
        webcam.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        total_time = int(time.time() - start_time)
        print(f"\nSession completed! Total time: {total_time} seconds")
        print("Thank you for using Social Anxiety Tracking!")

if __name__ == "__main__":
    main()
