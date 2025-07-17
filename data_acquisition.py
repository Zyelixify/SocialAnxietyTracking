import cv2
import time
from gaze_tracking import GazeTracking


class DataAcquisition:
    def __init__(self):
        self.gaze_tracker = GazeTracking()
        self.webcam = None
        self.is_running = False
        
    def initialize_camera(self):
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise RuntimeError("Could not open webcam")
        return True
    
    def start_acquisition(self):
        if not self.webcam:
            self.initialize_camera()
        self.is_running = True
        
    def stop_acquisition(self):
        self.is_running = False
        if self.webcam:
            self.webcam.release()
            
    def get_frame_data(self):
        if not self.webcam or not self.is_running:
            return None
            
        ret, frame = self.webcam.read()
        if not ret:
            return None
            
        # Process frame
        self.gaze_tracker.refresh(frame)
        
        # Extract raw pupil and gaze data
        frame_data = {
            'timestamp': time.time(),
            'frame': frame,
            'annotated_frame': self.gaze_tracker.annotated_frame(),
            'pupils_located': self.gaze_tracker.pupils_located,
            'is_blinking': self.gaze_tracker.is_blinking(),
            'left_pupil': self.gaze_tracker.pupil_left_coords(),
            'right_pupil': self.gaze_tracker.pupil_right_coords(),
            'horizontal_ratio': self.gaze_tracker.horizontal_ratio(),
            'vertical_ratio': self.gaze_tracker.vertical_ratio(),
            'gaze_direction': {
                'is_right': self.gaze_tracker.is_right(),
                'is_left': self.gaze_tracker.is_left(),
                'is_center': self.gaze_tracker.is_center()
            }
        }
        
        return frame_data
    
    def get_gaze_tracker(self):
        return self.gaze_tracker
    
    def is_camera_ready(self):
        return self.webcam is not None and self.webcam.isOpened()
    
    def cleanup(self):
        self.stop_acquisition()
        cv2.destroyAllWindows()
