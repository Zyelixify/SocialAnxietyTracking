"""
Social Anxiety Tracking - Simple Proof of Concept
Basic social anxiety detection using eye tracking and simple calculations.
"""

import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
import time
from gaze_tracking import GazeTracking

class GazeDataCollector:
    """
    A class to collect and store gaze tracking data for analysis
    """
    
    def __init__(self, participant_id=None, session_name=None):
        self.participant_id = participant_id or f"participant_{int(time.time())}"
        self.session_name = session_name or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.gaze_data = []
        self.session_start_time = time.time()
        
        # Setup CSV file for data logging
        self.csv_filename = self.data_dir / f"{self.participant_id}_{self.session_name}.csv"
        self.setup_csv_file()
        
    def setup_csv_file(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'session_time', 'gaze_direction', 'is_blinking',
                'left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y',
                'horizontal_ratio', 'vertical_ratio', 'pupils_detected'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def collect_frame_data(self, gaze):
        """
        Collect data from a single frame and store it
        
        Args:
            gaze: GazeTracking object with current frame analysis
        """
        current_time = time.time()
        session_time = current_time - self.session_start_time
        
        # Determine gaze direction
        gaze_direction = "unknown"
        if gaze.is_blinking():
            gaze_direction = "blinking"
        elif gaze.is_center():
            gaze_direction = "center"
        elif gaze.is_left():
            gaze_direction = "left"
        elif gaze.is_right():
            gaze_direction = "right"
        
        # Get pupil coordinates
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        
        # Create data record
        data_record = {
            'timestamp': datetime.datetime.fromtimestamp(current_time).isoformat(),
            'session_time': round(session_time, 3),
            'gaze_direction': gaze_direction,
            'is_blinking': gaze.is_blinking(),
            'left_pupil_x': left_pupil[0] if left_pupil else None,
            'left_pupil_y': left_pupil[1] if left_pupil else None,
            'right_pupil_x': right_pupil[0] if right_pupil else None,
            'right_pupil_y': right_pupil[1] if right_pupil else None,
            'horizontal_ratio': gaze.horizontal_ratio(),
            'vertical_ratio': gaze.vertical_ratio(),
            'pupils_detected': gaze.pupils_located
        }
        
        # Store in memory
        self.gaze_data.append(data_record)
        
        # Write to CSV file
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = data_record.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(data_record)
    
    def save_session_summary(self):
        """Save a summary of the session as JSON"""
        session_summary = {
            'participant_id': self.participant_id,
            'session_name': self.session_name,
            'start_time': datetime.datetime.fromtimestamp(self.session_start_time).isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'duration_seconds': time.time() - self.session_start_time,
            'total_frames': len(self.gaze_data),
            'data_file': str(self.csv_filename)
        }
        
        # Calculate some basic statistics
        if self.gaze_data:
            gaze_directions = [d['gaze_direction'] for d in self.gaze_data]
            direction_counts = {
                'center': gaze_directions.count('center'),
                'left': gaze_directions.count('left'),
                'right': gaze_directions.count('right'),
                'blinking': gaze_directions.count('blinking'),
                'unknown': gaze_directions.count('unknown')
            }
            session_summary['gaze_direction_counts'] = direction_counts
            
            # Calculate detection rate
            detected_frames = sum(1 for d in self.gaze_data if d['pupils_detected'])
            session_summary['detection_rate'] = detected_frames / len(self.gaze_data) if self.gaze_data else 0
        
        # Save summary as JSON
        summary_filename = self.data_dir / f"{self.participant_id}_{self.session_name}_summary.json"
        with open(summary_filename, 'w') as jsonfile:
            json.dump(session_summary, jsonfile, indent=2)
        
        return session_summary

def run_data_collection_session(participant_id=None, duration_minutes=5):
    """
    Run a data collection session for specified duration
    
    Args:
        participant_id: Unique identifier for the participant
        duration_minutes: Duration of the session in minutes
    """
    print(f"Starting data collection session for {duration_minutes} minutes")
    
    # Initialize components
    gaze = GazeTracking()
    collector = GazeDataCollector(participant_id)
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return
    
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    print(f"Data will be saved to: {collector.csv_filename}")
    print("Press ESC to end session early")
    
    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            
            # Analyze frame
            gaze.refresh(frame)
            
            # Collect data
            collector.collect_frame_data(gaze)
            
            # Display frame with info
            display_frame = gaze.annotated_frame()
            
            # Add session info to display
            elapsed = time.time() - start_time
            remaining = max(0, duration_seconds - elapsed)
            
            cv2.putText(display_frame, f"Collecting Data - {remaining:.0f}s remaining", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frames: {len(collector.gaze_data)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow("Data Collection", display_frame)
            
            # Check for exit conditions
            if cv2.waitKey(1) == 27 or elapsed >= duration_seconds:  # ESC or time up
                break
                
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        
        # Save session summary
        summary = collector.save_session_summary()
        
        print(f"\nSession completed!")
        print(f"Total frames collected: {summary['total_frames']}")
        print(f"Detection rate: {summary.get('detection_rate', 0):.2%}")
        print(f"Data saved to: {collector.csv_filename}")

if __name__ == "__main__":
    # Example usage
    participant_id = input("Enter participant ID (or press Enter for auto-generated): ").strip()
    participant_id = participant_id if participant_id else None
    
    try:
        duration = float(input("Enter session duration in minutes (default 2): ") or "2")
    except ValueError:
        duration = 2.0
    
    run_data_collection_session(participant_id, duration)
