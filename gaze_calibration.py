"""
Gaze Tracking Calibration System
A comprehensive calibration module for accurate gaze tracking in social anxiety therapy applications.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import time
import threading
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pickle
import os
from gaze_tracking import GazeTracking

class GazeCalibration:
    """
    Handles the calibration process for gaze tracking.
    Uses 5-point calibration (center + 4 corners) to map pupil coordinates to screen coordinates.
    """
    
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration points (center and four corners)
        self.calibration_points = [
            (screen_width // 2, screen_height // 2),  # Center
            (100, 100),                               # Top-left
            (screen_width - 100, 100),               # Top-right
            (100, screen_height - 100),              # Bottom-left
            (screen_width - 100, screen_height - 100) # Bottom-right
        ]
        
        # Data storage
        self.pupil_data = []  # (left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y)
        self.screen_data = []  # (screen_x, screen_y)
        
        # Models for gaze prediction
        self.x_model = None
        self.y_model = None
        
        # Gaze tracking system
        self.gaze_tracker = GazeTracking()
        self.webcam = None
        
        # GUI components
        self.root = None
        self.canvas = None
        
        # Calibration state
        self.current_point_index = 0
        self.samples_collected = 0
        self.is_collecting = False
        self.collection_thread = None
        
    def initialize_camera(self):
        """Initialize the webcam for gaze tracking"""
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties for better performance
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)
    
    def create_gui(self):
        """Create the calibration GUI"""
        self.root = tk.Tk()
        self.root.title("Gaze Tracking Calibration")
        self.root.configure(bg='black')
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind escape key to exit
        self.root.bind('<Escape>', self.exit_calibration)
        
        # Instructions
        self.show_instructions()
    
    def show_instructions(self):
        """Display calibration instructions"""
        instructions = [
            "Gaze Tracking Calibration",
            "",
            "Instructions:",
            "1. Look at each red dot when it appears",
            "2. Keep your head still and focus on the dot",
            "3. Each point will collect data for 3 seconds",
            "4. Try to minimize blinking during collection",
            "",
            "Press SPACE to start calibration",
            "Press ESCAPE to exit"
        ]
        
        self.canvas.delete("all")
        
        y_start = self.screen_height // 2 - len(instructions) * 15
        for i, line in enumerate(instructions):
            self.canvas.create_text(
                self.screen_width // 2,
                y_start + i * 30,
                text=line,
                fill='white',
                font=('Arial', 16 if i == 0 else 12),
                anchor='center'
            )
        
        self.root.bind('<space>', self.start_calibration)
    
    def start_calibration(self, event=None):
        """Start the calibration process"""
        self.root.unbind('<space>')
        self.current_point_index = 0
        self.pupil_data = []
        self.screen_data = []
        
        try:
            self.initialize_camera()
            self.show_next_calibration_point()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize camera: {e}")
            self.exit_calibration()
    
    def show_next_calibration_point(self):
        """Display the next calibration point"""
        if self.current_point_index >= len(self.calibration_points):
            self.finish_calibration()
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Get current point
        point_x, point_y = self.calibration_points[self.current_point_index]
        
        # Draw calibration point
        dot_size = 20
        self.canvas.create_oval(
            point_x - dot_size, point_y - dot_size,
            point_x + dot_size, point_y + dot_size,
            fill='red', outline='red'
        )
        
        # Draw point number and instructions
        self.canvas.create_text(
            point_x, point_y - 50,
            text=f"Point {self.current_point_index + 1} of {len(self.calibration_points)}",
            fill='white',
            font=('Arial', 14),
            anchor='center'
        )
        
        self.canvas.create_text(
            point_x, point_y + 50,
            text="Look at the red dot",
            fill='white',
            font=('Arial', 12),
            anchor='center'
        )
        
        # Start data collection after a brief delay
        self.root.after(1000, self.start_data_collection)
    
    def start_data_collection(self):
        """Start collecting pupil data for the current point"""
        self.samples_collected = 0
        self.is_collecting = True
        
        # Update display to show collection status
        point_x, point_y = self.calibration_points[self.current_point_index]
        self.canvas.create_text(
            point_x, point_y + 80,
            text="Collecting data...",
            fill='yellow',
            font=('Arial', 12),
            anchor='center',
            tags='status'
        )
        
        # Start collection in separate thread
        self.collection_thread = threading.Thread(target=self.collect_data_for_point)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    def collect_data_for_point(self):
        """Collect pupil data for the current calibration point"""
        start_time = time.time()
        point_samples = []
        
        while (self.is_collecting and 
               len(point_samples) < 10 and 
               time.time() - start_time < 3.0):
            
            # Capture frame
            ret, frame = self.webcam.read()
            if not ret:
                continue
            
            # Process with gaze tracker
            self.gaze_tracker.refresh(frame)
            
            # Get pupil coordinates
            left_pupil = self.gaze_tracker.pupil_left_coords()
            right_pupil = self.gaze_tracker.pupil_right_coords()
            
            if left_pupil and right_pupil:
                point_samples.append((left_pupil[0], left_pupil[1], 
                                    right_pupil[0], right_pupil[1]))
                
                # Update status on main thread
                self.root.after(0, self.update_collection_status, len(point_samples))
            
            time.sleep(0.1)  # Small delay to prevent overwhelming
        
        # Calculate average pupil position for this point
        if point_samples:
            avg_left_x = np.mean([s[0] for s in point_samples])
            avg_left_y = np.mean([s[1] for s in point_samples])
            avg_right_x = np.mean([s[2] for s in point_samples])
            avg_right_y = np.mean([s[3] for s in point_samples])
            
            # Store data
            self.pupil_data.append((avg_left_x, avg_left_y, avg_right_x, avg_right_y))
            screen_x, screen_y = self.calibration_points[self.current_point_index]
            self.screen_data.append((screen_x, screen_y))
            
            # Move to next point
            self.current_point_index += 1
            self.root.after(500, self.show_next_calibration_point)
        else:
            # No data collected, show error and retry
            self.root.after(0, self.show_collection_error)
    
    def update_collection_status(self, samples_count):
        """Update the collection status display"""
        self.canvas.delete('status')
        point_x, point_y = self.calibration_points[self.current_point_index]
        self.canvas.create_text(
            point_x, point_y + 80,
            text=f"Collecting data... {samples_count}/10 samples",
            fill='yellow',
            font=('Arial', 12),
            anchor='center',
            tags='status'
        )
    
    def show_collection_error(self):
        """Show error when data collection fails"""
        self.canvas.delete('status')
        point_x, point_y = self.calibration_points[self.current_point_index]
        self.canvas.create_text(
            point_x, point_y + 80,
            text="No pupil data detected. Retrying...",
            fill='red',
            font=('Arial', 12),
            anchor='center',
            tags='status'
        )
        self.root.after(2000, self.start_data_collection)
    
    def finish_calibration(self):
        """Complete the calibration and train models"""
        self.is_collecting = False
        
        if len(self.pupil_data) < 3:
            messagebox.showerror("Error", "Insufficient calibration data. Please restart.")
            self.exit_calibration()
            return
        
        # Show processing message
        self.canvas.delete("all")
        self.canvas.create_text(
            self.screen_width // 2,
            self.screen_height // 2,
            text="Processing calibration data...",
            fill='white',
            font=('Arial', 16),
            anchor='center'
        )
        self.root.update()
        
        # Train calibration models
        success = self.train_models()
        
        if success:
            self.show_completion_message()
        else:
            messagebox.showerror("Error", "Failed to create calibration models.")
            self.exit_calibration()
    
    def train_models(self):
        """Train the gaze prediction models using collected data"""
        try:
            # Prepare features (pupil coordinates)
            X = []
            for left_x, left_y, right_x, right_y in self.pupil_data:
                # Use average pupil position and individual positions as features
                avg_x = (left_x + right_x) / 2
                avg_y = (left_y + right_y) / 2
                X.append([avg_x, avg_y, left_x, left_y, right_x, right_y])
            
            X = np.array(X)
            
            # Prepare targets (screen coordinates)
            screen_x = [point[0] for point in self.screen_data]
            screen_y = [point[1] for point in self.screen_data]
            
            # Create polynomial regression models for better accuracy
            self.x_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            
            self.y_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            
            # Train models
            self.x_model.fit(X, screen_x)
            self.y_model.fit(X, screen_y)
            
            # Save models
            self.save_calibration()
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def save_calibration(self):
        """Save the calibration models to file"""
        calibration_data = {
            'x_model': self.x_model,
            'y_model': self.y_model,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_points': self.calibration_points,
            'pupil_data': self.pupil_data,
            'screen_data': self.screen_data
        }
        
        with open('gaze_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
    
    def show_completion_message(self):
        """Show calibration completion message"""
        self.canvas.delete("all")
        
        messages = [
            "Calibration Complete!",
            "",
            "Your gaze tracking is now calibrated.",
            "Calibration data has been saved.",
            "",
            "Press ESCAPE to exit"
        ]
        
        y_start = self.screen_height // 2 - len(messages) * 15
        for i, message in enumerate(messages):
            self.canvas.create_text(
                self.screen_width // 2,
                y_start + i * 30,
                text=message,
                fill='green' if i == 0 else 'white',
                font=('Arial', 18 if i == 0 else 14),
                anchor='center'
            )
    
    def exit_calibration(self, event=None):
        """Exit the calibration process"""
        self.is_collecting = False
        
        if self.webcam:
            self.webcam.release()
        
        if self.root:
            self.root.destroy()
    
    def run(self):
        """Run the calibration process"""
        self.create_gui()
        self.root.mainloop()


class GazePredictor:
    """
    Uses calibrated models to predict gaze coordinates from pupil positions.
    """
    
    def __init__(self, calibration_file='gaze_calibration.pkl'):
        self.x_model = None
        self.y_model = None
        self.screen_width = 1920
        self.screen_height = 1080
        self.load_calibration(calibration_file)
    
    def load_calibration(self, calibration_file):
        """Load calibration models from file"""
        try:
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.x_model = calibration_data['x_model']
            self.y_model = calibration_data['y_model']
            self.screen_width = calibration_data['screen_width']
            self.screen_height = calibration_data['screen_height']
            
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def predict_gaze(self, left_pupil, right_pupil):
        """
        Predict screen coordinates from pupil positions.
        
        Args:
            left_pupil: (x, y) coordinates of left pupil
            right_pupil: (x, y) coordinates of right pupil
            
        Returns:
            (screen_x, screen_y) predicted gaze coordinates, or None if prediction fails
        """
        if not self.x_model or not self.y_model:
            return None
        
        if not left_pupil or not right_pupil:
            return None
        
        try:
            # Prepare features
            left_x, left_y = left_pupil
            right_x, right_y = right_pupil
            avg_x = (left_x + right_x) / 2
            avg_y = (left_y + right_y) / 2
            
            features = np.array([[avg_x, avg_y, left_x, left_y, right_x, right_y]])
            
            # Predict coordinates
            screen_x = self.x_model.predict(features)[0]
            screen_y = self.y_model.predict(features)[0]
            
            # Clamp to screen bounds
            screen_x = max(0, min(self.screen_width, screen_x))
            screen_y = max(0, min(self.screen_height, screen_y))
            
            return (int(screen_x), int(screen_y))
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


if __name__ == "__main__":
    # Run calibration
    calibrator = GazeCalibration()
    calibrator.run()
