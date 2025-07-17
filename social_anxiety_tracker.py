"""
Social Anxiety Tracking - Calibration and Detection System
A proof-of-concept application for detecting social anxiety indicators
through gaze patterns and eye movement analysis.
"""

import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
import time
import json
from pathlib import Path
from gaze_tracking import GazeTracking
import threading

class GazeCalibrator:
    """
    Handles gaze calibration using 5-point screen mapping
    """
    
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = self._generate_calibration_points()
        self.pupil_data = []
        self.screen_data = []
        self.gaze_tracker = GazeTracking()
        self.webcam = None
        self.model_x = None
        self.model_y = None
        
    def _generate_calibration_points(self):
        """Generate 5 calibration points: center and four corners"""
        margin = 100  # Pixels from edge
        points = [
            (self.screen_width // 2, self.screen_height // 2),  # Center
            (margin, margin),  # Top-left
            (self.screen_width - margin, margin),  # Top-right
            (margin, self.screen_height - margin),  # Bottom-left
            (self.screen_width - margin, self.screen_height - margin)  # Bottom-right
        ]
        return points
    
    def run_calibration(self):
        """Run the calibration process"""
        print("Starting gaze calibration...")
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Create Tkinter window for calibration points
        root = tk.Tk()
        root.title("Gaze Calibration")
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        
        canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg='black')
        canvas.pack()
        
        for i, (x, y) in enumerate(self.calibration_points):
            print(f"Calibrating point {i+1}/5 at ({x}, {y})")
            
            # Clear canvas and draw calibration point
            canvas.delete("all")
            canvas.create_oval(x-20, y-20, x+20, y+20, fill='red', outline='white', width=2)
            canvas.create_text(x, y-40, text=f"Look at the red dot\nPoint {i+1}/5", 
                             fill='white', font=('Arial', 14))
            root.update()
            
            # Collect data for this point
            point_samples = self._collect_samples_for_point(x, y)
            
            if point_samples:
                avg_pupil = np.mean(point_samples, axis=0)
                self.pupil_data.append(avg_pupil)
                self.screen_data.append([x, y])
            
            time.sleep(0.5)  # Brief pause between points
        
        root.destroy()
        self.webcam.release()
        
        # Train calibration models
        if len(self.pupil_data) >= 3:
            self._train_calibration_models()
            print("Calibration completed successfully!")
            return True
        else:
            print("Calibration failed - insufficient data points")
            return False
    
    def _collect_samples_for_point(self, screen_x, screen_y, duration=3.0, max_samples=10):
        """Collect pupil samples for a specific calibration point"""
        samples = []
        start_time = time.time()
        
        while len(samples) < max_samples and (time.time() - start_time) < duration:
            ret, frame = self.webcam.read()
            if not ret:
                continue
                
            self.gaze_tracker.refresh(frame)
            
            if self.gaze_tracker.pupils_located:
                left_pupil = self.gaze_tracker.pupil_left_coords()
                right_pupil = self.gaze_tracker.pupil_right_coords()
                
                if left_pupil and right_pupil:
                    # Average pupil position
                    avg_x = (left_pupil[0] + right_pupil[0]) / 2
                    avg_y = (left_pupil[1] + right_pupil[1]) / 2
                    samples.append([avg_x, avg_y])
            
            time.sleep(0.1)  # Small delay between samples
        
        return samples
    
    def _train_calibration_models(self):
        """Train linear regression models for pupil-to-screen mapping"""
        if len(self.pupil_data) < 3:
            return False
            
        X = np.array(self.pupil_data)
        y_screen = np.array(self.screen_data)
        
        # Train separate models for X and Y coordinates
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        
        self.model_x.fit(X, y_screen[:, 0])  # X coordinates
        self.model_y.fit(X, y_screen[:, 1])  # Y coordinates
        
        return True
    
    def predict_gaze_position(self, left_pupil, right_pupil):
        """Predict screen gaze position from pupil coordinates"""
        if not (self.model_x and self.model_y and left_pupil and right_pupil):
            return None
            
        # Average pupil position
        avg_x = (left_pupil[0] + right_pupil[0]) / 2
        avg_y = (left_pupil[1] + right_pupil[1]) / 2
        
        # Predict screen coordinates
        screen_x = self.model_x.predict([[avg_x, avg_y]])[0]
        screen_y = self.model_y.predict([[avg_x, avg_y]])[0]
        
        return (int(screen_x), int(screen_y))

class SocialAnxietyDetector:
    """
    Detects social anxiety indicators based on predefined gaze patterns
    """
    
    def __init__(self):
        self.blink_count = 0
        self.blink_timestamps = []
        self.gaze_positions = []
        self.avoidance_count = 0
        self.session_start = time.time()
        
        # Anxiety indicators thresholds
        self.high_blink_rate_threshold = 30  # blinks per minute
        self.gaze_avoidance_threshold = 0.7  # 70% of time looking away
        self.rapid_movement_threshold = 100  # pixels per frame
        
    def analyze_frame(self, gaze_tracker, gaze_position=None):
        """Analyze current frame for anxiety indicators"""
        current_time = time.time()
        
        # Track blinking
        if gaze_tracker.is_blinking():
            if not self.blink_timestamps or (current_time - self.blink_timestamps[-1]) > 0.3:
                self.blink_count += 1
                self.blink_timestamps.append(current_time)
        
        # Track gaze position if available
        if gaze_position:
            self.gaze_positions.append((gaze_position, current_time))
        
        # Check for gaze avoidance (looking away from center)
        if gaze_position:
            center_x, center_y = 960, 540  # Assuming 1920x1080 screen
            distance_from_center = np.sqrt((gaze_position[0] - center_x)**2 + 
                                         (gaze_position[1] - center_y)**2)
            if distance_from_center > 400:  # Far from center
                self.avoidance_count += 1
    
    def get_blink_rate(self):
        """Calculate blinks per minute"""
        session_duration = (time.time() - self.session_start) / 60  # minutes
        if session_duration > 0:
            return self.blink_count / session_duration
        return 0
    
    def get_gaze_avoidance_ratio(self):
        """Calculate ratio of time spent looking away from center"""
        if not self.gaze_positions:
            return 0
        return self.avoidance_count / len(self.gaze_positions)
    
    def detect_anxiety_indicators(self):
        """Detect social anxiety indicators and return assessment"""
        blink_rate = self.get_blink_rate()
        avoidance_ratio = self.get_gaze_avoidance_ratio()
        
        indicators = []
        anxiety_score = 0
        
        # High blink rate indicator
        if blink_rate > self.high_blink_rate_threshold:
            indicators.append(f"High blink rate: {blink_rate:.1f} blinks/min")
            anxiety_score += 2
        
        # Gaze avoidance indicator
        if avoidance_ratio > self.gaze_avoidance_threshold:
            indicators.append(f"Gaze avoidance: {avoidance_ratio:.1%} of time")
            anxiety_score += 3
        
        # Rapid eye movements (simplified check)
        if len(self.gaze_positions) > 10:
            recent_positions = self.gaze_positions[-10:]
            movements = []
            for i in range(1, len(recent_positions)):
                prev_pos, _ = recent_positions[i-1]
                curr_pos, _ = recent_positions[i]
                movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                movements.append(movement)
            
            avg_movement = np.mean(movements) if movements else 0
            if avg_movement > self.rapid_movement_threshold:
                indicators.append(f"Rapid eye movements detected")
                anxiety_score += 1
        
        return {
            'anxiety_score': anxiety_score,
            'indicators': indicators,
            'blink_rate': blink_rate,
            'gaze_avoidance': avoidance_ratio,
            'assessment': self._get_anxiety_assessment(anxiety_score)
        }
    
    def _get_anxiety_assessment(self, score):
        """Convert anxiety score to assessment level"""
        if score >= 5:
            return "High anxiety indicators detected"
        elif score >= 3:
            return "Moderate anxiety indicators detected"
        elif score >= 1:
            return "Mild anxiety indicators detected"
        else:
            return "No significant anxiety indicators"

class SocialAnxietyApp:
    """
    Main application for social anxiety tracking
    """
    
    def __init__(self):
        self.calibrator = GazeCalibrator()
        self.detector = SocialAnxietyDetector()
        self.gaze_tracker = GazeTracking()
        self.webcam = None
        self.is_running = False
        
    def run_calibration(self):
        """Run the calibration process"""
        messagebox.showinfo("Calibration", 
                           "Gaze calibration will start.\n\n"
                           "Look at each red dot that appears on screen.\n"
                           "Keep your head still and focus on the center of each dot.\n\n"
                           "Click OK to begin.")
        
        success = self.calibrator.run_calibration()
        
        if success:
            messagebox.showinfo("Success", "Calibration completed successfully!")
            return True
        else:
            messagebox.showerror("Error", "Calibration failed. Please try again.")
            return False
    
    def start_monitoring(self):
        """Start the anxiety monitoring session"""
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_running = True
        print("Starting social anxiety monitoring...")
        print("Press ESC to stop monitoring")
        
        try:
            while self.is_running:
                ret, frame = self.webcam.read()
                if not ret:
                    break
                
                # Analyze frame with gaze tracker
                self.gaze_tracker.refresh(frame)
                
                # Get gaze position if calibrated
                gaze_position = None
                if self.calibrator.model_x and self.calibrator.model_y:
                    left_pupil = self.gaze_tracker.pupil_left_coords()
                    right_pupil = self.gaze_tracker.pupil_right_coords()
                    gaze_position = self.calibrator.predict_gaze_position(left_pupil, right_pupil)
                
                # Analyze for anxiety indicators
                self.detector.analyze_frame(self.gaze_tracker, gaze_position)
                
                # Display frame with information
                display_frame = self._create_display_frame(frame, gaze_position)
                cv2.imshow("Social Anxiety Monitoring", display_frame)
                
                # Check for exit
                if cv2.waitKey(1) == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        finally:
            self.is_running = False
            self.webcam.release()
            cv2.destroyAllWindows()
            
            # Show final assessment
            self._show_final_assessment()
    
    def _create_display_frame(self, frame, gaze_position):
        """Create display frame with overlays"""
        display_frame = self.gaze_tracker.annotated_frame()
        
        # Add status information
        y_offset = 30
        
        # Pupil detection status
        status_color = (0, 255, 0) if self.gaze_tracker.pupils_located else (0, 0, 255)
        cv2.putText(display_frame, f"Pupils: {'Detected' if self.gaze_tracker.pupils_located else 'Not Detected'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_offset += 30
        
        # Gaze position
        if gaze_position:
            cv2.putText(display_frame, f"Gaze: ({gaze_position[0]}, {gaze_position[1]})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            cv2.putText(display_frame, "Gaze: Not calibrated", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
        y_offset += 30
        
        # Real-time anxiety indicators
        blink_rate = self.detector.get_blink_rate()
        cv2.putText(display_frame, f"Blink rate: {blink_rate:.1f}/min", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_offset += 30
        
        avoidance_ratio = self.detector.get_gaze_avoidance_ratio()
        cv2.putText(display_frame, f"Gaze avoidance: {avoidance_ratio:.1%}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return display_frame
    
    def _show_final_assessment(self):
        """Show final anxiety assessment"""
        assessment = self.detector.detect_anxiety_indicators()
        
        result_text = f"Session Assessment:\n\n"
        result_text += f"Anxiety Score: {assessment['anxiety_score']}/6\n"
        result_text += f"Assessment: {assessment['assessment']}\n\n"
        result_text += f"Blink Rate: {assessment['blink_rate']:.1f} blinks/min\n"
        result_text += f"Gaze Avoidance: {assessment['gaze_avoidance']:.1%}\n\n"
        
        if assessment['indicators']:
            result_text += "Indicators detected:\n"
            for indicator in assessment['indicators']:
                result_text += f"• {indicator}\n"
        else:
            result_text += "No significant anxiety indicators detected."
        
        messagebox.showinfo("Assessment Results", result_text)

def main():
    """Main application entry point"""
    print("Social Anxiety Tracking - Proof of Concept")
    print("==========================================")
    
    app = SocialAnxietyApp()
    
    # Ask user if they want to calibrate
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    calibrate = messagebox.askyesno("Calibration", 
                                   "Do you want to run gaze calibration?\n\n"
                                   "Calibration improves gaze position accuracy but is optional.\n"
                                   "Basic anxiety detection will work without calibration.")
    
    if calibrate:
        if not app.run_calibration():
            print("Calibration failed, continuing without calibration...")
    
    # Start monitoring
    start_monitoring = messagebox.askyesno("Start Monitoring", 
                                         "Ready to start social anxiety monitoring?\n\n"
                                         "The system will track eye movements and detect\n"
                                         "potential anxiety indicators in real-time.")
    
    if start_monitoring:
        app.start_monitoring()
    
    root.destroy()

if __name__ == "__main__":
    main()
