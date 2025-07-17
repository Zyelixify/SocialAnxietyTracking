import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
import time
import json
from pathlib import Path
from gaze_tracking import GazeTracking

class PreciseGazeCalibrator:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_data = []  # [(pupil_x, pupil_y, screen_x, screen_y), ...]
        self.gaze_tracker = GazeTracking()
        self.is_calibrated = False
        
        # Calibration quality metrics
        self.min_samples_per_point = 20
        self.calibration_duration_per_point = 4.0  # seconds
        
    def run_calibration(self):
        print("Starting precision calibration with dlib...")
        
        # Define 5 calibration points covering screen area
        margin = 150
        points = [
            (self.screen_width // 2, self.screen_height // 2),  # Center
            (margin, margin),  # Top-left
            (self.screen_width - margin, margin),  # Top-right
            (margin, self.screen_height - margin),  # Bottom-left
            (self.screen_width - margin, self.screen_height - margin)  # Bottom-right
        ]
        
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Create fullscreen calibration window
        root = tk.Tk()
        root.title("Precision Gaze Calibration")
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        
        canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg='black')
        canvas.pack()
        
        successful_points = 0
        
        for i, (x, y) in enumerate(points):
            print(f"Calibrating point {i+1}/5 at ({x}, {y})")
            
            # Show calibration point with instructions
            canvas.delete("all")
            canvas.create_oval(x-25, y-25, x+25, y+25, fill='red', outline='white', width=3)
            canvas.create_text(x, y-70, text=f"Focus on the RED DOT\nPoint {i+1}/5", 
                             fill='white', font=('Arial', 18), justify='center')
            canvas.create_text(x, y+70, text="Keep your head still\nLook directly at the center", 
                             fill='gray', font=('Arial', 12), justify='center')
            root.update()
            
            # Allow time for user to focus
            time.sleep(1.5)
            
            # Collect high-quality samples
            samples = []
            start_time = time.time()
            
            while (len(samples) < self.min_samples_per_point and 
                   (time.time() - start_time) < self.calibration_duration_per_point):
                
                ret, frame = webcam.read()
                if ret:
                    self.gaze_tracker.refresh(frame)
                    
                    # Only collect samples when pupils are well-detected
                    if self.gaze_tracker.pupils_located:
                        left_pupil = self.gaze_tracker.pupil_left_coords()
                        right_pupil = self.gaze_tracker.pupil_right_coords()
                        
                        if left_pupil and right_pupil:
                            # Calculate precise average pupil position
                            avg_x = (left_pupil[0] + right_pupil[0]) / 2.0
                            avg_y = (left_pupil[1] + right_pupil[1]) / 2.0
                            
                            # Quality check: ensure pupils are detected with confidence
                            horizontal_ratio = self.gaze_tracker.horizontal_ratio()
                            vertical_ratio = self.gaze_tracker.vertical_ratio()
                            
                            if horizontal_ratio is not None and vertical_ratio is not None:
                                samples.append((avg_x, avg_y, horizontal_ratio, vertical_ratio))
                
                time.sleep(0.05)  # 20 FPS sampling
            
            # Process samples for this calibration point
            if len(samples) >= 10:  # Minimum quality threshold
                # Remove outliers (simple method: remove samples too far from median)
                x_coords = [s[0] for s in samples]
                y_coords = [s[1] for s in samples]
                
                median_x = np.median(x_coords)
                median_y = np.median(y_coords)
                
                # Keep samples within reasonable distance from median
                filtered_samples = []
                for sx, sy, hr, vr in samples:
                    if (abs(sx - median_x) < 20 and abs(sy - median_y) < 20):
                        filtered_samples.append((sx, sy, hr, vr))
                
                if filtered_samples:
                    # Use average of filtered samples
                    avg_pupil_x = sum(s[0] for s in filtered_samples) / len(filtered_samples)
                    avg_pupil_y = sum(s[1] for s in filtered_samples) / len(filtered_samples)
                    avg_h_ratio = sum(s[2] for s in filtered_samples) / len(filtered_samples)
                    avg_v_ratio = sum(s[3] for s in filtered_samples) / len(filtered_samples)
                    
                    self.calibration_data.append((avg_pupil_x, avg_pupil_y, x, y, avg_h_ratio, avg_v_ratio))
                    successful_points += 1
                    print(f"  Successfully calibrated point {i+1} with {len(filtered_samples)} samples")
                else:
                    print(f"  Failed to get quality samples for point {i+1}")
            else:
                print(f"  Insufficient samples for point {i+1} ({len(samples)} samples)")
        
        root.destroy()
        webcam.release()
        
        if successful_points >= 4:  # Need at least 4 good points
            self.is_calibrated = True
            print(f"Calibration completed successfully! ({successful_points}/5 points)")
            self._save_calibration()
            return True
        else:
            print(f"Calibration failed - only {successful_points}/5 points successful")
            return False
    
    def _save_calibration(self):
        calibration_file = Path("calibration_data.json")
        data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_points': self.calibration_data,
            'timestamp': time.time()
        }
        
        with open(calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved to {calibration_file}")
    
    def load_calibration(self):
        calibration_file = Path("calibration_data.json")
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                
                if (data['screen_width'] == self.screen_width and 
                    data['screen_height'] == self.screen_height):
                    self.calibration_data = data['calibration_points']
                    self.is_calibrated = True
                    print("Previous calibration loaded successfully")
                    return True
                else:
                    print("Previous calibration is for different screen resolution")
            except Exception as e:
                print(f"Failed to load calibration: {e}")
        
        return False
    
    def predict_gaze_position(self, left_pupil, right_pupil):
        if not self.is_calibrated or not left_pupil or not right_pupil:
            return None
        
        # Calculate precise average pupil position
        pupil_x = (left_pupil[0] + right_pupil[0]) / 2.0
        pupil_y = (left_pupil[1] + right_pupil[1]) / 2.0
        
        # Get current gaze ratios from dlib
        h_ratio = self.gaze_tracker.horizontal_ratio()
        v_ratio = self.gaze_tracker.vertical_ratio()
        
        if h_ratio is None or v_ratio is None or len(self.calibration_data) < 3:
            return None
        
        # Enhanced prediction using both pupil coordinates AND gaze ratios
        distances = []
        for cal_pupil_x, cal_pupil_y, cal_screen_x, cal_screen_y, cal_h_ratio, cal_v_ratio in self.calibration_data:
            # Combined distance metric: spatial + ratio similarity
            spatial_dist = np.sqrt((pupil_x - cal_pupil_x)**2 + (pupil_y - cal_pupil_y)**2)
            ratio_dist = np.sqrt((h_ratio - cal_h_ratio)**2 + (v_ratio - cal_v_ratio)**2)
            
            # Weighted combination (spatial distance is more important)
            combined_dist = spatial_dist + (ratio_dist * 100)  # Scale ratio distance
            
            distances.append((combined_dist, cal_screen_x, cal_screen_y))
        
        # Use weighted average of closest calibration points
        distances.sort()
        num_points = min(4, len(distances))  # Use up to 4 closest points
        closest_points = distances[:num_points]
        
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for dist, screen_x, screen_y in closest_points:
            weight = 1 / (dist + 1)  # Inverse distance weighting
            weighted_x += screen_x * weight
            weighted_y += screen_y * weight
            total_weight += weight
        
        if total_weight > 0:
            pred_x = int(weighted_x / total_weight)
            pred_y = int(weighted_y / total_weight)
            
            # Apply smoothing and bounds checking
            pred_x = max(0, min(self.screen_width, pred_x))
            pred_y = max(0, min(self.screen_height, pred_y))
            
            return (pred_x, pred_y)
        
        return None

class AdvancedAnxietyDetector:
    def __init__(self):
        # Core tracking
        self.blink_count = 0
        self.gaze_positions = []
        self.gaze_velocities = []
        self.saccade_count = 0
        
        # Precise blink tracking
        self.blink_durations = []
        self.last_blink_time = None
        self.is_currently_blinking = False
        self.blink_start_time = None
        
        # Session tracking
        self.session_start = time.time()
        self.frame_count = 0
        self.center_gaze_count = 0
        self.edge_gaze_count = 0
        
        # Thresholds
        self.anxiety_blink_rate = 30  # blinks per minute
        self.anxiety_saccade_rate = 6  # saccades per second
        self.center_zone_radius = 200  # pixels from center
        self.edge_zone_margin = 100   # pixels from edges
        self.fixation_threshold = 50   # Pixels for fixation detection
        self.min_fixation_duration = 0.2  # Minimum fixation time in seconds
        
        # Recent gaze history for velocity calculation
        self.recent_gazes = []
        self.max_history = 10
        
    def analyze_frame(self, gaze_tracker, gaze_position=None):
        self.frame_count += 1
        current_time = time.time()
        
        # Precise blink tracking with duration analysis
        currently_blinking = gaze_tracker.is_blinking()
        
        if currently_blinking and not self.is_currently_blinking:
            # Blink started
            self.blink_start_time = current_time
            self.is_currently_blinking = True
            
        elif not currently_blinking and self.is_currently_blinking:
            # Blink ended
            if self.blink_start_time:
                blink_duration = current_time - self.blink_start_time
                self.blink_durations.append(blink_duration)
                self.blink_count += 1
                self.last_blink_time = current_time
            self.is_currently_blinking = False
            self.blink_start_time = None
        
        # Advanced gaze position analysis
        if gaze_position:
            self.gaze_positions.append((gaze_position, current_time))
            
            # Calculate gaze velocity and detect saccades
            if len(self.recent_gazes) > 0:
                prev_pos, prev_time = self.recent_gazes[-1]
                distance = np.sqrt((gaze_position[0] - prev_pos[0])**2 + 
                                 (gaze_position[1] - prev_pos[1])**2)
                time_diff = current_time - prev_time
                
                if time_diff > 0:
                    velocity = distance / time_diff
                    self.gaze_velocities.append(velocity)
                    
                    # Detect saccades (rapid eye movements)
                    if velocity > 300:  # Pixels per second threshold for saccades
                        self.saccade_count += 1
            
            # Update recent gaze history
            self.recent_gazes.append((gaze_position, current_time))
            if len(self.recent_gazes) > self.max_history:
                self.recent_gazes.pop(0)
            
            # Analyze gaze zones
            screen_center_x, screen_center_y = 960, 540  # Assuming 1920x1080
            
            # Distance from center
            center_distance = np.sqrt((gaze_position[0] - screen_center_x)**2 + 
                                    (gaze_position[1] - screen_center_y)**2)
            
            if center_distance <= self.center_zone_radius:
                self.center_gaze_count += 1
            
            # Check if looking at screen edges (avoidance behavior)
            if (gaze_position[0] <= self.edge_zone_margin or 
                gaze_position[0] >= (1920 - self.edge_zone_margin) or
                gaze_position[1] <= self.edge_zone_margin or 
                gaze_position[1] >= (1080 - self.edge_zone_margin)):
                self.edge_gaze_count += 1

    def get_comprehensive_assessment(self):
        session_duration = (time.time() - self.session_start) / 60  # minutes
        
        # Calculate advanced metrics
        blink_rate = self.blink_count / session_duration if session_duration > 0 else 0
        saccade_rate = self.saccade_count / session_duration if session_duration > 0 else 0
        
        center_gaze_ratio = self.center_gaze_count / max(1, len(self.gaze_positions))
        edge_gaze_ratio = self.edge_gaze_count / max(1, len(self.gaze_positions))
        
        # Average gaze velocity
        avg_velocity = np.mean(self.gaze_velocities) if self.gaze_velocities else 0
        
        # Enhanced blink analysis
        avg_blink_duration = np.mean(self.blink_durations) if self.blink_durations else 0
        blink_frequency_variance = np.var([
            self.blink_durations[i] - self.blink_durations[i-1] 
            for i in range(1, len(self.blink_durations))
        ]) if len(self.blink_durations) > 1 else 0
        
        # Anxiety scoring system
        anxiety_indicators = []
        anxiety_score = 0
        
        # Blink rate analysis
        if blink_rate > self.anxiety_blink_rate:
            anxiety_indicators.append(f"High blink rate: {blink_rate:.1f}/min (normal: ~15-20/min)")
            anxiety_score += 3
        elif blink_rate < 8:
            anxiety_indicators.append(f"Very low blink rate: {blink_rate:.1f}/min (may indicate stress)")
            anxiety_score += 1
        
        # Blink duration analysis (anxiety can cause shorter, more frequent blinks)
        if avg_blink_duration > 0:
            if avg_blink_duration < 0.1:  # Very short blinks
                anxiety_indicators.append(f"Rapid blinking pattern: {avg_blink_duration:.3f}s avg duration")
                anxiety_score += 2
            elif avg_blink_duration > 0.5:  # Unusually long blinks
                anxiety_indicators.append(f"Prolonged blinks: {avg_blink_duration:.3f}s avg duration")
                anxiety_score += 1
        
        # Blink pattern irregularity
        if blink_frequency_variance > 0.2:
            anxiety_indicators.append(f"Irregular blink patterns detected")
            anxiety_score += 1
        
        # Saccade rate analysis
        if saccade_rate > self.anxiety_saccade_rate:
            anxiety_indicators.append(f"Excessive eye movements: {saccade_rate:.1f}/min")
            anxiety_score += 2
        
        # Gaze avoidance patterns
        if center_gaze_ratio < 0.2:
            anxiety_indicators.append(f"Strong center avoidance: Only {center_gaze_ratio:.1%} center focus")
            anxiety_score += 3
        elif center_gaze_ratio < 0.4:
            anxiety_indicators.append(f"Moderate center avoidance: {center_gaze_ratio:.1%} center focus")
            anxiety_score += 2
        
        # Edge fixation (looking away behavior)
        if edge_gaze_ratio > 0.3:
            anxiety_indicators.append(f"High edge fixation: {edge_gaze_ratio:.1%} edge focus")
            anxiety_score += 2
        
        # Rapid scanning behavior
        if avg_velocity > 150:
            anxiety_indicators.append(f"Rapid gaze scanning: {avg_velocity:.0f} pixels/sec")
            anxiety_score += 1
        
        # Overall assessment
        if anxiety_score >= 8:
            assessment = "HIGH anxiety indicators detected"
        elif anxiety_score >= 5:
            assessment = "MODERATE anxiety indicators detected"
        elif anxiety_score >= 2:
            assessment = "MILD anxiety indicators detected"
        else:
            assessment = "No significant anxiety indicators"
        
        return {
            'assessment': assessment,
            'anxiety_score': anxiety_score,
            'max_score': 15,
            'session_duration': session_duration,
            'blink_rate': blink_rate,
            'avg_blink_duration': avg_blink_duration,
            'blink_count': self.blink_count,
            'center_gaze_ratio': center_gaze_ratio,
            'edge_gaze_ratio': edge_gaze_ratio,
            'saccade_rate': saccade_rate,
            'avg_gaze_velocity': avg_velocity,
            'indicators': anxiety_indicators
        }

class SocialAnxietyApp:
    def __init__(self):
        self.calibrator = PreciseGazeCalibrator()
        self.detector = AdvancedAnxietyDetector()
        self.gaze_tracker = GazeTracking()
        self.webcam = None
        
    def run_calibration(self):
        response = messagebox.askyesno("Calibration", 
                                     "Gaze calibration will improve accuracy.\n\n"
                                     "You'll see 5 red dots on screen.\n"
                                     "Look at each dot when it appears.\n"
                                     "Keep your head still.\n\n"
                                     "Run calibration now?")
        
        if response:
            return self.calibrator.run_calibration()
        return False
    
    def start_monitoring(self):
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        print("Starting anxiety monitoring...")
        print("Press ESC to stop")
        
        try:
            while True:
                ret, frame = self.webcam.read()
                if not ret:
                    break
                
                # Analyze frame
                self.gaze_tracker.refresh(frame)
                
                # Get gaze position if calibrated
                gaze_position = None
                if self.calibrator.is_calibrated:
                    left_pupil = self.gaze_tracker.pupil_left_coords()
                    right_pupil = self.gaze_tracker.pupil_right_coords()
                    gaze_position = self.calibrator.predict_gaze_position(left_pupil, right_pupil)
                
                # Analyze for anxiety
                self.detector.analyze_frame(self.gaze_tracker, gaze_position)
                
                # Create display
                display_frame = self._create_display(frame, gaze_position)
                cv2.imshow("Social Anxiety Monitor", display_frame)
                
                if cv2.waitKey(1) == 27:  # ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            self.webcam.release()
            cv2.destroyAllWindows()
            self._show_results()
    
    def _create_display(self, frame, gaze_position):
        display_frame = self.gaze_tracker.annotated_frame()
        
        # Status info
        status = "Pupils: " + ("✓" if self.gaze_tracker.pupils_located else "✗")
        cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.gaze_tracker.pupils_located else (0, 0, 255), 2)
        
        # Gaze position
        if gaze_position and self.calibrator.is_calibrated:
            cv2.putText(display_frame, f"Gaze: ({gaze_position[0]}, {gaze_position[1]})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(display_frame, "Gaze: Not calibrated", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        # Real-time metrics
        assessment = self.detector.get_comprehensive_assessment()
        cv2.putText(display_frame, f"Blinks: {self.detector.blink_count} ({assessment['blink_rate']:.1f}/min)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(display_frame, f"Center focus: {assessment['center_gaze_ratio']:.1%}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(display_frame, f"Assessment: {assessment['assessment']}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if assessment['anxiety_score'] == 0 else (0, 165, 255), 1)
        
        return display_frame
    
    def _show_results(self):
        assessment = self.detector.get_comprehensive_assessment()
        
        result = f"Session Results:\n"
        result += f"Duration: {assessment['session_duration']:.1f} minutes\n\n"
        result += f"Assessment: {assessment['assessment']}\n"
        result += f"Anxiety Score: {assessment['anxiety_score']}/{assessment['max_score']}\n\n"
        result += f"Detailed Metrics:\n"
        result += f"• Blinks: {assessment['blink_count']} total ({assessment['blink_rate']:.1f}/min)\n"
        result += f"• Avg Blink Duration: {assessment['avg_blink_duration']:.3f}s\n"
        result += f"• Center Focus: {assessment['center_gaze_ratio']:.1%}\n"
        result += f"• Edge Focus: {assessment['edge_gaze_ratio']:.1%}\n"
        result += f"• Saccade Rate: {assessment['saccade_rate']:.1f}/min\n"
        result += f"• Avg Gaze Velocity: {assessment['avg_gaze_velocity']:.0f} px/s\n\n"
        
        if assessment['indicators']:
            result += "Anxiety Indicators:\n"
            for indicator in assessment['indicators']:
                result += f"• {indicator}\n"
        else:
            result += "No anxiety indicators detected."
        
        messagebox.showinfo("Session Results", result)

def main():
    print("Social Anxiety Tracker - Proof of Concept")
    print("=========================================")
    
    # Hide main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    app = SocialAnxietyApp()
    
    # Optional calibration
    app.run_calibration()
    
    # Start monitoring
    if messagebox.askyesno("Start Monitoring", "Ready to start monitoring?\n\nThe system will track eye movements for anxiety indicators."):
        app.start_monitoring()
    
    root.destroy()

if __name__ == "__main__":
    main()
