"""
Calibration Module
Maps pupil coordinates to screen coordinates. Displays 5 different points 
on screen where gaze must be held for either 3 seconds or long enough 
for 10 samples to be taken.
"""

import time
import json
import numpy as np
from pathlib import Path


class CalibrationModule:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_data = []  # [(pupil_x, pupil_y, screen_x, screen_y, h_ratio, v_ratio), ...]
        self.is_calibrated = False
        
        # Calibration parameters
        self.min_samples_per_point = 10
        self.calibration_duration_per_point = 3.0  # seconds
        self.outlier_threshold = 20  # pixels
        
    def get_calibration_points(self):
        """Define 5 calibration points covering screen area"""
        margin = 150
        points = [
            (self.screen_width // 2, self.screen_height // 2),  # Center
            (margin, margin),  # Top-left
            (self.screen_width - margin, margin),  # Top-right
            (margin, self.screen_height - margin),  # Bottom-left
            (self.screen_width - margin, self.screen_height - margin)  # Bottom-right
        ]
        return points
    
    def collect_samples_for_point(self, data_acquisition, point_x, point_y):
        """
        Collect calibration samples for a specific point
        Continues until either 10 samples or 3 seconds elapsed
        """
        samples = []
        start_time = time.time()
        
        while (len(samples) < self.min_samples_per_point and 
               (time.time() - start_time) < self.calibration_duration_per_point):
            
            frame_data = data_acquisition.get_frame_data()
            if frame_data and frame_data['pupils_located']:
                left_pupil = frame_data['left_pupil']
                right_pupil = frame_data['right_pupil']
                h_ratio = frame_data['horizontal_ratio']
                v_ratio = frame_data['vertical_ratio']
                
                if left_pupil and right_pupil and h_ratio is not None and v_ratio is not None:
                    # Calculate precise average pupil position
                    avg_x = (left_pupil[0] + right_pupil[0]) / 2.0
                    avg_y = (left_pupil[1] + right_pupil[1]) / 2.0
                    
                    samples.append((avg_x, avg_y, h_ratio, v_ratio))
            
            time.sleep(0.05)  # 20 FPS sampling
        
        return samples
    
    def filter_outliers(self, samples):
        """Remove outlier samples using median-based filtering"""
        if len(samples) < 3:
            return samples
            
        x_coords = [s[0] for s in samples]
        y_coords = [s[1] for s in samples]
        
        median_x = np.median(x_coords)
        median_y = np.median(y_coords)
        
        # Keep samples within reasonable distance from median
        filtered_samples = []
        for sx, sy, hr, vr in samples:
            if (abs(sx - median_x) < self.outlier_threshold and 
                abs(sy - median_y) < self.outlier_threshold):
                filtered_samples.append((sx, sy, hr, vr))
        
        return filtered_samples
    
    def process_calibration_point(self, samples, screen_x, screen_y):
        """Process samples for a calibration point and add to calibration data"""
        if len(samples) < 5:  # Minimum quality threshold
            return False
            
        filtered_samples = self.filter_outliers(samples)
        
        if filtered_samples:
            # Use average of filtered samples
            avg_pupil_x = sum(s[0] for s in filtered_samples) / len(filtered_samples)
            avg_pupil_y = sum(s[1] for s in filtered_samples) / len(filtered_samples)
            avg_h_ratio = sum(s[2] for s in filtered_samples) / len(filtered_samples)
            avg_v_ratio = sum(s[3] for s in filtered_samples) / len(filtered_samples)
            
            self.calibration_data.append((
                avg_pupil_x, avg_pupil_y, screen_x, screen_y, avg_h_ratio, avg_v_ratio
            ))
            return True
        
        return False
    
    def complete_calibration(self, successful_points):
        """Complete calibration if enough points were successful"""
        if successful_points >= 4:  # Need at least 4 good points
            self.is_calibrated = True
            self.save_calibration()
            return True
        else:
            self.calibration_data = []
            return False
    
    def predict_gaze_position(self, left_pupil, right_pupil, gaze_tracker):
        """Map pupil coordinates to screen coordinates using calibration data"""
        if not self.is_calibrated or not left_pupil or not right_pupil:
            return None
        
        # Calculate precise average pupil position
        pupil_x = (left_pupil[0] + right_pupil[0]) / 2.0
        pupil_y = (left_pupil[1] + right_pupil[1]) / 2.0
        
        # Get current gaze ratios
        h_ratio = gaze_tracker.horizontal_ratio()
        v_ratio = gaze_tracker.vertical_ratio()
        
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
            
            # Apply bounds checking
            pred_x = max(0, min(self.screen_width, pred_x))
            pred_y = max(0, min(self.screen_height, pred_y))
            
            return (pred_x, pred_y)
        
        return None
    
    def save_calibration(self):
        """Save calibration data to file"""
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
        """Load previous calibration data"""
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
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_data = []
        self.is_calibrated = False
    
    def get_calibration_status(self):
        """Get calibration status information"""
        return {
            'is_calibrated': self.is_calibrated,
            'num_points': len(self.calibration_data),
            'screen_resolution': (self.screen_width, self.screen_height)
        }
