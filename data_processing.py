import time
import numpy as np
from collections import deque


class DataProcessing:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width // 2
        self.screen_center_y = screen_height // 2
        
        # Data storage
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
        
        # Thresholds and parameters
        self.anxiety_blink_rate = 30  # blinks per minute
        self.anxiety_saccade_rate = 6  # saccades per second
        self.center_zone_radius = 200  # pixels from center
        self.edge_zone_margin = 100   # pixels from edges
        self.saccade_velocity_threshold = 300  # pixels per second
        
        self.recent_gazes = deque(maxlen=10)  # For velocity calculation
        self.gaze_smoothing_window = 5
        self.smoothed_positions = deque(maxlen=self.gaze_smoothing_window)
        
    def smooth_gaze_data(self, gaze_position):
        if gaze_position is None:
            return None
            
        self.smoothed_positions.append(gaze_position)
        
        if len(self.smoothed_positions) < 3:
            return gaze_position
            
        # Moving average
        avg_x = sum(pos[0] for pos in self.smoothed_positions) / len(self.smoothed_positions)
        avg_y = sum(pos[1] for pos in self.smoothed_positions) / len(self.smoothed_positions)
        
        return (int(avg_x), int(avg_y))
    
    def process_blink_data(self, frame_data):
        current_time = frame_data['timestamp']
        currently_blinking = frame_data['is_blinking']
        
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
    
    def process_gaze_position(self, gaze_position, timestamp):
        if gaze_position is None:
            return
            
        # Smooth the gaze position
        smoothed_position = self.smooth_gaze_data(gaze_position)
        if smoothed_position is None:
            return
            
        self.gaze_positions.append((smoothed_position, timestamp))
        
        # Calculate gaze velocity and detect saccades
        if len(self.recent_gazes) > 0:
            prev_pos, prev_time = self.recent_gazes[-1]
            distance = np.sqrt((smoothed_position[0] - prev_pos[0])**2 + 
                             (smoothed_position[1] - prev_pos[1])**2)
            time_diff = timestamp - prev_time
            
            if time_diff > 0:
                velocity = distance / time_diff
                self.gaze_velocities.append(velocity)
                
                # Detect saccades (rapid eye movements)
                if velocity > self.saccade_velocity_threshold:
                    self.saccade_count += 1
        
        self.recent_gazes.append((smoothed_position, timestamp))
        self._analyze_gaze_zones(smoothed_position)
    
    def _analyze_gaze_zones(self, gaze_position):
        # Distance from center
        center_distance = np.sqrt((gaze_position[0] - self.screen_center_x)**2 + 
                                (gaze_position[1] - self.screen_center_y)**2)
        
        if center_distance <= self.center_zone_radius:
            self.center_gaze_count += 1
        
        # Check if looking at screen edges (avoidance behavior)
        if (gaze_position[0] <= self.edge_zone_margin or 
            gaze_position[0] >= (self.screen_width - self.edge_zone_margin) or
            gaze_position[1] <= self.edge_zone_margin or 
            gaze_position[1] >= (self.screen_height - self.edge_zone_margin)):
            self.edge_gaze_count += 1
    
    def process_frame(self, frame_data, gaze_position=None):
        self.frame_count += 1
        self.process_blink_data(frame_data)
        
        # Process gaze position if available
        if gaze_position:
            self.process_gaze_position(gaze_position, frame_data['timestamp'])
    
    def calculate_center_gaze_accuracy(self):
        if not self.gaze_positions:
            return 0.0
            
        center_positions = []
        for pos, _ in self.gaze_positions:
            distance = np.sqrt((pos[0] - self.screen_center_x)**2 + 
                             (pos[1] - self.screen_center_y)**2)
            if distance <= self.center_zone_radius:
                center_positions.append(distance)
        
        if not center_positions:
            return 0.0
            
        # Calculate accuracy 
        avg_distance = np.mean(center_positions)
        max_distance = self.center_zone_radius
        accuracy = (max_distance - avg_distance) / max_distance
        return max(0.0, accuracy)
    
    def calculate_look_away_frequency(self):
        if len(self.gaze_positions) < 2:
            return 0.0
            
        look_away_events = 0
        was_center = False
        
        for pos, _ in self.gaze_positions:
            distance = np.sqrt((pos[0] - self.screen_center_x)**2 + 
                             (pos[1] - self.screen_center_y)**2)
            is_center = distance <= self.center_zone_radius
            
            if was_center and not is_center:
                look_away_events += 1
                
            was_center = is_center
        
        session_duration = (time.time() - self.session_start) / 60  # minutes
        return look_away_events / max(0.1, session_duration)  # events per minute
    
    def get_comprehensive_analysis(self):
        session_duration = (time.time() - self.session_start) / 60  # minutes
        
        # Calculate metrics
        blink_rate = self.blink_count / session_duration if session_duration > 0 else 0
        saccade_rate = self.saccade_count / session_duration if session_duration > 0 else 0
        
        center_gaze_ratio = self.center_gaze_count / max(1, len(self.gaze_positions))
        edge_gaze_ratio = self.edge_gaze_count / max(1, len(self.gaze_positions))
        
        avg_velocity = np.mean(self.gaze_velocities) if self.gaze_velocities else 0
        avg_blink_duration = np.mean(self.blink_durations) if self.blink_durations else 0
        
        # Calculate advanced metrics
        center_accuracy = self.calculate_center_gaze_accuracy()
        look_away_frequency = self.calculate_look_away_frequency()
        
        # Blink pattern analysis
        blink_frequency_variance = np.var([
            self.blink_durations[i] - self.blink_durations[i-1] 
            for i in range(1, len(self.blink_durations))
        ]) if len(self.blink_durations) > 1 else 0
        
        # Anxiety scoring
        anxiety_indicators = []
        anxiety_score = 0
        
        # Blink rate analysis
        if blink_rate > self.anxiety_blink_rate:
            anxiety_indicators.append(f"High blink rate: {blink_rate:.1f}/min (normal: ~15-20/min)")
            anxiety_score += 3
        elif blink_rate < 8:
            anxiety_indicators.append(f"Very low blink rate: {blink_rate:.1f}/min (may indicate stress)")
            anxiety_score += 1
        
        # Blink duration analysis
        if avg_blink_duration > 0:
            if avg_blink_duration < 0.1:
                anxiety_indicators.append(f"Rapid blinking pattern: {avg_blink_duration:.3f}s avg duration")
                anxiety_score += 2
            elif avg_blink_duration > 0.5:
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
        
        # Edge fixation
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
            'blink_count': self.blink_count,
            'blink_rate': blink_rate,
            'avg_blink_duration': avg_blink_duration,
            'blink_frequency_variance': blink_frequency_variance,
            'center_gaze_ratio': center_gaze_ratio,
            'center_gaze_accuracy': center_accuracy,
            'edge_gaze_ratio': edge_gaze_ratio,
            'look_away_frequency': look_away_frequency,
            'saccade_count': self.saccade_count,
            'saccade_rate': saccade_rate,
            'avg_gaze_velocity': avg_velocity,
            'total_gaze_positions': len(self.gaze_positions),
            'indicators': anxiety_indicators
        }
    
    def reset_session(self):
        self.blink_count = 0
        self.gaze_positions = []
        self.gaze_velocities = []
        self.saccade_count = 0
        self.blink_durations = []
        self.last_blink_time = None
        self.is_currently_blinking = False
        self.blink_start_time = None
        self.session_start = time.time()
        self.frame_count = 0
        self.center_gaze_count = 0
        self.edge_gaze_count = 0
        self.recent_gazes.clear()
        self.smoothed_positions.clear()
