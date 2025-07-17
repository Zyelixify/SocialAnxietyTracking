import cv2
import time
import numpy as np

class SimpleFaceTracker:
    def __init__(self):
        # Load OpenCV's pre-trained classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Tracking variables
        self.faces = []
        self.eyes = []
        self.frame = None
        
    def refresh(self, frame):
        self.frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        self.faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Detect eyes within faces
        self.eyes = []
        for (x, y, w, h) in self.faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            # Convert eye coordinates to global frame coordinates
            for (ex, ey, ew, eh) in eyes:
                self.eyes.append((x + ex, y + ey, ew, eh))
    
    def has_face_detected(self):
        """Check if any faces are detected"""
        return len(self.faces) > 0
    
    def has_eyes_detected(self):
        """Check if any eyes are detected"""
        return len(self.eyes) > 0
    
    def get_face_center(self):
        """Get the center point of the first detected face"""
        if self.faces is not None and len(self.faces) > 0:
            x, y, w, h = self.faces[0]
            return (x + w//2, y + h//2)
        return None
    
    def get_eye_centers(self):
        """Get the center points of detected eyes"""
        eye_centers = []
        for (x, y, w, h) in self.eyes:
            center_x = x + w//2
            center_y = y + h//2
            eye_centers.append((center_x, center_y))
        return eye_centers
    
    def estimate_gaze_direction(self):
        if not self.has_face_detected() or not self.has_eyes_detected():
            return "unknown"
        
        face_center = self.get_face_center()
        eye_centers = self.get_eye_centers()
        
        if face_center is None or len(eye_centers) < 2:
            return "unknown"
        
        # Calculate average eye position
        avg_eye_x = sum(eye[0] for eye in eye_centers) / len(eye_centers)
        avg_eye_y = sum(eye[1] for eye in eye_centers) / len(eye_centers)
        
        # Compare eye position to face center
        face_x, face_y = face_center
        
        # Horizontal gaze estimation
        horizontal_offset = avg_eye_x - face_x
        vertical_offset = avg_eye_y - face_y
        
        # Simple thresholding for gaze direction
        if abs(horizontal_offset) > 10:
            if horizontal_offset > 0:
                return "right"
            else:
                return "left"
        else:
            return "center"
    
    def annotated_frame(self):
        if self.frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        annotated = self.frame.copy()
        
        # Draw face rectangles
        for (x, y, w, h) in self.faces:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(annotated, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eye rectangles
        for (x, y, w, h) in self.eyes:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, "Eye", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return annotated

def main():
    print("Simple Face Tracking Demo (OpenCV-based)")
    print("This is a fallback version that doesn't require dlib")
    print("Press ESC to exit")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    # Initialize the face tracker
    tracker = SimpleFaceTracker()
    
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = webcam.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Analyze frame
            tracker.refresh(frame)
            
            # Get annotated frame
            display_frame = tracker.annotated_frame()
            
            # Add status information
            status_text = ""
            status_color = (0, 0, 255)  # Red by default
            
            if tracker.has_face_detected():
                if tracker.has_eyes_detected():
                    gaze_direction = tracker.estimate_gaze_direction()
                    status_text = f"Gaze: {gaze_direction}"
                    status_color = (0, 255, 0)  # Green for successful detection
                else:
                    status_text = "Face detected, no eyes"
                    status_color = (0, 255, 255)  # Yellow for partial detection
            else:
                status_text = "No face detected"
                status_color = (0, 0, 255)  # Red for no detection
            
            # Add text overlays
            cv2.putText(display_frame, "Simple Face Tracking (OpenCV)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, status_text, (90, 70), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2)
            
            # Add detection counts
            cv2.putText(display_frame, f"Faces: {len(tracker.faces)}", (90, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Eyes: {len(tracker.eyes)}", (90, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add session time
            session_time = int(time.time() - start_time)
            cv2.putText(display_frame, f"Session: {session_time}s", (90, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Simple Face Tracking", display_frame)
            
            # Check for ESC key
            if cv2.waitKey(1) == 27:  # ESC key
                break
                
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    
    finally:
        # Clean up
        webcam.release()
        cv2.destroyAllWindows()
        
        total_time = int(time.time() - start_time)
        print(f"\nSession completed! Total time: {total_time} seconds")
        print("Note: This was a simplified version using OpenCV face detection.")
        print("For better accuracy, consider installing dlib for full gaze tracking.")

if __name__ == "__main__":
    main()
