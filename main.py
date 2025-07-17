import cv2
import tkinter as tk
from data_acquisition import DataAcquisition
from calibration import CalibrationModule
from data_processing import DataProcessing
from visualization_ui import VisualizationUI


class SocialAnxietyTracker:
    def __init__(self, screen_width=1920, screen_height=1080):
        # Initialize all modules
        self.data_acquisition = DataAcquisition()
        self.calibration = CalibrationModule(screen_width, screen_height)
        self.data_processing = DataProcessing(screen_width, screen_height)
        self.ui = VisualizationUI(screen_width, screen_height)
        
        # System state
        self.is_monitoring = False
        
    def run_calibration_process(self):
        print("Starting calibration...")
        
        # Check if user wants to calibrate
        if not self.ui.show_calibration_prompt():
            return False
        
        # Try to load existing calibration first
        if self.calibration.load_calibration():
            return True
        
        # Initialize camera for calibration
        try:
            self.data_acquisition.initialize_camera()
            self.data_acquisition.start_acquisition()
            
            # Run calibration with UI
            success = self.ui.show_calibration_interface(self.calibration, self.data_acquisition)
            
            if success:
                print("Calibration done!")
                return True
            else:
                print("Calibration failed")
                self.ui.show_error_message("Calibration Failed", 
                                         "Couldn't complete calibration. Want to try again?")
                return False
                
        except Exception as e:
            print(f"Calibration error: {e}")
            self.ui.show_error_message("Calibration Error", 
                                     f"Error during calibration: {str(e)}")
            return False
        finally:
            self.data_acquisition.stop_acquisition()
    
    def start_monitoring_session(self):
        # Start monitoring session
        if not self.ui.show_monitoring_prompt():
            return
        
        # Reset data processing for new session
        self.data_processing.reset_session()
        
        try:
            # Initialize camera
            self.data_acquisition.initialize_camera()
            self.data_acquisition.start_acquisition()
            self.is_monitoring = True
            
            print("Starting monitoring...")
            print("Press ESC to stop")
            
            while self.is_monitoring:
                # Get frame data from acquisition module
                frame_data = self.data_acquisition.get_frame_data()
                
                if frame_data is None:
                    break
                
                # Get gaze position from calibration module (if calibrated)
                gaze_position = None
                if self.calibration.is_calibrated and frame_data['pupils_located']:
                    left_pupil = frame_data['left_pupil']
                    right_pupil = frame_data['right_pupil']
                    gaze_tracker = self.data_acquisition.get_gaze_tracker()
                    gaze_position = self.calibration.predict_gaze_position(
                        left_pupil, right_pupil, gaze_tracker
                    )
                
                # Process frame
                self.data_processing.process_frame(frame_data, gaze_position)
                
                # Get current analysis for display
                current_analysis = self.data_processing.get_comprehensive_analysis()
                current_analysis['pupils_located'] = frame_data['pupils_located']
                
                display_frame = self.ui.create_monitoring_display(
                    frame_data['annotated_frame'], gaze_position, current_analysis
                )
                
                # Show monitoring display
                cv2.imshow("Eye Tracker", display_frame)
                
                # Check for exit
                if cv2.waitKey(1) == 27:  # ESC key
                    self.is_monitoring = False
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
            self.ui.show_error_message("Error", 
                                     f"Something went wrong: {str(e)}")
        finally:
            self.data_acquisition.cleanup()
            self._show_session_results()
    
    def _show_session_results(self):
        analysis_results = self.data_processing.get_comprehensive_analysis()
        self.ui.show_results_dialog(analysis_results)
        
        # Save visualization if possible
        try:
            fig = self.ui.create_visualization_plots(analysis_results)
            fig.savefig('results.png', dpi=150, bbox_inches='tight')
            print("Saved results as 'results.png'")
            fig.show()
        except Exception as e:
            print(f"Couldn't save plots: {e}")
    
    def run_complete_session(self):
        print("Eye Tracker")
        print("================")
        
        # Hide main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        try:
            # Step 1: Calibration
            calibration_success = self.run_calibration_process()
            
            if not calibration_success:
                self.ui.show_info_message("No Calibration!", 
                                        "Will continue without calibration.\n"
                                        "Eye tracking will be less accurate.")
            
            # Step 2: Monitoring
            self.start_monitoring_session()
            
        except Exception as e:
            print(f"App error: {e}")
            self.ui.show_error_message("Error", 
                                     f"Something went wrong: {str(e)}")
        finally:
            root.destroy()


def main():
    app = SocialAnxietyTracker()
    app.run_complete_session()


if __name__ == "__main__":
    main()
