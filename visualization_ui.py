import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class VisualizationUI:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.root = None
        self.canvas = None
        
    def show_calibration_interface(self, calibration_module, data_acquisition):
        self.root = tk.Tk()
        self.root.title("Eye Calibration")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        
        self.canvas = tk.Canvas(self.root, width=self.screen_width, height=self.screen_height, bg='black')
        self.canvas.pack()
        
        points = calibration_module.get_calibration_points()
        successful_points = 0
        
        for i, (x, y) in enumerate(points):
            print(f"Calibrating point {i+1}/5 at ({x}, {y})")
            
            self._display_calibration_point(x, y, i+1)
            
            # Wait for user to focus
            time.sleep(3)
            
            samples = calibration_module.collect_samples_for_point(data_acquisition, x, y)
            
            # Process samples
            if calibration_module.process_calibration_point(samples, x, y):
                successful_points += 1
                print(f"  Point {i+1} calibrated! Got {len(samples)} samples")
                self._show_success_feedback(x, y)
            else:
                print(f"  Point {i+1} failed calibration")
                self._show_failure_feedback(x, y)
            
            time.sleep(0.5)  # Brief pause between points
        
        self.root.destroy()
        return calibration_module.complete_calibration(successful_points)
    
    def _display_calibration_point(self, x, y, point_num):
        self.canvas.delete("all")
        
        # Draw calibration point
        self.canvas.create_oval(x-25, y-25, x+25, y+25, fill='red', outline='white', width=3)
        
        self.canvas.create_text(x, y-70, text=f"Focus on the dot:\nPoint {point_num}/5", 
                               fill='white', font=('Arial', 18), justify='center')
        self.canvas.create_text(x, y+70, text="Try to keep your head still", 
                               fill='gray', font=('Arial', 12), justify='center')
        
        # Add progress indicator
        self.canvas.create_text(self.screen_width//2, 50, 
                               text=f"Progress: {point_num}/5", 
                               fill='white', font=('Arial', 16))
        
        self.root.update()
    
    def _show_success_feedback(self, x, y):
        self.canvas.create_text(x, y+100, text="✓ Success!", 
                               fill='green', font=('Arial', 14, 'bold'))
        self.root.update()
    
    def _show_failure_feedback(self, x, y):
        self.canvas.create_text(x, y+100, text="✗ Failed", 
                               fill='red', font=('Arial', 14, 'bold'))
        self.root.update()
    
    def show_calibration_prompt(self):
        root = tk.Tk()
        root.withdraw()
        
        response = messagebox.askyesno("Calibration", 
                                     "You'll see 5 dots on the screen.\n"
                                     "Focus on each dot as it appears.\n\n"
                                     "Ready to start?")
        root.destroy()
        return response
    
    def create_monitoring_display(self, frame, gaze_position, analysis_data):
        display_frame = frame.copy()
        
        # Status info - make it more casual
        status_color = (0, 255, 0) if analysis_data.get('pupils_located', False) else (0, 0, 255)
        status_text = "Eyes: " + ("OK" if analysis_data.get('pupils_located', False) else "Lost")
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if gaze_position:
            cv2.putText(display_frame, f"Looking at: ({gaze_position[0]}, {gaze_position[1]})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(display_frame, "Looking at: Unknown", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        # Real-time metrics
        if 'blink_count' in analysis_data:
            cv2.putText(display_frame, f"Blinks: {analysis_data['blink_count']} ({analysis_data.get('blink_rate', 0):.1f}/min)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if 'center_gaze_ratio' in analysis_data:
            cv2.putText(display_frame, f"Center focus: {analysis_data['center_gaze_ratio']:.1%}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if 'assessment' in analysis_data:
            assessment_color = (0, 255, 0) if analysis_data.get('anxiety_score', 0) == 0 else (0, 165, 255)
            cv2.putText(display_frame, f"Assessment: {analysis_data['assessment']}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, assessment_color, 1)
        
        return display_frame
    
    def show_results_dialog(self, analysis_data):
        root = tk.Tk()
        root.title("How did I do?")
        root.geometry("600x700")
        
        # Create scrollable text widget
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        result_text = self._format_results_text(analysis_data)
        text_widget = tk.Text(frame, wrap=tk.WORD, width=70, height=35)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert(tk.END, result_text)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        encouragement = self._generate_encouragement_message(analysis_data)
        
        encouragement_label = ttk.Label(frame, text=encouragement, 
                                       font=('Arial', 11, 'bold'),
                                       foreground='blue', wraplength=550)
        encouragement_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Close button
        close_button = ttk.Button(frame, text="Close", command=root.destroy)
        close_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        root.mainloop()
    
    def _format_results_text(self, analysis_data):
        result = f"Session Results:\n"
        result += f"Duration: {analysis_data.get('session_duration', 0):.1f} minutes\n\n"
        result += f"Assessment: {analysis_data.get('assessment', 'Unknown')}\n"
        result += f"Anxiety Score: {analysis_data.get('anxiety_score', 0)}/{analysis_data.get('max_score', 15)}\n\n"
        
        result += f"Detailed Metrics:\n"
        result += f"• Blinks: {analysis_data.get('blink_count', 0)} total ({analysis_data.get('blink_rate', 0):.1f}/min)\n"
        result += f"• Avg Blink Duration: {analysis_data.get('avg_blink_duration', 0):.3f}s\n"
        result += f"• Center Focus: {analysis_data.get('center_gaze_ratio', 0):.1%}\n"
        result += f"• Center Accuracy: {analysis_data.get('center_gaze_accuracy', 0):.1%}\n"
        result += f"• Edge Focus: {analysis_data.get('edge_gaze_ratio', 0):.1%}\n"
        result += f"• Look-away Frequency: {analysis_data.get('look_away_frequency', 0):.1f}/min\n"
        result += f"• Saccade Rate: {analysis_data.get('saccade_rate', 0):.1f}/min\n"
        result += f"• Avg Gaze Velocity: {analysis_data.get('avg_gaze_velocity', 0):.0f} px/s\n"
        result += f"• Total Gaze Points: {analysis_data.get('total_gaze_positions', 0)}\n\n"
        
        indicators = analysis_data.get('indicators', [])
        if indicators:
            result += "Notable concerns:\n"
            for indicator in indicators:
                result += f"• {indicator}\n"
        else:
            result += "Nothing concerning detected.\n"
        
        return result
    
    def _generate_encouragement_message(self, analysis_data):
        anxiety_score = analysis_data.get('anxiety_score', 0)
        max_score = analysis_data.get('max_score', 15)
        center_focus = analysis_data.get('center_gaze_ratio', 0)
        
        if anxiety_score == 0:
            return "Nice job! You stayed pretty calm throughout the session."
        
        elif anxiety_score <= 3:
            return "Pretty good! Only a few minor signs of stress."
        
        elif anxiety_score <= 6:
            if center_focus > 0.5:
                return "Good! Even with some stress, you kept looking at the center."
            else:
                return "It's totally normal to feel anxious. Maybe try to focus on breathing?"
        
        elif anxiety_score <= 10:
            return "Hang in there! Everyone gets anxious sometimes. Practice makes it easier."
        
        else:
            return "Thanks you for your time! Remember, if you're feeling anxious, talking to someone might help."
    
    def show_monitoring_prompt(self):
        root = tk.Tk()
        root.withdraw()
        
        response = messagebox.askyesno("Start Monitoring", 
                                     "Ready to start?\n\n"
                                     "I'll track your eye movements.\n"
                                     "Hit ESC to stop anytime.")
        root.destroy()
        return response
    
    def show_error_message(self, title, message):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    
    def show_info_message(self, title, message):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    
    def create_visualization_plots(self, analysis_data):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Social Anxiety Tracking Results', fontsize=16)
        
        # Anxiety score visualization
        anxiety_score = analysis_data.get('anxiety_score', 0)
        max_score = analysis_data.get('max_score', 15)
        
        ax1.bar(['Current Session'], [anxiety_score], color='lightcoral' if anxiety_score > 5 else 'lightgreen')
        ax1.axhline(y=max_score/3, color='yellow', linestyle='--', alpha=0.7, label='Mild threshold')
        ax1.axhline(y=2*max_score/3, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
        ax1.set_ylabel('Anxiety Score')
        ax1.set_title('Anxiety Level Assessment')
        ax1.set_ylim(0, max_score)
        ax1.legend()
        
        # Gaze focus distribution
        center_ratio = analysis_data.get('center_gaze_ratio', 0)
        edge_ratio = analysis_data.get('edge_gaze_ratio', 0)
        other_ratio = 1 - center_ratio - edge_ratio
        
        labels = ['Center Focus', 'Edge Focus', 'Other Areas']
        sizes = [center_ratio, edge_ratio, other_ratio]
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Gaze Distribution')
        
        # Blink analysis
        blink_rate = analysis_data.get('blink_rate', 0)
        normal_range = [15, 20]  # Normal blink rate range
        
        ax3.bar(['Current', 'Normal Min', 'Normal Max'], 
                [blink_rate, normal_range[0], normal_range[1]], 
                color=['lightcoral' if blink_rate > 30 or blink_rate < 8 else 'lightgreen', 'gray', 'gray'])
        ax3.set_ylabel('Blinks per Minute')
        ax3.set_title('Blink Rate Analysis')
        
        # Session metrics
        metrics = ['Center Accuracy', 'Blink Rate Normal', 'Low Saccades', 'Session Quality']
        
        center_acc = analysis_data.get('center_gaze_accuracy', 0)
        blink_normal = 1.0 if 8 <= blink_rate <= 30 else 0.5
        saccade_normal = 1.0 if analysis_data.get('saccade_rate', 0) <= 6 else 0.5
        session_quality = (center_acc + blink_normal + saccade_normal) / 3
        
        values = [center_acc, blink_normal, saccade_normal, session_quality]
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.4 else 'red' for v in values]
        
        ax4.barh(metrics, values, color=colors)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel('Quality Score (0-1)')
        ax4.set_title('Session Quality Metrics')
        
        plt.tight_layout()
        return fig
