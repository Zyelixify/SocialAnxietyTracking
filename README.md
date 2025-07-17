# Social Anxiety Tracker

A modular application for tracking social anxiety indicators through gaze and eye movement analysis.

## Module Architecture

### 1. Data Acquisition Module (`data_acquisition.py`)
**Purpose**: Captures video from camera using OpenCV's CV2
- Accesses webcam and initializes GazeTracking
- Extracts raw pupil and gaze data from video frames
- Provides real-time frame processing with timestamp information
- Returns structured data including pupil coordinates, gaze ratios, and blink status

### 2. Calibration Module (`calibration.py`) 
**Purpose**: Maps pupil coordinates to screen coordinates
- Displays 5 calibration points on screen (center + 4 corners)
- Holds gaze at each point for either 3 seconds OR until 10 samples collected
- Maps approximate screen location based on gaze information
- Uses outlier filtering and weighted averaging for precision
- Saves/loads calibration data for reuse

### 3. Data Processing Module (`data_processing.py`)
**Purpose**: Preprocesses and analyzes gaze/pupil data for therapy insights
- Smoothes collected data from acquisition module
- Compares data with calibrated screen information
- Calculates frequency and accuracy of center screen focus
- Tracks look-away patterns, pupillary changes, blink analysis
- Generates comprehensive anxiety indicators and scoring

### 4. Visualization and UI Module (`visualization_ui.py`)
**Purpose**: Displays information clearly using Tkinter
- Backs calibration module with calibration point display
- Shows real-time monitoring interface with metrics
- Displays test results with personalized encouragement messages
- Creates matplotlib visualizations of session data
- Provides user-friendly dialogs and feedback

## Usage

Run the complete application:
```bash
python main.py
```

Or use individual modules programmatically:
```python
from data_acquisition import DataAcquisition
from calibration import CalibrationModule
from data_processing import DataProcessing
from visualization_ui import VisualizationUI

# Initialize modules
data_acq = DataAcquisition()
calibration = CalibrationModule()
processing = DataProcessing()
ui = VisualizationUI()

# Use modules as needed...
```

## Features

- **5-Point Precision Calibration**: Maps eye movements to screen coordinates
- **Real-time Anxiety Detection**: Analyzes blink patterns, gaze velocity, focus areas
- **Comprehensive Metrics**: Center focus accuracy, look-away frequency, saccade detection
- **Personalized Feedback**: Encouragement messages based on performance
- **Data Visualization**: Charts and graphs showing session results
- **Modular Design**: Each component can be used independently

## Requirements

- Python 3.11+
- dlib (for precise pupil tracking)
- Install Python dependencies through `requirements.txt` 

## Module Dependencies

```
main.py
├── data_acquisition.py (OpenCV, GazeTracking)
├── calibration.py (NumPy, JSON)
├── data_processing.py (NumPy, Collections)
└── visualization_ui.py (Tkinter, Matplotlib, CV2)
```

Each module is designed to be independent with clear interfaces between components.
