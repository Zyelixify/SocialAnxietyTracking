# Eye Tracker for Social Anxiety

A simple app that tracks eye movements to detect signs of social anxiety.

## How it works

### 1. Data Collection (`data_acquisition.py`)
- Uses your webcam to track your eyes
- Gets pupil positions and detects blinks
- Processes video frames in real-time

### 2. Calibration (`calibration.py`) 
- Shows 5 dots on your screen
- You look at each dot for a few seconds
- Maps where you're looking to screen coordinates
- Saves calibration so you don't have to redo it

### 3. Data Analysis (`data_processing.py`)
- Analyzes your eye movements
- Tracks how often you look at the center vs edges
- Counts blinks and eye movement speed
- Generates an anxiety score

### 4. User Interface (`visualization_ui.py`)
- Shows the calibration dots
- Displays real-time tracking info
- Shows your results at the end
- Creates some charts if you want

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
