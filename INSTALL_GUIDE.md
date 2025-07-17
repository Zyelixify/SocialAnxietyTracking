# Social Anxiety Tracking - Installation Guide

## Current Status

✅ **Successfully installed:**
- Python virtual environment 
- OpenCV (opencv-python==4.10.0.82)
- NumPy (numpy==1.26.4)
- Matplotlib (for data visualization)
- Pandas (for data analysis)
- GazeTracking source code

❌ **Requires additional setup:**
- dlib (face detection library) - requires CMake

## Quick Start Options

### Option 1: Install CMake and dlib (Recommended)

This will give you the full functionality of the GazeTracking system.

1. **Install CMake:**
   - Download CMake from: https://cmake.org/download/
   - During installation, make sure to select "Add CMake to system PATH"
   - Restart your terminal/PowerShell after installation

2. **Install dlib:**
   ```powershell
   cd "c:\Users\zyeli\Stuff\coding\University\SocialAnxietyTracking"
   & "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/pip.exe" install dlib==19.24.4
   ```

3. **Test the installation:**
   ```powershell
   & "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/python.exe" test_setup.py
   ```

4. **Run the demo:**
   ```powershell
   & "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/python.exe" main.py
   ```

### Option 2: Use Alternative Face Detection (Fallback)

If you have trouble with dlib installation, I can create a version using OpenCV's built-in face detection.

**Pros:** Easier to install, no CMake required
**Cons:** Less accurate face detection, no facial landmarks

Let me know if you'd like me to create this alternative version!

## Verify Your Installation

Run this command to check what's working:

```powershell
cd "c:\Users\zyeli\Stuff\coding\University\SocialAnxietyTracking"
& "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/python.exe" test_setup.py
```

## Project Structure

```
SocialAnxietyTracking/
├── .venv/                   # Python virtual environment ✅
├── gaze_tracking/           # GazeTracking library ✅
├── GazeTracking/           # Original repository ✅
├── main.py                  # Main demo script ✅
├── data_collector.py        # Data collection tool ✅
├── test_setup.py           # Installation test ✅
├── requirements.txt         # Dependencies ✅
├── README.md               # Documentation ✅
└── INSTALL_GUIDE.md        # This file ✅
```

## Troubleshooting

### CMake Installation Issues
- Make sure to select "Add CMake to system PATH" during installation
- Restart your terminal after installing CMake
- Test CMake with: `cmake --version`

### Camera Issues
- Make sure no other applications are using your camera
- Try running the test script to verify camera access
- On some systems, you might need to use camera index 1 instead of 0

### dlib Installation Issues
- CMake must be installed first
- On some systems, you might need Visual Studio Build Tools
- Consider using Option 2 (OpenCV alternative) if issues persist

## Next Steps

Once dlib is installed:

1. **Run the basic demo:** See real-time gaze tracking
   ```powershell
   & "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/python.exe" main.py
   ```

2. **Collect research data:** Record gaze patterns for analysis
   ```powershell
   & "C:/Users/zyeli/Stuff/coding/University/SocialAnxietyTracking/.venv/Scripts/python.exe" data_collector.py
   ```

3. **Customize for your research:** Modify the scripts for your specific needs

## Need Help?

If you're having trouble with the installation, let me know which option you'd prefer:

1. **Help with CMake/dlib installation** - I can provide more detailed troubleshooting
2. **Create OpenCV alternative** - I can make a version that doesn't require dlib
3. **Both** - Set up both versions so you have a backup

The project structure is ready - we just need to get past the dlib installation hurdle!
