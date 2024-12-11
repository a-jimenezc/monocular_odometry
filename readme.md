# Monocular Visual Odometry
![License](https://img.shields.io/badge/license-MIT-green)


- [Monocular Visual Odometry](#monocular-visual-odometry)
  - [Background](#background)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
  
---

## Background

Robot Vision course project, implementing monocular visual odometry algorithm on UGV. 

## Usage
pip install -r requirements.txt
python3 main.py

## Code Structure

The project is organized as follows:

- **`src/`**: Main source code  
  - **`remote/`**: Flask server and robot general control  
  - **`bundle_adjustment.py`**: Bundle adjustment implementation  
  - **`client.py`**: Client for remote robot control  
  - **`camera_calibration.py`**: Camera calibration utilities  
  - **`visual_odometry.py`**: Visual odometry main process  
  - **`frame_processing.py`**: Frame processing operations  
  - **`scale_recovery.py`**: Scale recovery implementation  
  - **`utility_functions.py`**: Helper functions (e.g., PnP, triangulation, pose estimation)  
  - **`video_data_handler.py`**: Video/image data handler for iterative access  
  - **`...`**: Additional modules  

- **`tests/`**: Unit tests  
- **`test_data/`**: Test and demo video files  
- **`scripts/`**: Automation and task scripts  
- **`requirements.txt`**: Python dependencies  
- **`README.md`**: Project overview and documentation  
