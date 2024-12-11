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
root/
├── src/                      # Main source code
│   ├── remote/               # flask server / robot general control
│   ├── bundle_adjustment.py  # implemented bundle adjustment
│   ├── client.py             # client for remote robot control
│   ├── camera_calibration.py # 
│   ├── visual_odometry.py    # visual odometry main process
│   ├── frame_processing.py   # frame operations
│   ├── scale_recovery.py     # scale recovery implementation
│   ├── utility_functions.py  # pnp, triangulation, pose_estimation.
│   ├── video_data_handler.py # iterative video/image access handle
│   └── ...       
├── tests/                    # Unit tests
├── test_data/                # Videos for test and demo
├── scripts/                  # Scripts for automation and tasks
├── requirements.txt          # Python dependencies 
└── README.md                 # Project overview