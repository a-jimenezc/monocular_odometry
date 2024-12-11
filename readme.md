# Monocular Visual Odometry
![License](https://img.shields.io/badge/license-MIT-green)


- [Monocular Visual Odometry](#monocular-visual-odometry)
  - [Background](#background)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
  - [Sources](#sources)
  - [To do](#to-do)
  - [Authors](#authors)
  
---

## Background

Robot Vision course project, implementing monocular visual odometry algorithm on UGV.

The pose recovery in this project was implemented based on the procedure described in *Monocular Visual Odometry - MATLAB & Simulink* (MathWorks) [source](https://www.mathworks.com/help/vision/ug/monocular-visual-odometry.html). For scale recovery, we adopted the approach outlined by Kitt et al. in their work, "Monocular Visual Odometry Using a Planar Road Model to Solve Scale Ambiguity" (2011).


## Usage

Recomended installation using conda:

```bash
conda create -n vo_env python=3.10
conda deactivate
conda activate vo_env
pip install -r requirements.txt
python main.py
```

It runs main.py on a video from the KITTI dataset. To work with different videos, it is necessary to modify the camera's intrinsic parameters.

## Code Structure

```plaintext
root/
├── src/                        # Main source code
│   ├── remote/                 # Flask server / robot general control
│   ├── bundle_adjustment.py    # Implemented bundle adjustment
│   ├── client.py               # Client for remote robot control
│   ├── camera_calibration.py   # Camera calibration
│   ├── visual_odometry.py      # Visual odometry main process
│   ├── frame_processing.py     # Frame operations
│   ├── scale_recovery.py       # Scale recovery implementation
│   ├── utility_functions.py    # PnP, triangulation, pose estimation
│   ├── video_data_handler.py   # Iterative video/image access handle
│   └── ...                     # Other files
├── tests/                      # Testing functions with synthetic data
├── test_data/                  # Videos for test and demo
├── scale_recovery_functions.py # Scale recovery implementation
├── main.py                     # Main script
├── scripts/                    # Scripts for automation and tasks
├── requirements.txt            # Python dependencies
└── README.md                   # Project overvie
```
## Sources
- Freda, Luigi. “PySLAM V2.” *GitHub*, 18 Dec. 2022, [https://github.com/luigifreda/pyslam](https://github.com/luigifreda/pyslam).
- “Monocular Visual Odometry - MATLAB & Simulink.” *MathWorks*, [https://www.mathworks.com/help/vision/ug/monocular-visual-odometry.html](https://www.mathworks.com/help/vision/ug/monocular-visual-odometry.html).
- OpenAI. *ChatGPT* (December 9, 2024 version). OpenAI, 2024, [https://openai.com/chatgpt](https://openai.com/chatgpt).
- Kitt, Bernd & Rehder, Jörn & Chambers, Andrew & Schönbein, Miriam & Lategahn, Henning & Singh, Sanjiv. (2011). Monocular Visual Odometry using a Planar Road Model to Solve Scale Ambiguity.


## To do
* Fix scale recovery on real data.
* Implement modern SLAM Frontend/backend design.
* Loopback detection.
* Sensor fusion (IMU/Lidar).

## Authors
* Yipeng Hua
* Antonio Jimenez Caballero
