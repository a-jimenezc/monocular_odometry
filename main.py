from src.initialization import initialize
from src.plot_points import plot_point_cloud
from src.estimate_poses import estimate_poses
import numpy as np

video_path = 'test_data/vid3.mp4'
K = np.array([
    [288.44, 0, 322.12],  
    [0, 166.078, 323.426],  
    [0,    0,   1]   
])

# See all parameters associatd with initialize
init_keyframe_poses, keyframes, optimized_points_3d, video_handler = initialize(video_path, K, max_nfev=1)

poses = estimate_poses(K, init_keyframe_poses, keyframes, optimized_points_3d, video_handler, min_points_threshold=1000)

import json

# Save to a JSON file
with open("data.json", "w") as file:
    json.dump(poses, file)

#print(poses)

#print(optimized_points)
#plot_point_cloud(optimized_points, title="Optimized 3D Point Cloud")
