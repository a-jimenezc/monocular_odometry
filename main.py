from src.initialization import initialize
import numpy as np

video_path = 'test_data/vid3.mp4'
K = np.array([
    [288.44, 0, 322.12],  
    [0, 166.078, 323.426],  
    [0,    0,   1]   
])
optimized_poses, optimized_points = initialize(video_path, K)
print(optimized_poses)
