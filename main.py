from src.initialization import initialize
from src.plot_points import plot_point_cloud
from src.estimate_poses import estimate_poses
from src.plot_poses_plane import plot_poses
import numpy as np

video_path = 'test_data/vid7.avi'
K = np.array([[608.56811625, 0, 629.83269351],[0, 614.54502235, 346.79688358],[0, 0, 1]], dtype=np.float32)
#video_path = 'test_data/vid5.mp4'
#K = np.array([[288.44, 0, 322.12],  [0, 166.078, 323.426],  [0,    0,   1]   ], dtype=np.float32)
if K.shape != (3, 3) or np.linalg.det(K) == 0:
    raise ValueError("Invalid intrinsic matrix provided.")

# See all parameters associatd with initialize
unoptimized_poses, init_keyframe_poses, keyframes, optimized_points_3d, video_handler = initialize(
    video_path, K, max_nfev=10)



print('init 3d points', len(optimized_points_3d['points_3d']))
#plot_poses(unoptimized_poses)
#plot_poses(init_keyframe_poses)
#print(init_keyframe_poses)
poses = estimate_poses(K, init_keyframe_poses, keyframes, 
                       optimized_points_3d, video_handler, min_points_threshold=10, max_nfev=10)

plot_poses(poses, plane='xz')

#print(poses)

#print(optimized_points)
#plot_point_cloud(optimized_points, title="Optimized 3D Point Cloud")
