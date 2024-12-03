from src.initialization import initialize
from src.plot_points import plot_point_cloud
#from src.estimate_poses import estimate_poses
from src.estimate_poses_1 import estimate_poses
from src.plot_poses_plane import plot_poses
from src.video_data_handler import VideoDataHandler
import numpy as np
import cv2

video_path = 'test_data/vid9.avi'
#video_path = "http://10.0.0.90:5000/video_feed"

K = np.array([[608.56811625, 0, 629.83269351],[0, 614.54502235, 346.79688358],[0, 0, 1]], dtype=np.float32)
#video_path = 'test_data/vid5.mp4'
#K = np.array([[288.44, 0, 322.12],  [0, 166.078, 323.426],  [0,    0,   1]   ], dtype=np.float32)

if K.shape != (3, 3) or np.linalg.det(K) == 0:
    raise ValueError("Invalid intrinsic matrix provided.")

video_handler = VideoDataHandler(video_path, grayscale=True)

R_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

init_pose = {"R" : np.eye(3), "t" : np.zeros((3, 1))}

feature_detector = cv2.SIFT_create()
i=1
for frame in video_handler:
    keypoints, descriptors = feature_detector.detectAndCompute(frame, None)
    points = np.array([kp.pt for kp in keypoints])
    if points.shape[0] < 150: # Ensure minimun number of points
        continue

    init_keyframe = {
        "points": points,
        "descriptors": descriptors
    }
    break

# See all parameters associatd with initialize
init_keyframe_poses, keyframes, optimized_points_3d, video_handler = initialize(
    init_keyframe, init_pose, video_handler, K, max_nfev=10)

poses = estimate_poses(K, init_keyframe_poses, keyframes, 
                       optimized_points_3d, video_handler, min_points_threshold=6, max_nfev=10)

plot_poses(poses, plane='xz')
