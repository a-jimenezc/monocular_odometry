import numpy as np
from src.video_data_handler import VideoDataHandler
from src.visual_odometry import visual_odometry
from src.plot_poses_plane import plot_poses, plot_camera_poses
from src.plot_points import plot_point_cloud
from src.visualize_sift_on_video import visualize_sift_on_video

# Mobile robot feed
#video_path = 'test_data/vid2.mp4'
#K = np.array([[608.56811625, 0, 629.83269351],[0, 614.54502235, 346.79688358],[0, 0, 1]], dtype=np.float32)

# Kitti dataset
video_path = 'test_data/kitti_long.mp4'
K = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],[0.000000e+00, 9.808141e+02, 2.331966e+02],[0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=np.float32)

# Visual Odometry
video_handler = VideoDataHandler(video_path, grayscale=True)
points_matcher_treshold = 150
ransac_threshold = 2
init_skip = 2

poses, points_3d_est = visual_odometry(video_handler, K, points_matcher_treshold, 
                                       ransac_threshold,  init_skip, bundle_adjust=False, recover_scale=False)

# Plots
plot_poses(poses, plane='xz')
#plot_camera_poses(poses, ax=None, scale=0.1)

#plot_point_cloud(points_3d_est.points)

#visualize_sift_on_video(video_path, './output.mp4')
