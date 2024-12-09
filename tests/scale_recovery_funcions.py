from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose
from src.frame_processing import compute_relative_pose, triangulate_points, frame_processing
from src.bundle_adjustment import bundle_adjustment
#from test_data.synthetic_data import points_3d, descriptors_3d, poses_gt, frames_list, video_handler, K
from test_data.synthetic_data import poses_gt
from src.video_data_handler import EndOfFramesError, VideoDataHandler
from src.plot_poses_plane import plot_poses, plot_camera_poses
from src.plot_points import plot_point_cloud
import open3d as o3d

points_3d = [
    [0, -1, 10.9],
    [1, -1, 11.10],
    [-2, -1, 10.10],
    [-1, -1, 10.12],
    [2, -1, 11.11],
    [-1, -1, 10.50],
    [0, -1, 11.14],
    [-3, -1, 10.13],
    [2, -1, 11.12],
    [-2, -1, 10.12],
    [-1, -4, 10.50],
    [0, -13, 10.60],
    [-3, -5, 10.70],
    [4, -3, 10.80],
]


descriptors_3d = [
    [1.87, 4.75, 3.66, 2.99, 0.78],
    [0.78, 0.29, 4.33, 3.01, 3.54],
    [0.1, 4.85, 4.16, 1.06, 0.91],
    [0.92, 1.52, 2.62, 2.16, 1.46],
    [3.06, 0.7, 1.46, 1.83, 2.28],
    [3.93, 1.0, 2.57, 2.96, 0.23],
    [3.04, 0.85, 0.33, 4.74, 4.83],
    [4.04, 1.52, 0.49, 3.42, 2.2],
    [0.61, 2.48, 0.17, 4.55, 1.29],
    [3.31, 1.56, 2.6, 2.73, 0.92],
    [3.97, 2.51, 2.88, 2.46, 0.98],
    [3.61, 1.4, 0.12, 3.23, 0.89],
    [4.7, 4.77, 4.57, 1.85, 0.08],
    [4.64, 2.14, 4.83, 4.82, 4.27]
    ]

points_3d = np.array(points_3d)
descriptors_3d = np.array(descriptors_3d).astype(np.float32)

pose0 = CamPose(np.eye(3), np.array([0, 0, 0]))
pose1 = CamPose(R.from_euler('x', 10, degrees=True).as_matrix(), np.array([0.5, 0, 1]))

K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]]).astype(float)


frame0 = PointDescriptors(pose0.project_into_cam(points_3d, K), descriptors_3d)
frame1 = PointDescriptors(pose1.project_into_cam(points_3d, K), descriptors_3d)
matched_frame0, matched_frame1 = frame0.points_matcher(frame1, 0.001)
pose1_est, inlier_frame0, inlier_frame1 = compute_relative_pose(matched_frame0, 
                                                                matched_frame1, K, ransac_threshold=0.01)

print(matched_frame0.points)

def extract_region_points(frame, K):

    # Extract principal point (c_x, c_y) from K
    c_x = K[0, 2]
    c_y = K[1, 2]
    image_width = int(2 * c_x)
    image_height = int(2 * c_y)
    
    # Define the bounds for filtering
    y_min = 1 * image_height / 2.5
    x_min = image_width / 4
    x_max = 3 * image_width / 4

    # Apply conditions
    mask = (frame.points[:, 1] > y_min) & (frame.points[:, 0] > x_min) & (frame.points[:, 0] < x_max)
    filtered_frame_points = frame.points[mask]
    filtered_frame_descriptors = frame.descriptors[mask]
    return PointDescriptors(filtered_frame_points, filtered_frame_descriptors)

def estimate_scale(pose1, pose2_est, inlier_frame1, inlier_frame2, K, 
                   distance_threshold = 0.2,
                   ransac_n = 5,
                   num_iterations = 100):
    
    pcd = o3d.geometry.PointCloud()

    region_frame1 = extract_region_points(inlier_frame1, K)
    region_frame2 = extract_region_points(inlier_frame2, K)
    print(region_frame2.points)
    matched_frame0, matched_frame1 = region_frame1.points_matcher(region_frame2, distance_threshold)

    points_3d_est = triangulate_points(pose1, pose2_est, matched_frame0, matched_frame1,K)
    pcd.points = o3d.utility.Vector3dVector(points_3d_est.points)

    _, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)

    points_3d_plane_est = points_3d_est.points[inliers, :]
    mean_y = np.mean(points_3d_plane_est[:,1])
    height = -1
    scale = height / mean_y

    return scale

scale = estimate_scale(pose0, pose1_est, inlier_frame0, inlier_frame1, K, 
                   distance_threshold = 1,
                   ransac_n = 5,
                   num_iterations = 100)
#print(pose1.t, pose1_est.t*scale)
#print(mean_y)
#print(pose1.t, pose1_est.scaled_pose(pose1.t).t)
#print(points_3d_est.points)

#plot_point_cloud(points_3d_est.points)

#pose1_est = pose1_est.scaled_pose(poses_gt[1].t) # Scaling with ground truth

#plot_point_cloud(points_3d)
