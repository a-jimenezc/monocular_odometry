import numpy as np
from src.point_descriptors import PointDescriptors
from src.utility_functions import triangulate_points
import open3d as o3d

def extract_region_points(frame, K):

    # Extract principal point (c_x, c_y) from K
    c_x = K[0, 2]
    c_y = K[1, 2]
    image_width = int(2 * c_x)
    image_height = int(2 * c_y)
    print(image_width, image_height)
    
    # Define the bounds for filtering
    y_min = 1 * image_height / 2
    x_min = 0
    x_max = image_width

    # Apply conditions
    mask = (frame.points[:, 1] > y_min) & (frame.points[:, 0] > x_min) & (frame.points[:, 0] < x_max)
    filtered_frame_points = frame.points[mask]
    #print(mask)
    #print('filtered_frame_points',filtered_frame_points.shape)
    filtered_frame_descriptors = frame.descriptors[mask]
    return PointDescriptors(filtered_frame_points, filtered_frame_descriptors)

def estimate_scale(pose1, pose2_est, inlier_frame1, inlier_frame2, K, 
                distance_threshold = 0.2,
                ransac_n = 5,
                num_iterations = 100):
    
    pcd = o3d.geometry.PointCloud()
    #print('inlier_frame1',inlier_frame1.points.shape)
    region_frame1 = extract_region_points(inlier_frame1, K)
    region_frame2 = extract_region_points(inlier_frame2, K)
    #print('region_frame1.points',region_frame1.points.shape)
    matched_frame0, matched_frame1 = region_frame1.points_matcher(region_frame2, distance_threshold)
    #print('matched_frame0', matched_frame0.points)
    points_3d_est = triangulate_points(pose1, pose2_est, matched_frame0, matched_frame1,K)
    pcd.points = o3d.utility.Vector3dVector(points_3d_est.points)

    _, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)

    points_3d_plane_est = points_3d_est.points[inliers, :]
    mean_y = np.mean(points_3d_plane_est[:,1])
    height = -1.65
    scale = height / mean_y

    return scale
