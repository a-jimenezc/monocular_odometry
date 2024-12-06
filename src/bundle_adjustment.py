import cv2
import numpy as np
from scipy.optimize import least_squares
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose

def reprojection_error(params, num_cameras, frames, init_pose, points_3d, distance_matcher, K):
    """
    Compute reprojection errors.
    Returns:
    - Residuals: Flattened reprojection error vector.
    """
    # Extract camera poses and 3D points
    camera_params = params[:num_cameras * 6].reshape((-1, 6))
    points_3d_points = params[num_cameras * 6:].reshape((-1, 3))
    points_3d = PointDescriptors(points_3d_points, points_3d.descriptors)
    residuals = []

    for i, frame in enumerate(frames):
        # Extract rotation and translation for the camera
        if i == 0:
            camera_pose = init_pose
        else:
            rvec = camera_params[i-1, :3]
            tvec = camera_params[i-1, 3:]
            R, _ = cv2.Rodrigues(rvec)
            camera_pose = CamPose(R, tvec)

        # Project points onto the camera
        matched_points_3d, matched_points_frame = points_3d.points_matcher(frame, distance_matcher)
        projected_points = camera_pose.project_into_cam(matched_points_3d.points, K)

        # Compute residuals (difference between observed and projected points)
        residuals.extend(np.linalg.norm((projected_points - matched_points_frame.points), axis=1).ravel())
    # least_squares() squares each element of the residuals vector and the performs the sumation 
    return np.array(residuals)

def bundle_adjustment(cam_poses, frames, points_3d, K, last_n=5, distance_matcher=1, max_nfev=1):
    """
    Perform bundle adjustment to optimize camera poses and 3D points.
    Modifies, cam_poses
    """
    poses_to_optimize = cam_poses[-last_n:]
    frames_to_optimize = frames[-last_n:]

    initial_camera_params = []
    init_pose = poses_to_optimize[0]
    optimizing_poses = poses_to_optimize[1:]
    for pose in optimizing_poses:
        initial_camera_params.append(pose.flatten())

    initial_params = np.hstack((np.concatenate(initial_camera_params), points_3d.points.ravel()))
    num_cameras = len(optimizing_poses)

    result = least_squares(
        reprojection_error,
        initial_params,
        args=(num_cameras, frames_to_optimize, init_pose, points_3d, distance_matcher, K),
        method="trf",
        verbose=2,
        max_nfev=max_nfev
    )

    # Extract optimized camera poses
    optimized_params = result.x
    optimized_poses_flat = optimized_params[:num_cameras * 6].reshape((-1, 6))
    optimized_points_3d_points = optimized_params[num_cameras * 6:].reshape((-1, 3))
    optimized_points_3d = PointDescriptors(optimized_points_3d_points, points_3d.descriptors)

    optimized_poses = []
    optimized_poses.append(init_pose)
    for pose_array in optimized_poses_flat:
        optimized_pose = CamPose.unflatten(pose_array)
        optimized_poses.append(optimized_pose)

    cam_poses[-last_n:] = optimized_poses

    return cam_poses, optimized_points_3d