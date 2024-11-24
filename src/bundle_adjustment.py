import cv2
import numpy as np
from scipy.optimize import least_squares

def project(points_3d, pose, K):
    """
    Project 3D points onto the 2D image plane.

    Args:
    - points_3d: 3D points (Nx3 array).
    - pose: Camera pose {"R": rotation matrix, "t": translation vector}.
    - K: Camera intrinsic matrix.

    Returns:
    - 2D projected points (Nx2 array).
    """
    R = pose["R"]
    t = pose["t"].reshape(-1, 1)
    
    # Construct the inverse extrinsic matrix
    R_inv = R.T
    t_inv = -R_inv @ t
    inverse_extrinsic = np.hstack((R_inv, t_inv))
    
    # Compute the projection matrix
    P = K @ inverse_extrinsic

    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    projections = P @ points_homogeneous.T
    projections /= projections[2, :]  # Normalize by z-coordinate
    return projections[:2, :].T

def reprojection_error(params, num_cameras, num_points, K, observations):
    """
    Compute reprojection errors.

    Args:
    - params: Flattened camera poses and 3D points.
    - num_cameras: Number of cameras/keyframes.
    - num_points: Number of 3D points.
    - K: Camera intrinsic matrix.
    - observations: Observed 2D points in keyframes.

    Returns:
    - Residuals: Flattened reprojection error vector.
    """
    # Extract camera poses and 3D points
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    points_3d = params[num_cameras * 6:].reshape((num_points, 3))
    residuals = []

    for i, keyframe in enumerate(observations):
        # Extract rotation and translation for the camera
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:]
        R, _ = cv2.Rodrigues(rvec)

        pose = {"R": R, "t": tvec}

        # Project points onto the camera
        projected_points = project(points_3d, pose, K)

        # Compute residuals (difference between observed and projected points)
        residuals.extend((projected_points - keyframe["points"]).ravel())

    return np.array(residuals)

def bundle_adjustment(K, camera_poses, points_3d, keyframes):
    """
    Perform bundle adjustment to optimize camera poses and 3D points.

    Args:
    - K: Camera intrinsic matrix (3x3).
    - camera_poses: List of camera poses, each as {"R": rotation matrix, "t": translation vector}.
    - points_3d: Initial 3D points (Nx3 array).
    - keyframes: List of keyframes, each containing observed 2D points.

    Returns:
    - optimized_camera_poses: Refined camera poses as {"R": rotation matrix, "t": translation vector}.
    - optimized_points_3d: Refined 3D points (Nx3 array).
    """

    # Flatten the camera poses into [rvec, tvec] for each camera
    initial_camera_params = []
    for pose in camera_poses:
        rvec, _ = cv2.Rodrigues(pose["R"])
        #initial_camera_params.append(np.hstack((rvec.ravel(), pose["t"])))
        initial_camera_params.append(np.hstack((rvec.ravel(), pose["t"].ravel())))

    # Flatten all parameters into a single array for optimization
    initial_params = np.hstack((np.concatenate(initial_camera_params), points_3d.ravel()))

    # Bundle adjustment using least squares optimization
    num_cameras = len(camera_poses)
    num_points = points_3d.shape[0]
    observations = keyframes

    result = least_squares(
        reprojection_error,
        initial_params,
        args=(num_cameras, num_points, K, observations),
        method="trf",
        verbose=2
    )

    # Extract optimized camera poses
    optimized_params = result.x
    optimized_camera_params = optimized_params[:num_cameras * 6].reshape((num_cameras, 6))
    optimized_points_3d = optimized_params[num_cameras * 6:].reshape((num_points, 3))

    optimized_camera_poses = []
    for params in optimized_camera_params:
        rvec = params[:3]
        tvec = params[3:]
        R, _ = cv2.Rodrigues(rvec)
        optimized_camera_poses.append({"R": R, "t": tvec})

    return optimized_camera_poses, optimized_points_3d
