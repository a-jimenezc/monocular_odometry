import cv2
import numpy as np
from scipy.optimize import least_squares

def project(points_3d, camera_pose, K):
    """
    Project 3D points onto the 2D image plane.

    Args:
    - points_3d: 3D points (Nx3 array).
    - pose: Camera pose {"R": rotation matrix, "t": translation vector}.
    - K: Camera intrinsic matrix.

    Returns:
    - 2D projected points (Nx2 array).
    """
    R = camera_pose["R"]
    t = camera_pose["t"].reshape(-1, 1)
    
    # Construct the the projection matrix
    R_inv = R.T
    t_inv = -R_inv @ t
    inverse_extrinsic = np.hstack((R_inv, t_inv))
    P = K @ inverse_extrinsic

    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    projections = P @ points_homogeneous.T
    projections /= projections[2, :]  # Normalize by z- homogeneous coordinate
    return projections[:2, :].T

def reprojection_error(params, num_cameras, K, keyframes, init_pose):
    """
    Compute reprojection errors.
    Returns:
    - Residuals: Flattened reprojection error vector.
    """
    # Extract camera poses and 3D points
    camera_params = params[:num_cameras * 6].reshape((-1, 6))
    points_3d = params[num_cameras * 6:].reshape((-1, 3))
    residuals = []
    print('num_cameras:',len(camera_params))
    print('len(points_3d)', len(points_3d))
    print('len keyframes', len(keyframes))

    for i, keyframe in enumerate(keyframes):
        # Extract rotation and translation for the camera
        if i == 0:
            camera_pose = init_pose
        else:
            rvec = camera_params[i-1, :3]
            tvec = camera_params[i-1, 3:]
            R, _ = cv2.Rodrigues(rvec)
            camera_pose = {"R": R, "t": tvec}

        # Project points onto the camera
        projected_points = project(points_3d, camera_pose, K)

        # Compute residuals (difference between observed and projected points)
        residuals.extend(np.linalg.norm((projected_points - keyframe["points"]), axis=1).ravel())
    # least_squares() squares each element of the residuals vector and the performs the sumation 
    return np.array(residuals)

def bundle_adjustment(K, camera_poses, keyframes, points_3d, max_nfev=1):
    """
    Perform bundle adjustment to optimize camera poses and 3D points.
    """

    # Flatten the camera poses into [rvec, tvec] for each camera
    initial_camera_params = []
    init_pose = camera_poses[0]
    optimizing_poses = camera_poses[1:]
    for pose in optimizing_poses:
        rvec, _ = cv2.Rodrigues(pose["R"])
        initial_camera_params.append(np.hstack((rvec.ravel(), pose["t"].ravel())))

    # Flatten all parameters into a single array for optimization
    initial_params = np.hstack((np.concatenate(initial_camera_params), points_3d.ravel()))

    # Bundle adjustment using least squares optimization
    num_cameras = len(optimizing_poses)
    num_points = len(points_3d)

    result = least_squares(
        reprojection_error,
        initial_params,
        args=(num_cameras, K, keyframes, init_pose),
        method="trf",
        verbose=2,
        max_nfev=max_nfev
    )

    # Extract optimized camera poses
    optimized_params = result.x
    optimized_camera_params = optimized_params[:num_cameras * 6].reshape((-1, 6))
    optimized_points_3d = optimized_params[num_cameras * 6:].reshape((-1, 3))

    optimized_camera_poses = []
    optimized_camera_poses.append(init_pose)
    for params in optimized_camera_params:
        rvec = params[:3]
        tvec = params[3:]
        R, _ = cv2.Rodrigues(rvec)
        optimized_camera_poses.append({"R": R, "t": tvec})

    return optimized_camera_poses, optimized_points_3d
