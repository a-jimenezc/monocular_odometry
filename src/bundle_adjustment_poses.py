from scipy.optimize import least_squares
import numpy as np
import cv2

# Define the reprojection error function
def reprojection_error(params, num_cameras, num_points, K, observations):
    """
    Compute the reprojection error for bundle adjustment.
    """
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    points_3d = params[num_cameras * 6:].reshape((num_points, 3))
    residuals = []

    for i, keyframe in enumerate(observations):
        # Extract camera parameters
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:]
        R, _ = cv2.Rodrigues(rvec)

        # Project points into the current keyframe
        P = K @ np.hstack((R, tvec.reshape(-1, 1)))
        points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        projections = P @ points_homogeneous.T
        projections /= projections[2, :]  # Normalize by z-coordinate
        projected_points = projections[:2, :].T

        # Compute residuals between observed and projected points
        residuals.extend((projected_points - keyframe["points"]).ravel())

    return np.array(residuals)

def bundle_adjustment(K, window_keyframe_poses, window_keyframes, points_descriptors_3d, m, max_nfev=None):
    """
    Perform bundle adjustment on a sliding window of keyframes, optimizing only the most recent `m` keyframes.

    Args:
    - K: Camera intrinsic matrix.
    - window_keyframe_poses: List of poses within the sliding window, each as {"R": rotation matrix, "t": translation vector}.
    - window_keyframes: List of keyframes in the sliding window, each containing 2D points and descriptors.
    - points_descriptors_3d: Dictionary with "points_3d" (Nx3) and "descriptors_3d".
    - m: Number of most recent keyframes to optimize.
    - max_nfev: Maximum number of function evaluations for the optimizer.

    Returns:
    - optimized_window_keyframe_poses: Updated poses for the sliding window.
    - optimized_points_descriptors_3d: Updated 3D points and descriptors.
    """
    # Extract 3D points
    points_3d = points_descriptors_3d["points_3d"]
    descriptors_3d = points_descriptors_3d["descriptors_3d"]

    # Prepare parameters for optimization
    num_keyframes = len(window_keyframe_poses)
    fixed_keyframes = num_keyframes - m  # Number of fixed keyframes

    # Flatten camera poses for the optimizer
    camera_params = []
    for i, pose in enumerate(window_keyframe_poses):
        rvec, _ = cv2.Rodrigues(pose["R"])
        camera_params.append(np.hstack((rvec.ravel(), pose["t"].ravel())))
    camera_params = np.array(camera_params)

    # Flatten all parameters for optimization
    initial_params = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # Observations: Collect 2D-3D correspondences for each keyframe
    observations = []
    for keyframe in window_keyframes:
        observations.append({"points": keyframe["points"]})

    num_cameras = len(window_keyframe_poses)
    num_points = points_3d.shape[0]

    # Perform optimization
    result = least_squares(
        reprojection_error,
        initial_params,
        args=(num_cameras, num_points, K, observations),
        method="lm",
        max_nfev=max_nfev,
        verbose=2
    )

    # Extract optimized parameters
    optimized_params = result.x
    optimized_camera_params = optimized_params[:num_cameras * 6].reshape((num_cameras, 6))
    optimized_points_3d = optimized_params[num_cameras * 6:].reshape((num_points, 3))

    # Convert optimized camera parameters back to pose format
    optimized_window_keyframe_poses = []
    for i, params in enumerate(optimized_camera_params):
        rvec = params[:3]
        tvec = params[3:]
        R, _ = cv2.Rodrigues(rvec)
        optimized_window_keyframe_poses.append({"R": R, "t": tvec})

    # Return optimized poses and updated points with descriptors
    optimized_points_descriptors_3d = {
        "points_3d": optimized_points_3d,
        "descriptors_3d": descriptors_3d  # Descriptors remain unchanged
    }

    return optimized_window_keyframe_poses, optimized_points_descriptors_3d
