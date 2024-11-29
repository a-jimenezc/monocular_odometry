from scipy.optimize import least_squares
import numpy as np
import cv2

def reprojection_error(params, K, window_keyframes, fixed_poses, m, descriptors_3d, matches):
    """
    Compute the reprojection error for bundle adjustment, optimizing only the most recent keyframes.
    """
    # Extract optimized camera parameters and 3D points
    optimized_camera_params = params[:m * 6].reshape((-1, 6))
    points_3d = params[m * 6:].reshape((-1, 3))
    residuals = []

    if len(points_3d) != len(descriptors_3d):
        raise ValueError("Mismatch points_3d descriptors_3d")

    for i, keyframe in enumerate(window_keyframes):
        if i < len(fixed_poses):
            # Use fixed poses for older keyframes
            R = fixed_poses[i]["R"]
            t = fixed_poses[i]["t"]
        else:
            # Use optimized poses for recent keyframes
            rvec = optimized_camera_params[i - len(fixed_poses), :3]
            tvec = optimized_camera_params[i - len(fixed_poses), 3:]
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

        # Project points into the current keyframe
        R_inv = R.T
        t_inv = -R_inv @ t.reshape(-1, 1)
        P = K @ np.hstack((R_inv, t_inv))
        points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        projections = P @ points_homogeneous.T
        projections /= projections[2, :]  # Normalize by z-coordinate
        projected_points = projections[:2, :].T

        if len(matches[i]) == 0:
            print('No bf match')
            continue  # Skip keyframe if no matches

        matched_projected_points = np.array([projected_points[m.queryIdx] for m in matches[i]])
        matched_keyframe_points = np.array([keyframe["points"][m.trainIdx] for m in matches[i]])


        # Compute residuals for the matched points
        residuals.extend((matched_projected_points - matched_keyframe_points).ravel())

    return np.array(residuals)

def precompute_matches(descriptors_3d, window_keyframes):
    """
    Precompute descriptor matches between 3D descriptors and keyframe descriptors.

    Args:
    - descriptors_3d: 3D descriptors (NxD array).
    - window_keyframes: List of keyframes, each with "descriptors" and "points".

    Returns:
    - matches_list: List of matches for each keyframe.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_list = []
    for keyframe in window_keyframes:
        matches = bf.match(descriptors_3d, keyframe["descriptors"])
        matches_list.append(matches)
    return matches_list

def bundle_adjustment(K, window_keyframe_poses, window_keyframes, points_descriptors_3d, m, max_nfev=1):
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

    num_cameras = len(window_keyframe_poses)
    num_optimized_cameras = len(window_keyframe_poses[-m:])
    num_fixed_poses = len(window_keyframe_poses[:-m])

    # Separating keyframes
    fixed_poses = window_keyframe_poses[:-m]

    optimized_camera_params = []
    for pose in window_keyframe_poses[-m:]:
        rvec, _ = cv2.Rodrigues(pose["R"])
        tvec = pose["t"].ravel()
        optimized_camera_params.append(np.hstack((rvec.ravel(), tvec)))

    initial_params = np.hstack((np.concatenate(optimized_camera_params), points_3d.ravel()))

    matches_list = precompute_matches(descriptors_3d, window_keyframes)

    result = least_squares(
        reprojection_error,
        initial_params,
        args=(K, window_keyframes, fixed_poses, m, descriptors_3d, matches_list),
        method="trf",
        max_nfev=max_nfev,
        verbose=2
    )

    # Extract optimized parameters
    optimized_params = result.x
    optimized_camera_params = optimized_params[:num_optimized_cameras * 6].reshape((-1, 6))
    optimized_points_3d = optimized_params[num_optimized_cameras * 6:].reshape((-1, 3))

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
