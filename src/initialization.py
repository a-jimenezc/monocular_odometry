import numpy as np
import cv2

def bundle_adjustment(K, poses, points_3d, keyframes):
    """
    Simplified bundle adjustment using OpenCV's built-in bundle adjustment API.
    
    Args:
    - K: Camera intrinsic matrix (3x3).
    - poses: Initial camera poses, list of dictionaries with "R" (rotation) and "t" (translation).
    - points_3d: Initial 3D points (Nx3 array).
    - keyframes: List of keyframes with observed 2D points.
    
    Returns:
    - optimized_poses: List of optimized poses as {"R": Rotation matrix, "t": Translation vector}.
    - optimized_points: Optimized 3D points (Nx3 array).
    """
    # Prepare data for OpenCV's bundle adjustment
    camera_matrix = K.astype(np.float64)
    rvecs = [cv2.Rodrigues(pose["R"])[0].astype(np.float64) for pose in poses]
    tvecs = [pose["t"].astype(np.float64) for pose in poses]
    object_points = points_3d.astype(np.float64)
    image_points = [kf["points"].astype(np.float64) for kf in keyframes]
    
    # Indices of corresponding 3D points for each keyframe's 2D observations
    point_indices = [np.arange(len(object_points)) for _ in keyframes]
    
    # Perform bundle adjustment using OpenCV
    retval, rvecs, tvecs, optimized_points = cv2.bundleAdjust(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        rvecs=rvecs,
        tvecs=tvecs,
        pointIndices=point_indices,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    )
    
    # Convert optimized Rodrigues vectors back to rotation matrices
    optimized_poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        R = cv2.Rodrigues(rvec)[0]
        optimized_poses.append({"R": R, "t": tvec})
    
    return optimized_poses, optimized_points

def pose_initialization(K, keyframes):
    """
    Pose initialization for three keyframes using the provided intrinsic matrix and keyframes.
    """
    # Ensure at least 3 keyframes are provided
    if len(keyframes) < 3:
        raise ValueError("At least 3 keyframes are required for pose initialization.")

    # Convert intrinsic matrix to float64
    K = K.astype(np.float64)

    # Compute essential matrices and poses
    def compute_essential_matrix(pts1, pts2, K):
        F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        E = K.T @ F @ K
        return E, inliers.ravel() > 0

    def recover_pose(E, pts1, pts2, K):
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        return R, t, mask.ravel() > 0

    def triangulate_points(P1, P2, pts1, pts2):
        points_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_homogeneous[:3] / points_homogeneous[3]
        return points_3d.T

    # Compute essential matrices
    E1, inliers1 = compute_essential_matrix(keyframes[0]["points"], keyframes[1]["points"], K)
    E2, inliers2 = compute_essential_matrix(keyframes[0]["points"], keyframes[2]["points"], K)
    R2, t2, mask_2 = recover_pose(E2, keyframes[0]["points"], keyframes[2]["points"], K)

    # Find common inliers
    common_inliers = inliers1 & inliers2
    pts1_common = keyframes[0]["points"][common_inliers]
    pts2_common = keyframes[1]["points"][common_inliers]
    pts3_common = keyframes[2]["points"][common_inliers]
    assert len(pts1_common) == len(pts2_common) == len(pts3_common)

    # Triangulate points
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R2, t2))
    points_3d = triangulate_points(P0, P2, pts1_common, pts3_common)

    # Pose estimation for K1
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, pts2_common, K, None)
    R1_refined, _ = cv2.Rodrigues(rvec)
    t1_refined = tvec

    # Return unoptimized poses and points (placeholder for bundle adjustment)
    poses = [{"R": np.eye(3), "t": np.zeros((3,))}, {"R": R1_refined, "t": t1_refined}, {"R": R2, "t": t2}]
    
    #optimized_poses, optimized_points = bundle_adjustment(K, poses, points_3d)
    #return optimized_poses, optimized_points
    return poses, points_3d