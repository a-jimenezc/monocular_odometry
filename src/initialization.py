import cv2
import numpy as np
from scipy.optimize import least_squares
from src.video_data_handler import VideoDataHandler
from src.bundle_adjustment import bundle_adjustment

def initial_keyframes(video_handler, threshold = 3):

    # compute initial 3 keyframes
    feature_detector = cv2.SIFT_create()
    previous_descriptors = None
    keyframes = []
    for frame in video_handler:
        keypoints, descriptors = feature_detector.detectAndCompute(frame, None)
        
        # Compute similarity with previous frame's descriptors
        if previous_descriptors is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(previous_descriptors, descriptors)
            match_distances = [m.distance for m in matches]
            if np.mean(match_distances) < threshold:
                continue
        
        # Store keyframe data
        points = np.array([kp.pt for kp in keypoints])
        if points.shape[0] < 1000:
            continue
        keyframes.append({
            "points": points,
            "descriptors": descriptors
        })
        
        previous_descriptors = descriptors
        
        # Limit to 3 keyframes
        if len(keyframes) >= 3:
            break
    return keyframes

def keyframe_matcher(keyframe1, keyframe2, distance_threshold=50.0):
    if keyframe1["descriptors"] is None or keyframe2["descriptors"] is None:
        raise TypeError("Descriptors must not be None.")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_not_filtered = bf.match(keyframe1["descriptors"], keyframe2["descriptors"])

    # Filter matches by distance threshold
    matches = [m for m in matches_not_filtered if m.distance < distance_threshold] # Check threshold

    # Extract matched points and descriptors
    matched_points1 = np.array([keyframe1["points"][m.queryIdx] for m in matches])
    matched_descriptors1 = np.array([keyframe1["descriptors"][m.queryIdx] for m in matches])

    matched_points2 = np.array([keyframe2["points"][m.trainIdx] for m in matches])
    matched_descriptors2 = np.array([keyframe2["descriptors"][m.trainIdx] for m in matches])

    # Return matched keyframes
    matched_keyframe1 = {"points": matched_points1, "descriptors": matched_descriptors1}
    matched_keyframe2 = {"points": matched_points2, "descriptors": matched_descriptors2}

    return matched_keyframe1, matched_keyframe2

def compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=3.0):
    """
    Compute the essential matrix between two keyframes using their feature points and descriptors.
    Returns:
    - E: Essential matrix (3x3).
    - inliers: Boolean mask of inliers used for the computation.
    """

    # Compute the fundamental and essential matrix
    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    if len(matched_keyframe1["points"]) < 5 or len(matched_keyframe2["points"]) < 5:
        raise ValueError("Insufficient points for essential matrix computation (minimum 5 points required).")
    if K.shape != (3, 3) or np.linalg.det(K) == 0:
        raise ValueError("Invalid intrinsic matrix provided.")
    
    F, inliers = cv2.findFundamentalMat(matched_keyframe1["points"],
                                        matched_keyframe2["points"], 
                                        cv2.FM_RANSAC, 
                                        ransac_threshold)

    E = K.T @ F @ K

    # Compute pose
    retval, R, t, mask = cv2.recoverPose(E, 
                                         matched_keyframe1["points"], 
                                         matched_keyframe2["points"], 
                                         K)

    inliers = inliers.ravel() > 0
    inlier_keyframe1 = {
        "points": matched_keyframe1["points"][inliers],
        "descriptors": matched_keyframe1["descriptors"][inliers],
    }
    
    inlier_keyframe2 = {
        "points": matched_keyframe2["points"][inliers],
        "descriptors": matched_keyframe2["descriptors"][inliers],
    }


    return E, R, t, inlier_keyframe1, inlier_keyframe2

def align_keyframes(keyframe0, keyframe1, keyframe2):
    """
    Retains only common points.
    Assumes there is a single correspondence based on descriptors.
    """
    # Initial matching
    for i, keyframe in enumerate([keyframe0, keyframe1, keyframe2]):
        if keyframe["points"].shape[0] < 2:
            raise ValueError(f"Keyframe{i} is empty. All keyframes must contain points and descriptors.")
        
    matched_keyframe0, matched_keyframe2 = keyframe_matcher(keyframe0, keyframe2)
    if matched_keyframe0["points"].shape[0] < 2:
        raise ValueError("No matches found between keyframe0 and keyframe2.")

    # Alignment
    matched_keyframe0_a, matched_keyframe1 = keyframe_matcher(matched_keyframe0, keyframe1)
    if matched_keyframe0_a["points"].shape[0] < 2:
        raise ValueError("No matches found between aligned keyframe0 and keyframe1.")

    # Sanity Check
    rechecked_keyframe0, rechecked_keyframe2 = keyframe_matcher(matched_keyframe0_a, matched_keyframe2)
    if rechecked_keyframe0["points"].shape != matched_keyframe0_a["points"].shape:
        raise ValueError("Mismatch in aligned points between iterations.")

    return matched_keyframe0_a, matched_keyframe1, rechecked_keyframe2

def triangulate_points(P1, P2, pts1, pts2):
    assert pts1.shape == pts2.shape, \
    "Points in the two views must have the same shape for triangulation."

    points_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_homogeneous[:3] / points_homogeneous[3]
    
    valid_mask = np.abs(points_homogeneous[3]) > 1e-6
    if not np.all(valid_mask):
        raise ValueError("Invalid triangulated points detected. Some points have near-zero or zero homogeneous coordinates.")
    points_3d = points_homogeneous[:3, valid_mask] / points_homogeneous[3, valid_mask]
    return points_3d.T
    
def initialize(video_path, K, max_nfev=None):

    video_handler = VideoDataHandler(video_path, grayscale=True)
    keyframes = initial_keyframes(video_handler, threshold = 3)

    # Compute essential matrices
    K = K.astype(np.float64)
    E1, R1, t1, inlier_keyframe0_E1, inlier_keyframe1_E1 = compute_essential_matrix(
        keyframes[0], keyframes[1], K)
    E2, R2, t2, inlier_keyframe0_E2, inlier_keyframe2_E2 = compute_essential_matrix(
        keyframes[0], keyframes[2], K)

    aligned_keyframe0, aligned_keyframe1, aligned_keyframe2 = align_keyframes(
        inlier_keyframe0_E2, inlier_keyframe1_E1, inlier_keyframe2_E2)
    # Triangulate points
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R2, t2))
    points_3d = triangulate_points(P0, P2, aligned_keyframe0["points"], aligned_keyframe2["points"])

    # Pose estimation for K1
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, aligned_keyframe1["points"], K, None)
    R1_refined, _ = cv2.Rodrigues(rvec)
    t1_refined = tvec

    aligned_keyframes = [aligned_keyframe0, aligned_keyframe1, aligned_keyframe2]
    poses = [{"R": np.eye(3), "t": np.zeros((3,))}, {"R": R1_refined, "t": t1_refined}, {"R": R2, "t": t2}]
    print(poses)
    optimized_poses, optimized_points = bundle_adjustment(K, poses, points_3d, aligned_keyframes, max_nfev=max_nfev)
    descriptors_3d = aligned_keyframe2["descriptors"] 

    optimized_points_3d = {
        "points_3d" : optimized_points,
        "descriptors_3d" : descriptors_3d
    }

    return optimized_poses, aligned_keyframes, optimized_points_3d, video_handler
