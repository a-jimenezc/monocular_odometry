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
        if descriptors is None:
            continue
        
        # Compute similarity with previous frame's descriptors
        if previous_descriptors is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(previous_descriptors, descriptors)
            match_distances = [m.distance for m in matches]

            if np.mean(match_distances) < threshold:
                continue
        
        # Store keyframe data
        points = np.array([kp.pt for kp in keypoints])
        keyframes.append({
            "points": points,
            "descriptors": descriptors
        })
        
        previous_descriptors = descriptors
        
        # Limit to 3 keyframes
        if len(keyframes) >= 3:
            break
    return keyframes

def keyframe_matcher(keyframe1, keyframe2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(keyframe1["descriptors"], keyframe2["descriptors"])

    # Extract matched points and descriptors
    matched_points1 = np.array([keyframe1["points"][m.queryIdx] for m in matches])
    matched_descriptors1 = np.array([keyframe1["descriptors"][m.queryIdx] for m in matches])

    matched_points2 = np.array([keyframe2["points"][m.trainIdx] for m in matches])
    matched_descriptors2 = np.array([keyframe2["descriptors"][m.trainIdx] for m in matches])

    # Return matched keyframes
    matched_keyframe1 = {"points": matched_points1, "descriptors": matched_descriptors1}
    matched_keyframe2 = {"points": matched_points2, "descriptors": matched_descriptors2}

    return matched_keyframe1, matched_keyframe2

def compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0):
    """
    Compute the essential matrix between two keyframes using their feature points and descriptors.
    Returns:
    - E: Essential matrix (3x3).
    - inliers: Boolean mask of inliers used for the computation.
    """

    # Compute the fundamental and essential matrix
    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)
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
    inlier_keyframe1 = matched_keyframe1[inliers]
    inlier_keyframe2 = matched_keyframe2[inliers]

    return E, R, t, inlier_keyframe1, inlier_keyframe2

def align_keyframes(keyframe0, keyframe1, keyframe2):
    """
    Retains only common points.
    Assumes there is a single correspondence based on descriptors.
    """
    # Initial matching
    matched_keyframe0, matched_keyframe2 = keyframe_matcher(keyframe0, keyframe2)

    # Alignment
    matched_keyframe0_a, matched_keyframe1 = keyframe_matcher(matched_keyframe0, keyframe1)

    # Sanity Check
    matched_keyframe0_b, matched_keyframe2 = keyframe_matcher(matched_keyframe0_a, keyframe2)
    if matched_keyframe0_a["points"].shape != matched_keyframe0_b["points"].shape:
        raise ValueError("The points in the two keyframes do not have the same shape.")

    return matched_keyframe0_a, matched_keyframe1, matched_keyframe2

def triangulate_points(P1, P2, pts1, pts2):
    assert pts1.shape == pts2.shape, \
    "Points in the two views must have the same shape for triangulation."

    points_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_homogeneous[:3] / points_homogeneous[3]
    return points_3d.T
    
def initialize(video_path, K):

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

    # Return unoptimized poses and points (placeholder for bundle adjustment)
    aligned_keyframes = [aligned_keyframe0, aligned_keyframe1, aligned_keyframe2]
    poses = [{"R": np.eye(3), "t": np.zeros((3,))}, {"R": R1_refined, "t": t1_refined}, {"R": R2, "t": t2}]
    optimized_poses, optimized_points = bundle_adjustment(K, poses, points_3d, aligned_keyframes)

    return optimized_poses, optimized_points
