import cv2
import numpy as np
from src.video_data_handler import VideoDataHandler

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

def compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0):
    """
    Compute the essential matrix between two keyframes using their feature points and descriptors.
    Returns:
    - E: Essential matrix (3x3).
    - inliers: Boolean mask of inliers used for the computation.
    """
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(keyframe1["descriptors"], keyframe2["descriptors"])
    matched_keyframe1 = np.array([keyframe1[m.queryIdx] for m in matches])
    matched_keyframe2 = np.array([keyframe2[m.trainIdx] for m in matches])

    # Compute the fundamental and essential matrix
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

def filter_keyframe_by_correspondence(k1, k2, tolerance=1e-6):
    """
    Filters k2 to retain only points that correspond to points in k1.
    Returns:
    - filtered_k2: Modified k2 with points and descriptors corresponding to k1.
    """
    # Extract points from k1 and k2
    points_k1 = k1["points"]
    points_k2 = k2["points"]

    # Compute pairwise distances between points in k1 and k2
    dists = np.linalg.norm(points_k1[:, None] - points_k2[None, :], axis=2)

    # Identify points in k2 that have at least one match in k1
    matches = np.any(dists < tolerance, axis=0)

    # Filter points and descriptors in k2 based on matches
    filtered_points_k2 = points_k2[matches]
    filtered_descriptors_k2 = k2["descriptors"][matches]

    # Return the modified keyframe 2
    filtered_k2 = {
        "points": filtered_points_k2,
        "descriptors": filtered_descriptors_k2
    }
    return filtered_k2

def initialize(video_path, K):

    video_handler = VideoDataHandler(video_path, grayscale=True)
    keyframes = initial_keyframes(video_handler, threshold = 3)

    # Compute essential matrices
    K = K.astype(np.float64)
    E1, R1, t1, inlier_keyframe0_E1, inlier_keyframe1_E1 = compute_essential_matrix(
        keyframes[0], keyframes[1], K)
    E2, R2, t2, inlier_keyframe0_E2, inlier_keyframe2_E2 = compute_essential_matrix(
        keyframes[0], keyframes[2], K)





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