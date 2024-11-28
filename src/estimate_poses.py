import cv2
import numpy as np
from src.bundle_adjustment import bundle_adjustment_poses

def triangulate_new_points(keyframe_poses, keyframes, points_descriptors_3d, reprojection_threshold=1.0):
    """
    Triangulate new 3D points using the last three keyframes and their poses.

    Args:
    - keyframe_poses: List of keyframe poses, each as {"R": rotation matrix, "t": translation vector}.
    - keyframes: List of keyframes, each containing {"points": 2D feature points, "descriptors": feature descriptors}.
    - reprojection_threshold: Maximum allowed reprojection error for valid points.

    Returns:
    - new_points_3d: Array of newly triangulated 3D points (Nx3).
    - new_descriptors_3d: Array of descriptors associated with the new 3D points (NxD).
    """

    # Ensure there are at least 3 keyframes
    if len(keyframe_poses) < 3 or len(keyframes) < 3:
        raise ValueError("At least 3 keyframes are required for three-view triangulation.")

    # Use the last three keyframes and their poses
    pose1, pose2, pose3 = keyframe_poses[-3:]
    keyframe1, keyframe2, keyframe3 = keyframes[-3:]

    # Extract the projection matrices
    def compute_projection_matrix(pose):
        R, t = pose["R"], pose["t"].reshape(-1, 1)
        R_inv = R.T
        t_inv = -R_inv @ t
        return np.hstack((R_inv, t_inv))
    
    P1 = compute_projection_matrix(pose1)
    P2 = compute_projection_matrix(pose2)
    P3 = compute_projection_matrix(pose3)

    # Match points between keyframes
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches_12 = bf.match(keyframe1["descriptors"], keyframe2["descriptors"])
    matches_23 = bf.match(keyframe2["descriptors"], keyframe3["descriptors"])
    matches_13 = bf.match(keyframe1["descriptors"], keyframe3["descriptors"])

    # Keep only consistent matches across all three views
    common_matches = {}
    for m in matches_12:
        common_matches[m.trainIdx] = { # Using indexes in 2
            "query_idx_12": m.queryIdx,
            "train_idx_23": None
        }

    for m in matches_23:
        if m.queryIdx in common_matches: # Filtering elements that are not present in 1
            common_matches[m.queryIdx]["train_idx_23"] = m.trainIdx

    valid_matches = []
    for train_idx_12, match_data in common_matches.items():
        if match_data["train_idx_23"] is not None:  # Filtering elements that are not present in 3
            valid_matches.append((match_data["query_idx_12"], train_idx_12, match_data["train_idx_23"]))

    # Aligned 2d points
    points1 = np.array([keyframe1["points"][query_idx_12] for query_idx_12, _, _ in valid_matches])
    points2 = np.array([keyframe2["points"][train_idx1] for _, train_idx1, _ in valid_matches])
    points3 = np.array([keyframe3["points"][train_idx_23] for _, _, train_idx_23 in valid_matches])

    # To do: implement 3 view triangulation
    points_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = (points_homogeneous[:3] / points_homogeneous[3]).T  # Convert to non-homogeneous

    # Reprojection error check
    valid_indices = []
    for i, point in enumerate(points_3d):
        reproj1 = P1 @ np.append(point, 1)
        reproj2 = P2 @ np.append(point, 1)
        reproj3 = P3 @ np.append(point, 1)

        reproj1 = reproj1[:2] / reproj1[2]
        reproj2 = reproj2[:2] / reproj2[2]
        reproj3 = reproj3[:2] / reproj3[2]

        error1 = np.linalg.norm(reproj1 - points1[i])
        error2 = np.linalg.norm(reproj2 - points2[i])
        error3 = np.linalg.norm(reproj3 - points3[i])

        if error1 < reprojection_threshold and error2 < reprojection_threshold and error3 < reprojection_threshold:
            valid_indices.append(i)

    # Filter points and descriptors
    new_points_3d = points_3d[valid_indices]
    new_descriptors_3d_list = []
    for i, (_, _, train_idx_23) in enumerate(valid_matches):
        if i in valid_indices: # It will keep the order as in points_3d[valid_indices]
            new_descriptors_3d_list.append(keyframe3["descriptors"][train_idx_23])

    new_descriptors_3d = np.array(new_descriptors_3d_list)

    # Remove existing 3D points using descriptors
    if points_descriptors_3d is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.match(new_descriptors_3d, points_descriptors_3d["descriptors"])

        matched_indices = [m.queryIdx for m in matches]  # Indices of new points that match existing points
        filtered_indices = []
        for i in range(len(new_points_3d)):
            if i not in matched_indices:
                filtered_indices.append(i)

        new_points_3d = new_points_3d[filtered_indices]
        new_descriptors_3d = new_descriptors_3d[filtered_indices]

    return new_points_3d, new_descriptors_3d

def points_updater(keyframe_poses, keyframes, points_descriptors_3d, K, L=5, m=3):
    # extracting L previous keyframes and poses,
    window_keyframe_poses = keyframe_poses[-L:].copy()
    window_keyframes = keyframes[-L:].copy()
    optimized_window_keyframe_poses, optimized_points_descriptors_3d = bundle_adjustment_poses(K, 
                        window_keyframe_poses,
                        window_keyframes,
                        points_descriptors_3d,
                        m,
                        max_nfev=None)
    # modify original list
    keyframe_poses[-L:] = optimized_window_keyframe_poses
    # New points_3d
    new_points_3d, new_descriptors_3d = triangulate_new_points(keyframe_poses, keyframes)
    return new_points_3d, new_descriptors_3d

def estimate_poses(K, 
                   init_keyframe_poses,
                   keyframes,
                   points_descriptors_3d, 
                   video_handler,
                   min_points_threshold=1000):

    """
    Estimates poses for subsequent frames in a video sequence.

    Args:
    - optimized_poses: List of existing poses, each as {"R": rotation matrix, "t": translation vector}.
    - optimized_points_3d: Dictionary with keys "points_3d" and "descriptors_3d".
    - video_handler: An instance of VideoDataHandler to provide video frames.
    - min_points_threshold: Minimum number of 3D points required to continue.

    Returns:
    - poses: Updated list of poses with the new estimated poses.
    """
    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    keyframe_poses = init_keyframe_poses.copy()
    output_poses = init_keyframe_poses.copy()

    points_3d = points_descriptors_3d["points_3d"]
    descriptors_3d = points_descriptors_3d["descriptors_3d"]

    for frame in video_handler:
        # Detect feature points in the current frame
        keypoints, descriptors = feature_detector.detectAndCompute(frame, None)
        if descriptors is None or len(descriptors) < 500: # see threshold
            print(len(descriptors))
            print("Not enough descriptors in the current frame.")
            continue

        # Match descriptors with the 3D point descriptors
        matches = bf.match(descriptors_3d, descriptors)

        # Filter matches to retain the association between 3D points and 2D points
        matched_points_3d = np.array([points_3d[m.queryIdx] for m in matches])
        matched_descriptors_3d = np.array([descriptors_3d[m.queryIdx] for m in matches])
        matched_points_2d = np.array([keypoints[m.trainIdx].pt for m in matches])
        matched_descriptors_2d = np.array([descriptors[m.trainIdx] for m in matches])

        # Estimate the pose using PnP using only matched points
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_points_3d,
            matched_points_2d,
            K,
            None
        )
        if not retval:
            print("PnP failed for the current frame.")
            continue

        # Redefine pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        frame_pose = {"R": R, "t": t}

        if len(matched_points_3d) < min_points_threshold: # still keeps continuity
            print("Too few 3D points matched. Stopping pose estimation.")

            # Adding unfiltered new keyframe
            new_keyframe = {"points": keypoints.pt, "descriptors": descriptors}
            keyframe_poses.append(frame_pose)
            keyframes.append(new_keyframe)
            
            new_points_3d, new_descriptors_3d = points_updater(keyframe_poses,
                                                               keyframes, 
                                                               points_descriptors_3d,
                                                               K,
                                                               L=5, 
                                                               m=3)
            points_3d.append(new_points_3d)
            descriptors_3d.append(new_descriptors_3d)

            break

        # Append the estimated pose to the list of poses
        output_poses.append(frame_pose)

    return output_poses

