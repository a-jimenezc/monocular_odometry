import cv2
import numpy as np

def estimate_poses(K, 
                   init_keyframe_poses,
                   keyframes,
                   optimized_points_3d, 
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
    poses = init_keyframe_poses.copy()

    points_3d = optimized_points_3d["points_3d"]
    descriptors_3d = optimized_points_3d["descriptors_3d"]

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
        matched_2d_points = np.array([keypoints[m.trainIdx].pt for m in matches])

        if len(matched_points_3d) < min_points_threshold: # still keeps continuity
            print("Too few 3D points matched. Stopping pose estimation.")

            new_keyframe = {"points": keypoints, "descriptors": descriptors}
            def points_updater(keyframe_poses, keyframes, new_keyframe):
                return keyframe_poses, keyframes

            break

        # Estimate the pose using PnP
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_points_3d,
            matched_2d_points,
            K,
            None
        )
        if not retval:
            print("PnP failed for the current frame.")
            continue

        # Refine pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        # Append the estimated pose to the list of poses
        poses.append({"R": R, "t": t})

    return poses

