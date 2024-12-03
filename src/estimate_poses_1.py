import cv2
import copy
import numpy as np
from src.bundle_adjustment_poses import bundle_adjustment
from src.initialization import initialize

def estimate_poses(K, 
                   init_keyframe_poses,
                   keyframes,
                   points_descriptors_3d, 
                   video_handler,
                   min_points_threshold,
                   max_nfev=1):

    """
    Estimates poses for subsequent frames in a video sequence.
    Returns:
    - poses: Updated list of poses with the new estimated poses.
    """
    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    keyframe_poses = copy.deepcopy(init_keyframe_poses)
    output_poses = copy.deepcopy(init_keyframe_poses)
    frames = copy.deepcopy(keyframes)

    points_3d = points_descriptors_3d["points_3d"]
    descriptors_3d = points_descriptors_3d["descriptors_3d"]

    i = 0
    for frame in video_handler:
        # Detect feature points in the current frame
        keypoints, descriptors = feature_detector.detectAndCompute(frame, None)
        points = np.array([kp.pt for kp in keypoints])
        current_frame = {'points': points, 'descriptors' : descriptors}

        if descriptors is None:
            print('empty frame detection')
            continue
        if len(descriptors) < 100:
            print("Not enough descriptors in the current frame.")
            continue

        # Match descriptors with the 3D point descriptors
        matches = bf.match(descriptors_3d, descriptors)
        matches = [m for m in matches if m.distance < 150]
        i = i + 1
        #if i > 15:
        #    break
        if len(matches) < 8:
            print("Too few 3D points matched. Calculating new ones.")
            try:
                init_keyframe_poses, keyframes_, \
                optimized_points_3d, video_handler = initialize(frames[-1], output_poses[-1],
                                                                video_handler, K, max_nfev=10)
            except:
                continue

            points_3d = np.concatenate((points_3d, optimized_points_3d["points_3d"]), axis=0)
            descriptors_3d = np.concatenate((descriptors_3d, optimized_points_3d["descriptors_3d"]), axis=0)
            output_poses = output_poses + init_keyframe_poses
            frames = frames + keyframes_

            continue

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

        # PnP returns the inverse of the camera pose. c_T_w
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        frame_pose = {"R": R.T, "t": -R.T @ t}
        output_poses.append(frame_pose)
        frames.append(current_frame)

        print('first 3d_points', len(points_3d))
        print('detected points current frame', len(keypoints))
        print('current point matches with 3d points: ', len(matches))
        print('number of poses', len(output_poses))

    return output_poses