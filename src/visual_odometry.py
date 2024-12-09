import cv2
import numpy as np
from src.cam_pose import CamPose
from src.utility_functions import compute_relative_pose, triangulate_points
from src.frame_processing import frame_processing
from src.bundle_adjustment import bundle_adjustment
from src.video_data_handler import EndOfFramesError
from src.scale_recovery import estimate_scale

def visual_odometry(video_handler, K, points_matcher_treshold, 
                    ransac_threshold,  init_skip, bundle_adjust=False, recover_scale=False):

    if K.shape != (3, 3) or np.linalg.det(K) == 0:
        raise ValueError("Invalid intrinsic matrix provided.")

    #initial two poses
    frames = []
    poses = [CamPose(np.eye(3), np.array([0, 0, 0]))]
    i, j = 0, 0
    for frame in video_handler:

        # Initial frame
        if j == 0:
            frames.append(frame)
            j = j + 1
            continue

        # Skipping initial frames
        if i < init_skip: 
            i = i + 1
            continue

        # Selecting a second frame suficiently different from the first one
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(frames[0].descriptors, frame.descriptors)
        match_distances = [m.distance for m in matches]
        if np.mean(match_distances) < 150: continue

        # Estimate pose of second view, extract inliers, triangulate points
        matched_frame0, matched_frame1 = frames[0].points_matcher(frame, points_matcher_treshold)
        if matched_frame0.points.shape[0] < 10: 
            print('Not enough matched frames at initialization')
            continue
        pose1_est, inlier_frame0, inlier_frame1 = compute_relative_pose(matched_frame0, 
                                                                        matched_frame1, K, 
                                                                        ransac_threshold=ransac_threshold)
        if inlier_frame0.points.shape[0] < 10:
            print('Not enough inliers frames at initialization')
            continue

        if recover_scale == True:
            scale = estimate_scale(poses[0], pose1_est, inlier_frame0, inlier_frame1, K, 
                distance_threshold = 30,
                ransac_n = 5,
                num_iterations = 100)
            pose1_est.t = pose1_est.t * scale

        points_3d_est = triangulate_points(poses[0], pose1_est, inlier_frame0, inlier_frame1, K)



        poses.append(pose1_est)
        frames.append(frame)
        if len(poses) > 1: break

    # Initial 15 poses
    frames_processing = 15

    poses_gt_init = None
    poses, frames, points_3d_est = frame_processing(poses, frames, points_3d_est, 
                                                    video_handler, poses_gt_init, K,
                                                    ransac_threshold,
                                                    points_matcher_treshold,
                                                    recover_scale=False,
                                                    frames_to_process=frames_processing)
    if bundle_adjust == True:
        poses, points_3d_est = bundle_adjustment(poses, frames, points_3d_est, K, last_n=frames_processing, 
                                                distance_matcher=points_matcher_treshold, 
                                                max_nfev=10)
        
    for i in range (15):
        frames_processing = 15
        try:
            gt_index = 4+(frames_processing*i)
            poses_gt_proc = None
            poses, frames, points_3d_est = frame_processing(poses, frames, points_3d_est, 
                                                        video_handler, poses_gt_proc, K,
                                                        ransac_threshold,
                                                        points_matcher_treshold,
                                                        recover_scale=False,
                                                        frames_to_process=frames_processing)
        except (EndOfFramesError, IndexError):
            print('breaking loop')
            break
        
        if bundle_adjust == True:
            poses, points_3d_est = bundle_adjustment(poses, frames, points_3d_est, K, last_n=2, 
                                                    distance_matcher=points_matcher_treshold, 
                                                    max_nfev=10)

    return poses, points_3d_est
