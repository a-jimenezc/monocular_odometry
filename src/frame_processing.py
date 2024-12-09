from src.scale_recovery import estimate_scale
from src.utility_functions import triangulate_points, compute_relative_pose, pnp

def frame_processing(poses, frames, points_3d_est, video_handler, poses_gt, K, ransac_threshold, 
                     points_matcher_treshold, recover_scale=False, frames_to_process=2):
    '''
    Modifieas lists of poses, frames, and points_3d_est object
    '''

    i = 0
    frames_processed = 0
    for frame in video_handler:
        
        # Estimate next pose using inliers, add new 3d points
        matched_frame1, matched_frame2 = frames[-1].points_matcher(frame, points_matcher_treshold)
        print('matched_frame1.points.shape[0]', matched_frame1.points.shape[0])
        i = i + 1

        if matched_frame1.points.shape[0] < 8:
            print(f'Not enough matched points points at step {i}')
            continue

        _, inlier_frame1, inlier_frame2 = compute_relative_pose(matched_frame1, matched_frame2, K, ransac_threshold=ransac_threshold)

        if inlier_frame1.points.shape[0] < 5:
            print(f'Not enough inliers points at step {i}')
            continue

        corresponding_3d, corresponding_2d = points_3d_est.points_matcher(inlier_frame2, points_matcher_treshold)
        
        if corresponding_2d.points.shape[0] < 5:
            print(f'Not enough corresponding points at step {i}')
            continue

        pose2_est = pnp(corresponding_3d, corresponding_2d, K)

        if recover_scale == True:
            scale = estimate_scale(poses[-1], pose2_est, inlier_frame1, inlier_frame2, K, 
                distance_threshold = 30,
                ransac_n = 5,
                num_iterations = 100)
            pose2_est.t = pose2_est.t * scale
    
        poses.append(pose2_est)
        frames.append(frame)

        frames_processed = frames_processed + 1

        if frames_processed > (frames_to_process-1) : 
            # Triangulating new-found points
            new_points_2 = inlier_frame2.subtract_points(corresponding_2d, 1)
            if new_points_2.points.shape[0]== 0:
                print(f'No new 3d points at step {i}')
                break
            new_points_matched_1, new_points_matched_2 = inlier_frame1.points_matcher(new_points_2, points_matcher_treshold)

            if new_points_matched_1.points.shape[0] == 0:
                print(f'No new 3d points at step {i}')
                break

            new_points_3d_est = triangulate_points(poses[-2], 
                                                pose2_est,
                                                new_points_matched_1,
                                                new_points_matched_2,
                                                K)
            points_3d_est = points_3d_est.extend_points(new_points_3d_est)
            break

        # Triangulating new-found points
        new_points_2 = inlier_frame2.subtract_points(corresponding_2d, 1)
        if new_points_2.points.shape[0]== 0:
            print(f'No new 3d points at step {i}')
            continue
        new_points_matched_1, new_points_matched_2 = inlier_frame1.points_matcher(new_points_2, points_matcher_treshold)

        if new_points_matched_1.points.shape[0] == 0:
            print(f'No new 3d points at step {i}')
            continue

        new_points_3d_est = triangulate_points(poses[-2], 
                                            pose2_est,
                                            new_points_matched_1,
                                            new_points_matched_2,
                                            K)
        points_3d_est = points_3d_est.extend_points(new_points_3d_est)


    return poses, frames, points_3d_est
