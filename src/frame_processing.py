import cv2
import numpy as np
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose

points_matcher_treshold = 30

def triangulate_points(pose1, pose2, matched_frame_1, matched_frame_2, K): #
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)

    points_homogeneous = cv2.triangulatePoints(P1, P2, matched_frame_1.points.T, matched_frame_2.points.T)
    points_3d_estimated = points_homogeneous[:3] / points_homogeneous[3]

    return PointDescriptors(points_3d_estimated.T, matched_frame_2.descriptors)

def compute_relative_pose(matched_frame1, matched_frame2, K, ransac_threshold): #

    if len(matched_frame1.points) < 5:
        raise ValueError("Insufficient points for essential matrix computation (minimum 5 points required).")

    F, inliers = cv2.findFundamentalMat(matched_frame1.points,
                                        matched_frame2.points, 
                                        cv2.FM_RANSAC, 
                                        ransac_threshold)
    
    if F is None or inliers is None:
        print("Fundamental matrix estimation failed.")
        return None
    
    # Compute pose
    E = K.T @ F @ K

    # use inliers to estimate the pose
    inliers = inliers.ravel() > 0
    inlier_frame1 = PointDescriptors(matched_frame1.points[inliers], 
                                     matched_frame1.descriptors[inliers])
    inlier_frame2 = PointDescriptors(matched_frame2.points[inliers], 
                                     matched_frame2.descriptors[inliers])

    retval, R, t, mask = cv2.recoverPose(E, 
                                         inlier_frame1.points, 
                                         inlier_frame2.points, 
                                         K)
    
    relative_pose = CamPose(R, t.reshape(-1)).cam_pose_inv()

    return relative_pose, inlier_frame1, inlier_frame2

def pnp(corresponding_3d, corresponding_2d, K): #
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(corresponding_3d.points, corresponding_2d.points, K, None)
    R, _ = cv2.Rodrigues(rvec)
    pose = CamPose(R, tvec.reshape(3,))
    pose = pose.cam_pose_inv() # Camera pose with respect to origin. w_T_c

    return pose

def frame_processing(poses, frames, points_3d_est, video_handler, poses_gt, K, frames_to_process=2):
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

        if matched_frame1.points.shape[0] < 7:
            print(f'Not enough matched points points at step {i}')
            continue

        _, inlier_frame1, inlier_frame2 = compute_relative_pose(matched_frame1, matched_frame2, K, ransac_threshold=1)
        #print('inlier_frame1.points.shape[0]', inlier_frame1.points.shape[0])
        if inlier_frame1.points.shape[0] < 5:
            print(f'Not enough inliers points at step {i}')
            continue

        corresponding_3d, corresponding_2d = points_3d_est.points_matcher(inlier_frame2, points_matcher_treshold)
        #print('corresponding_2d.points.shape[0]', corresponding_2d.points.shape[0])
        #print('points_3d_est.points.shape[0]', points_3d_est.points.shape[0])
        if corresponding_2d.points.shape[0] < 5:
            print(f'Not enough corresponding points at step {i}')
            continue

        pose2_est = pnp(corresponding_3d, corresponding_2d, K)

        pose2_est = pose2_est#.scaled_pose(poses_gt[i-1].t)
    
        poses.append(pose2_est) # exact correspondence between poses an frames
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