import cv2
import numpy as np
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose
from src.frame_processing import compute_relative_pose, triangulate_points, frame_processing
from src.bundle_adjustment import bundle_adjustment
#from test_data.synthetic_data import points_3d, descriptors_3d, poses_gt, frames_list, video_handler, K
from test_data.synthetic_data import poses_gt
from src.video_data_handler import EndOfFramesError, VideoDataHandler
from src.plot_poses_plane import plot_poses, plot_camera_poses
from src.plot_points import plot_point_cloud

points_matcher_treshold = 50
ransac_threshold = 2

video_path = 'test_data/output.mp4'

K = np.array([[608.56811625, 0, 629.83269351],[0, 614.54502235, 346.79688358],[0, 0, 1]], dtype=np.float32)

K_00 = np.array([
    [9.842439e+02, 0.000000e+00, 6.900000e+02],
    [0.000000e+00, 9.808141e+02, 2.331966e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
], dtype=np.float32)

if K.shape != (3, 3) or np.linalg.det(K) == 0:
    raise ValueError("Invalid intrinsic matrix provided.")

video_handler = VideoDataHandler(video_path, grayscale=True)

#initial two poses
frames = []
poses = [CamPose(np.eye(3), np.array([0, 0, 0]))]
i = 0
for frame in video_handler:

    # Initial frame
    if frame.points.shape[0] < 100: continue # Ensure minimun number of points
    print('frame.points.shape[0]', frame.points.shape[0])
    i = i + 1
    if i == 1:
        frames.append(frame)
        continue

    # Second frame suficiently different from the first one
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(frames[0].descriptors, frame.descriptors)
    print(len(matches))
    match_distances = [m.distance for m in matches]
    print(np.mean(match_distances))
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
    pose1_est = pose1_est #.scaled_pose(poses_gt[1].t) # Scaling with ground truth
    points_3d_est = triangulate_points(poses[0], pose1_est, inlier_frame0, inlier_frame1, K)

    poses.append(pose1_est)
    frames.append(frame)
    if len(poses) > 1: break

#initial 15 poses
frames_processing = 15
poses_gt_init = poses_gt#[2:4]
poses, frames, points_3d_est = frame_processing(poses, frames, points_3d_est, 
                                                video_handler, poses_gt_init, K,
                                                ransac_threshold,
                                                points_matcher_treshold,
                                                frames_to_process=frames_processing)

#poses, points_3d_est = bundle_adjustment(poses, frames, points_3d_est, K, last_n=frames_processing, 
#                                         distance_matcher=points_matcher_treshold, 
#                                         max_nfev=10)
for i in range (15):
    frames_processing = 15
    try:
        gt_index = 4+(frames_processing*i)
        poses_gt_proc = poses_gt#[gt_index:]
        poses, frames, points_3d_est = frame_processing(poses, frames, points_3d_est, 
                                                    video_handler, poses_gt_proc, K,
                                                    ransac_threshold,
                                                    points_matcher_treshold,
                                                    frames_to_process=frames_processing)
    except (EndOfFramesError, IndexError):
        print('breaking loop')
        break
    #poses, points_3d_est = bundle_adjustment(poses, frames, points_3d_est, K, last_n=2, 
    #                                        distance_matcher=points_matcher_treshold, 
    #                                        max_nfev=10)
plot_camera_poses(poses, ax=None, scale=0.1)
plot_poses(poses, plane='xz')
plot_point_cloud(points_3d_est.points)
#gt = [(pose.R) for pose in poses_gt]
#print('ground truth', gt[-1])
#est = [(pose.R) for pose in poses]
#print('estimated', len(est), est[-1])