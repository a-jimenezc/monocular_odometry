import cv2
import numpy as np
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose


def triangulate_points(pose1, pose2, matched_frame_1, matched_frame_2, K): #
    P1 = pose1.projection_matrix(K)
    P2 = pose2.projection_matrix(K)

    points_homogeneous = cv2.triangulatePoints(P1, P2, matched_frame_1.points.T, matched_frame_2.points.T)
    points_3d_estimated = points_homogeneous[:3] / points_homogeneous[3]

    return PointDescriptors(points_3d_estimated.T, matched_frame_2.descriptors)

def compute_relative_pose(matched_frame1, matched_frame2, K, ransac_threshold): #

    if len(matched_frame1.points) < 8:
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
                                         inlier_frame2.points,K) 
    
    relative_pose = CamPose(R, t.reshape(-1)).cam_pose_inv()

    return relative_pose, inlier_frame1, inlier_frame2

def pnp(corresponding_3d, corresponding_2d, K): #
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(corresponding_3d.points, corresponding_2d.points, K, None)
    R, _ = cv2.Rodrigues(rvec)
    pose = CamPose(R, tvec.reshape(3,))
    pose = pose.cam_pose_inv() # Camera pose with respect to origin. w_T_c

    return pose