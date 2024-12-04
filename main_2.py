import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class CamPose:

    def __init__(self, R, t):

        if R.shape != (3, 3): raise ValueError("R must be a 3x3 numpy array.")
        if t.shape != (3,): raise ValueError("t must be a 3-element numpy array.")
        self.R = R
        self.t = t
    
    def projection_matrix(self, K): #

        projection = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
        projection_matrix = K @ projection @ self.cam_pose_inv().to_matrix()
        return projection_matrix

    def cam_pose_inv(self): #

        R_inv = self.R.T
        t_inv = - R_inv @ self.t
        return CamPose(R_inv, t_inv)
    
    def to_matrix(self): #
            
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = self.R
            pose_matrix[:3, 3] = self.t
            return pose_matrix
    
    def from_matrix(self, pose_matrix):

        if pose_matrix.shape != (4, 4):
            raise ValueError("pose_matrix must be a 4x4 numpy array.")
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        return CamPose(R, t)
    
    def project_into_cam(self, points_3d, K): #

        cam_pose_inv_matrix = self.cam_pose_inv().to_matrix()
        points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

        transformed_points = cam_pose_inv_matrix @ points_3d_hom.T
        projected_points_hom = K  @ transformed_points[:3, :]
        if np.any(projected_points_hom[2] == 0):
            raise ValueError("Homogeneous coordinate w is zero for some points.")
        projected_points = projected_points_hom[:2] / projected_points_hom[2]

        return projected_points.T

    def compose_poses(self, pose2):

        if not isinstance(pose2, CamPose):
            raise ValueError("pose2 must be an instance of CamPose.")
        
        self_matrix = self.to_matrix()
        pose2_matrix = pose2.to_matrix()
        
        resulting_pose_matrix = self_matrix @ pose2_matrix
        resulting_pose = self.from_matrix(resulting_pose_matrix)
        
        return resulting_pose
    
    def scaled_pose(self, ground_truth_translation):

        estimated_norm = np.linalg.norm(self.t)
        gt_norm = np.linalg.norm(np.array(ground_truth_translation).reshape(-1))

        if abs(estimated_norm) < 1e-10:
            print("Estimated norm is too close to zero.")

        # Compute the scale factor
        scale = gt_norm / estimated_norm
        print(scale)

        return CamPose(self.R, self.t * scale)

class PointDescriptors():

    def __init__(self, points, descriptors):
        self.points = points
        self.descriptors = descriptors
    
    def points_matcher(self, points2, distance_threshold): #

        if not isinstance(points2, PointDescriptors):
            raise ValueError("Points must be an instance of PointDescriptors.")
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        raw_matches = bf.match(self.descriptors, points2.descriptors)
        
        matches = []
        for m in raw_matches:
            if m.distance < distance_threshold:
                matches.append(m) 
        
        if not matches:
            print("No matches found below the distance threshold.")
            return PointDescriptors(np.empty((0, self.points.shape[1])), np.empty((0, self.descriptors.shape[1]))), \
                   PointDescriptors(np.empty((0, points2.points.shape[1])), np.empty((0, points2.descriptors.shape[1])))

        matched_points1 = np.array([self.points[m.queryIdx] for m in matches])
        matched_descriptors1 = np.array([self.descriptors[m.queryIdx] for m in matches])

        matched_points2 = np.array([points2.points[m.trainIdx] for m in matches])
        matched_descriptors2 = np.array([points2.descriptors[m.trainIdx] for m in matches])

        matched_points_1 = PointDescriptors(matched_points1, matched_descriptors1)
        matched_points_2 = PointDescriptors(matched_points2, matched_descriptors2)

        return matched_points_1, matched_points_2
    
    def extend_points(self, other_points): #

        if not isinstance(other_points, PointDescriptors):
            raise ValueError("Point must be an instance of PointDescriptors.")
        if self.points.shape[1] != other_points.points.shape[1]:
            raise ValueError("Point dimensions do not match.")
        
        points = np.vstack((self.points, other_points.points))
        descriptors = np.vstack((self.descriptors, other_points.descriptors))
        return PointDescriptors(points, descriptors)
    
    def transform_points_3d(self, pose_matrix):
        if self.points.shape[1] != 3:
            raise ValueError("Points must be 3D (N x 3).")

        points_3d_hom = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        transformed_points_hom = pose_matrix @ points_3d_hom.T
        if np.any(transformed_points_hom[3] == 0):
            raise ValueError("Homogeneous coordinate w is zero for some points.")        
        transformed_points = transformed_points_hom[:3] / transformed_points_hom[3]

        return PointDescriptors(transformed_points.T, self.descriptors)

def triangulate_points(pose1, pose2, matched_frame_1, matched_frame_2): #
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
    
    E = K.T @ F @ K

    # Compute pose
    retval, R, t, mask = cv2.recoverPose(E, 
                                         matched_frame1.points, 
                                         matched_frame2.points, 
                                         K)

    inliers = inliers.ravel() > 0

    inliers = np.array(inliers).ravel() > 0

    inlier_frame1 = PointDescriptors(matched_frame0.points[inliers], 
                                     matched_frame0.descriptors[inliers])
    inlier_frame2 = PointDescriptors(matched_frame1.points[inliers], 
                                     matched_frame1.descriptors[inliers])
    
    relative_pose = CamPose(R, t.reshape(-1)).cam_pose_inv()

    return relative_pose, inlier_frame1, inlier_frame2

# Test data
points_3d = [
    [3, -2, 10.00],
    [1, 4, 10.10],
    [-4, -1, 10.20],
    [5, -3, 10.30],
    [2, 2, 10.40],
    [-1, -4, 10.50],
    [0, 5, 10.60],
    [-3, 1, 10.70],
    [4, -5, 10.80],
    [-2, 3, 10.90],
    [1, -3, 11.00],
    [-5, 0, 11.10],
    [2, 4, 11.20],
    [-3, -5, 11.30],
    [5, 1, 11.40],
    [0, -2, 11.50],
    [-4, 3, 11.60],
    [3, -1, 11.70],
    [1, 5, 11.80],
    [-2, -4, 11.90],
    [4, 0, 12.00],
    [-1, 3, 12.10],
    [0, -5, 12.20],
    [5, 2, 12.30],
    [-3, -2, 12.40],
    [2, 1, 12.50],
    [-5, 4, 12.60],
    [4, -3, 12.70],
    [-2, 5, 12.80],
    [3, 0, 12.90],
    [1, -4, 13.00],
    [-4, 2, 13.10],
    [5, -5, 13.20],
    [-1, 0, 13.30],
    [2, -3, 13.40],
    [-3, 4, 13.50],
    [0, 1, 13.60],
    [4, -2, 13.70],
    [-5, 3, 13.80],
    [1, -1, 13.90],
    [-2, 2, 14.00],
    [3, -5, 14.10],
    [0, 4, 14.20],
    [-4, -3, 14.30],
    [5, 0, 14.40],
    [-1, -2, 14.50],
    [2, 3, 14.60],
    [-3, -4, 14.70],
    [4, 5, 14.80],
    [-5, 1, 15.00]
]
points_3d = np.array(points_3d)

descriptors_3d = [
    [1.87, 4.75, 3.66, 2.99, 0.78],
    [0.78, 0.29, 4.33, 3.01, 3.54],
    [0.1, 4.85, 4.16, 1.06, 0.91],
    [0.92, 1.52, 2.62, 2.16, 1.46],
    [3.06, 0.7, 1.46, 1.83, 2.28],
    [3.93, 1.0, 2.57, 2.96, 0.23],
    [3.04, 0.85, 0.33, 4.74, 4.83],
    [4.04, 1.52, 0.49, 3.42, 2.2],
    [0.61, 2.48, 0.17, 4.55, 1.29],
    [3.31, 1.56, 2.6, 2.73, 0.92],
    [4.85, 3.88, 4.7, 4.47, 2.99],
    [4.61, 0.44, 0.98, 0.23, 1.63],
    [1.94, 1.36, 4.14, 1.78, 1.4],
    [2.71, 0.7, 4.01, 0.37, 4.93],
    [3.86, 0.99, 0.03, 4.08, 3.53],
    [3.65, 3.86, 0.37, 1.79, 0.58],
    [4.32, 3.12, 1.65, 0.32, 1.55],
    [1.63, 3.65, 3.19, 4.44, 2.36],
    [0.6, 3.57, 3.8, 2.81, 3.85],
    [2.47, 2.61, 2.14, 0.13, 0.54],
    [0.16, 3.18, 1.57, 2.54, 4.54],
    [1.25, 2.05, 3.78, 1.14, 0.38],
    [1.45, 0.81, 4.65, 4.04, 3.17],
    [4.36, 4.02, 0.93, 4.46, 2.7],
    [4.04, 4.48, 1.59, 0.55, 1.14],
    [2.14, 4.09, 4.3, 0.03, 2.55],
    [2.09, 1.11, 0.6, 1.69, 4.71],
    [1.62, 2.59, 3.52, 1.82, 4.86],
    [4.81, 1.26, 2.49, 1.5, 1.42],
    [0.18, 3.05, 2.51, 0.26, 1.39],
    [4.54, 1.2, 0.72, 2.45, 4.93],
    [1.21, 3.36, 3.81, 1.19, 3.64],
    [1.84, 3.16, 3.17, 2.68, 0.45],
    [4.18, 1.6, 0.93, 0.2, 2.95],
    [3.39, 0.08, 2.56, 1.13, 3.23],
    [0.87, 3.45, 1.93, 4.68, 0.69],
    [1.71, 0.57, 4.62, 4.39, 1.29],
    [3.3, 4.09, 2.78, 2.65, 1.21],
    [0.47, 4.49, 4.5, 3.17, 1.7],
    [1.75, 3.63, 4.49, 4.44, 3.9],
    [3.21, 0.42, 0.81, 4.49, 3.03],
    [0.05, 0.51, 3.32, 0.03, 0.8],
    [2.74, 3.46, 3.26, 1.12, 3.56],
    [1.19, 1.63, 3.73, 3.25, 4.25],
    [3.29, 2.84, 0.47, 1.84, 1.33],
    [1.22, 4.87, 1.97, 4.46, 3.16],
    [3.97, 2.51, 2.88, 2.46, 0.98],
    [3.61, 1.4, 0.12, 3.23, 0.89],
    [4.7, 4.77, 4.57, 1.85, 0.08],
    [4.64, 2.14, 4.83, 4.82, 4.27]
]
descriptors_3d = np.array(descriptors_3d).astype(np.float32)

pose0 = CamPose(np.eye(3), np.array([0, 0, 0]))
pose1 = CamPose(R.from_euler('x', 10, degrees=True).as_matrix(), np.array([0, 1, 0.5]))
pose2 = CamPose(R.from_euler('y', 10, degrees=True).as_matrix(), np.array([1, 1, 1]))
pose3 = CamPose(R.from_euler('z', 10, degrees=True).as_matrix(), np.array([1, 0, 1.5]))
pose4 = CamPose(R.from_euler('x', 5, degrees=True).as_matrix(), np.array([1.5, 1, 2]))
pose5 = CamPose(R.from_euler('y', 5, degrees=True).as_matrix(), np.array([0, 1.5, 2.5]))

K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])

frame0 = PointDescriptors(pose0.project_into_cam(points_3d[:15,:], K), descriptors_3d[:15,:])
frame1 = PointDescriptors(pose1.project_into_cam(points_3d[5:20,:], K), descriptors_3d[5:20,:])

matched_frame0, matched_frame1 = frame0.points_matcher(frame1, 0.1)


relative_pose, inlier_frame0, inlier_frame1 = compute_relative_pose(matched_frame0, 
                                                                   matched_frame1, K, ransac_threshold=1)


#print('points', matched_frame0.points)
#print('inliers', inlier_frame0.points)

#print('pose', pose1.t)
#print('relative pose', relative_pose.scaled_pose(pose1.t).t)
#points_3d_estimated = triangulate_points(pose0, pose1, matched_frame0, matched_frame1)

#print(points_3d[5:15,:])
#print('points_3d_estimated',points_3d_estimated.points)

#points3d_1 = PointDescriptors(points_3d[:10,:], descriptors_3d[:10,:])
#points3d_2 = PointDescriptors(points_3d[5:15,:], descriptors_3d[5:15,:])

#points3d_extended= points3d_1.extend_points(points3d_2)

#print(points3d_1.points)
#print(points3d_2.points)
#print('points3d_extended', points3d_extended.points)
