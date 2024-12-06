import cv2
import numpy as np

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

        return CamPose(self.R, self.t * scale)

    def flatten(self):#
        rvec, _ = cv2.Rodrigues(self.R)
        pose_array = np.hstack((rvec.ravel(), self.t.ravel()))
        return pose_array
    
    @staticmethod   
    def unflatten(pose_array):#
        rvec = pose_array[:3]
        tvec = pose_array[3:]
        R, _ = cv2.Rodrigues(rvec)
        return CamPose(R, tvec.reshape(3,))
    