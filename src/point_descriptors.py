import cv2
import numpy as np

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
    
    def subtract_points(self, other_points, distance_threshold): #

        if not isinstance(other_points, PointDescriptors):
            raise ValueError("Points must be an instance of PointDescriptors.")

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        raw_matches = bf.match(self.descriptors, other_points.descriptors)
        matched_indices = np.array([m.queryIdx for m in raw_matches if m.distance < distance_threshold])

        all_indices = np.arange(self.points.shape[0])
        unmatched_mask = ~np.isin(all_indices, matched_indices)

        unmatched_points = self.points[unmatched_mask]
        unmatched_descriptors = self.descriptors[unmatched_mask]

        if unmatched_points.size == 0:
            print("All points matched, no unmatched points remain.")
            return PointDescriptors(np.empty((0, self.points.shape[1])), np.empty((0, self.descriptors.shape[1])))

        return PointDescriptors(np.array(unmatched_points), np.array(unmatched_descriptors))
    
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