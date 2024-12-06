import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.point_descriptors import PointDescriptors
from src.cam_pose import CamPose

K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]]).astype(float)

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

points_3d = np.array(points_3d)
descriptors_3d = np.array(descriptors_3d).astype(np.float32)

pose0 = CamPose(np.eye(3), np.array([0, 0, 0]))
pose1 = CamPose(R.from_euler('x', 30, degrees=True).as_matrix(), np.array([0, 1, 0.5]))
pose2 = CamPose(R.from_euler('y', 30, degrees=True).as_matrix(), np.array([1, 1, 1]))
pose3 = CamPose(R.from_euler('z', 30, degrees=True).as_matrix(), np.array([1, 0, 1.5]))
pose4 = CamPose(R.from_euler('x', 20, degrees=True).as_matrix(), np.array([1.5, 1, 2]))
pose5 = CamPose(R.from_euler('y', 20, degrees=True).as_matrix(), np.array([0, 1.5, 2.5]))
pose6 = CamPose(R.from_euler('x', 15, degrees=True).as_matrix(), np.array([0, 1.5, 3.5]))
pose7 = CamPose(R.from_euler('x', 15, degrees=True).as_matrix(), np.array([0, 1.5, 3.5]))
pose8 = CamPose(R.from_euler('x', 15, degrees=True).as_matrix(), np.array([0, 1.5, 4.5]))
pose9 = CamPose(R.from_euler('y', 15, degrees=True).as_matrix(), np.array([0, 1.5, 5]))
pose10 = CamPose(R.from_euler('z', 15, degrees=True).as_matrix(), np.array([0, 1.5, 5.5]))

poses_gt = [pose0, pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8, pose9, pose10]

frame0 = PointDescriptors(pose0.project_into_cam(points_3d[:25,:], K), descriptors_3d[:25,:])
frame1 = PointDescriptors(pose1.project_into_cam(points_3d[5:25,:], K), descriptors_3d[5:25,:])
frame2 = PointDescriptors(pose2.project_into_cam(points_3d[10:40,:], K), descriptors_3d[10:40,:])
frame3 = PointDescriptors(pose3.project_into_cam(points_3d[20:50,:], K), descriptors_3d[20:50,:])
frame4 = PointDescriptors(pose4.project_into_cam(points_3d[30:50,:], K), descriptors_3d[30:50,:])
frame5 = PointDescriptors(pose5.project_into_cam(points_3d[25:50,:], K), descriptors_3d[25:50,:])
frame6 = PointDescriptors(pose6.project_into_cam(points_3d[40:50,:], K), descriptors_3d[40:50,:])
frame7 = PointDescriptors(pose7.project_into_cam(points_3d[40:50,:], K), descriptors_3d[40:50,:])
frame8 = PointDescriptors(pose8.project_into_cam(points_3d[30:50,:], K), descriptors_3d[30:50,:])
frame9 = PointDescriptors(pose9.project_into_cam(points_3d[40:50,:], K), descriptors_3d[40:50,:])
frame10 = PointDescriptors(pose10.project_into_cam(points_3d[35:50,:], K), descriptors_3d[35:50,:])

frames_list = [frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9, frame10]

class EndOfFramesError(Exception):
    """Custom exception to indicate the end of the frames."""
    pass

class FrameListHandler:
    def __init__(self, frames_list):
        self.frames_list = frames_list
        self.index = 0  # Track the current position in the list

    def __iter__(self):
        return self  # The class itself is an iterator

    def __next__(self):
        #print(f"Index: {self.index}, Length: {len(self.frames_list)}")
        if self.index >= len(self.frames_list):
            print("EndOfFramesError")
            raise EndOfFramesError("No more frames to process.")  # No more frames to process

        frame = self.frames_list[self.index]
        self.index += 1
        return frame

video_handler = FrameListHandler(frames_list)

