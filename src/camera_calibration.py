from video_data_handler import VideoDataHandler
import cv2
import time
import numpy as np
import logging



class CameraCalibration:
    
    def __init__(self,source, max_frames=20):
        self.vdh = VideoDataHandler(source) # for video input stream
        self.source = source
        self.chessboard_size = (6,4)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) # the grid point 
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.max_frames = max_frames # max frames going to be captured from stream
        #self.frame_interval = frame_interval # interval between capturing useless
        self.obj_pts = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.obj_pts[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)  
    
    def calibrate(self):
        
        obj_points = [] # 3d  
        img_points = [] # 2d
        
        print(f"[CameraCal] Started Calibration, max frame : {self.max_frames}")
        counter = 0
        
        for frame in self.vdh:
            
            if not self.vdh.grayscale:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

            cv2.imshow("Calibrating", frame)
                
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):
                ret, corners = cv2.findChessboardCorners(frame, self.chessboard_size, None)
                if ret:
                    
                    obj_points.append(self.obj_pts) 
                    corners_refined = cv2.cornerSubPix(frame, corners, (5, 5), (-1, -1), self.criteria)
                    print(f"[CameraCal] Corner Detected for Frame:{counter} ")
                    img_points.append(corners_refined)
                    cv2.drawChessboardCorners(frame, (6,4), corners_refined, ret)
                    cv2.imshow("Calibrating", frame)
                    counter += 1
                else:
                    print(f"[CameraCal] No corner detected for the frame")
                    
            if key == ord('q') or counter >= self.max_frames:
                print(f"[CameraCal] {self.max_frames} frames recieved, exit calibrating sequence")
                break
        mtx = np.zeros((3, 3))  
        dist = np.zeros((4, 1)) 
        
        np.expand_dims(np.asarray(obj_points), -2)
        
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(obj_points, img_points, frame.shape[::-1], mtx, dist, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)  
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frame.shape[::-1], None, None)
        if ret:
            print("Intrinsic Matrix:")
            print(mtx)
            print("Distortion:")
            print(dist)
        else:
            print("Calibration Failed")



if __name__ == "__main__":
    
    CamCal = CameraCalibration(source="http://10.0.0.90:5000/video_feed")
    CamCal.calibrate()
    
    
