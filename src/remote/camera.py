from picamera2 import Picamera2
import cv2
import time
import logging
import numpy as np
import time
import threading

class Camera:

    def __init__(self,grayscale=False, rectification=False):

        self.stream_active = False
        self.picam = None
        self.latest_frame = None
        self.lock = threading.Lock()

        self.fps = 30
        self.frame_width = 1280
        self.frame_height = 720
        self.fx = 552.8629348886877
        self.fy = 555.0655420698595
        self.cx = 645.2072488904695
        self.cy = 359.9575457536877
        self.camera_matrix = np.array([[self.fx, 0.0, self.cx], [0.0,self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeff = np.array([[0.049283385067669064], [-0.035488971050033756], [-0.000753093512931907], [0.0017761384160951163]], dtype=np.float32)
        self.grayscale = grayscale
        self.rectification = rectification

        h, w = (self.frame_height, self.frame_width)

        #self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, (w, h), 1 , (w, h))
        self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.camera_matrix,self.dist_coeff, (w, h), None)
        #self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeff, None, self.new_camera_matrix, (w, h), cv2.CV_16SC2)
        self.fisheye_map1, self.fisheye_map2 = cv2.fisheye.initUndistortRectifyMap(self.camera_matrix, self.dist_coeff, np.eye(3, 3), self.new_camera_matrix, (w,h), cv2.CV_16SC2)
        #self.start_camera()

    def start_camera(self):
        if not self.stream_active:
            if self.picam == None:
                self.picam = Picamera2()
                self.picam.configure(self.picam.create_video_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
                self.picam.start()
                threading.Thread(target=self._capture_frames, daemon=False).start()
            else:
                print("Camera Already started")
                self.picam.stop() # first stop stream before(a small bug here.)
                self.picam.start()
                threading.Thread(target=self._capture_frames, daemon=False).start()
        # if not self.stream_active:
        #     self.picam = Picamera2()
        #     self.picam.configure(self.picam.create_video_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
        #     self.picam.start()
        #     self.stream_active = True
        #     time.sleep(1)
        #     threading.Thread(target=self._capture_frames, daemon=True).start()

    def stop_camera(self):
        if self.stream_active:
            self.picam.stop_preview()
            self.picam = None
            self.stream_active = False

    def _capture_frames(self):
        while self.stream_active:
            frame = self.picam.capture_array()
            with self.lock:
                self.latest_frame = frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:

                if self.grayscale:
                    self.latest_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)

                if self.enable_rectification:
                    self.latest_frame =  cv2.remap(self.latest_frame, self.fisheye_map1, self.fisheye_map2, cv2.INTER_LINEAR)

                return cv2.imencode('.jpg', self.latest_frame)[1].tobytes()
            return None

    def generate_stream(self):
        """Yield frames for MJPEG streaming."""
        while self.stream_active:
            frame = self.get_latest_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break

    def generate_frames(self,fps=30):

        if self.picam == None:
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_video_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
            self.picam.start()
        else:
            print("Camera Already started")
            self.picam.stop() # first stop stream before(a small bug here.)
            self.picam.start()
        try:
            while True:
                # Capture frame from the camera
                frame = self.picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.rectification:
                    frame = cv2.remap(frame, self.fisheye_map1, self.fisheye_map2, cv2.INTER_LINEAR)

                ret, buffer = cv2.imencode('.jpg', frame)  # Encode to jpg
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        except Exception as e:
            print(f"Error during frame capture: {e}")

        self.picam.stop()
        print("Camera stopped.")