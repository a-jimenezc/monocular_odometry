import cv2
import numpy as np
from src.point_descriptors import PointDescriptors

class EndOfFramesError(Exception):
    """Custom exception to indicate the end of the frames."""
    pass

class VideoDataHandler:
    def __init__(self, source, grayscale=True):
        """
        Initializes the VideoDataHandler.
        Args:
            video_path (str): Path to the video file.
            grayscale (bool): Whether to convert frames to grayscale.
        """
        self.source = source
        self.grayscale = grayscale
        self.capture = cv2.VideoCapture(source)
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.feature_detector = cv2.SIFT_create()

        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
    def __iter__(self):
        return self
    
    def __next__(self):
        """Reads the next frame from the video."""
        ret, frame = self.capture.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            self.capture.release()
            raise EndOfFramesError
        
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        points = np.array([kp.pt for kp in keypoints])
        frame = PointDescriptors(points, descriptors)
        return frame
    
    def release(self):
        self.capture.release()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/video_data_handler.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]

    try:
        handler = VideoDataHandler(video_file, grayscale=True)
        print(f"Displaying video: {video_file} in grayscale")
        for frame in handler:
            cv2.imshow("Grayscale Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except ValueError as e:
        print(e)
    finally:
        handler.release()
        cv2.destroyAllWindows()
