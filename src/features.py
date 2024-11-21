import cv2
import numpy as np

class KeyframeGenerator:
    def __init__(self, threshold=0.7):
        """
        threshold (float): Threshold for selecting keyframes based on feature differences.
        """
        self.feature_detector = cv2.SIFT_create()
        self.threshold = threshold
        self.previous_descriptors = None

    def generate_keyframes(self, video_handler, max_keyframes=3):
        """
        Generates keyframes from a video.
        Args:
            video_handler (VideoDataHandler): The video handler instance.
            max_keyframes (int): The maximum number of keyframes to select.
        
        Returns:
            List[dict]: List of keyframes with points and descriptors.
        """
        keyframes = []
        for frame in video_handler:
            keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
            if descriptors is None:
                continue
            
            # Compute similarity with previous frame's descriptors
            if self.previous_descriptors is not None:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(self.previous_descriptors, descriptors)
                match_distances = [m.distance for m in matches]
                
                # Check if there is a significant change to consider a keyframe
                if np.mean(match_distances) < self.threshold:
                    continue
            
            # Store keyframe data
            points = np.array([kp.pt for kp in keypoints])
            keyframes.append({
                "points": points,
                "descriptors": descriptors
            })
            
            self.previous_descriptors = descriptors
            
            # Limit to max_keyframes
            if len(keyframes) >= max_keyframes:
                break
        
        return keyframes
