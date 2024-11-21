import pytest
import cv2
import numpy as np
from src.video_data_handler import VideoDataHandler
from src.features import KeyframeGenerator


# Helper function to create synthetic video data
def create_synthetic_video(output_path, frame_size=(100, 100), num_frames=10, movement=True):
    """
    Creates a synthetic video for testing purposes.
    Args:
        output_path (str): Path to save the video.
        frame_size (tuple): Frame dimensions (width, height).
        num_frames (int): Number of frames in the video.
        movement (bool): Whether to include synthetic motion.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, frame_size)
    
    for i in range(num_frames):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        if movement:
            cv2.circle(frame, (10 + i * 5, frame_size[1] // 2), 10, (255, 255, 255), -1)
        else:
            cv2.circle(frame, (frame_size[0] // 2, frame_size[1] // 2), 10, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()


# Test cases
@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "synthetic_video.mp4"
    create_synthetic_video(str(video_path))
    return str(video_path)


def test_keyframe_generator_with_synthetic_data(synthetic_video):
    """
    Test KeyframeGenerator with synthetic video data.
    """
    video_handler = VideoDataHandler(synthetic_video, grayscale=True)
    keyframe_generator = KeyframeGenerator(threshold=0.5)
    
    keyframes = keyframe_generator.generate_keyframes(video_handler, max_keyframes=3)
    
    video_handler.release()
    
    # Assertions
    assert len(keyframes) > 0, "No keyframes were generated"
    assert all("points" in kf and "descriptors" in kf for kf in keyframes), "Invalid keyframe structure"
    assert all(len(kf["points"]) > 0 for kf in keyframes), "Keyframes should contain points"


@pytest.mark.parametrize("video_path", ["test_data/vid3.mp4"])
def test_keyframe_generator_with_external_video(video_path):
    """
    Test KeyframeGenerator with an external video file.
    """
    try:
        video_handler = VideoDataHandler(video_path, grayscale=True)
    except ValueError as e:
        pytest.skip(f"External video file could not be opened: {e}")
        return
    
    keyframe_generator = KeyframeGenerator(threshold=0.7)
    keyframes = keyframe_generator.generate_keyframes(video_handler, max_keyframes=3)
    
    video_handler.release()
    
    # Assertions
    assert len(keyframes) > 0, "No keyframes were generated"
    assert all("points" in kf and "descriptors" in kf for kf in keyframes), "Invalid keyframe structure"
    assert all(len(kf["points"]) > 0 for kf in keyframes), "Keyframes should contain points"
