import pytest
import numpy as np
import cv2
from src.initialization import initial_keyframes


# Helper function to create synthetic video frames
def create_synthetic_video(num_frames, frame_size=(480, 640), pattern="static"):
    """
    Generate synthetic video frames for testing.
    
    Args:
    - num_frames: Number of frames to generate.
    - frame_size: Size of each frame (height, width).
    - pattern: Type of synthetic pattern ("static", "moving", "empty").
    
    Returns:
    - List of video frames.
    """
    if pattern == "static":
        # All frames are identical
        base_frame = np.random.randint(0, 256, frame_size, dtype=np.uint8)
        return [base_frame for _ in range(num_frames)]
    elif pattern == "moving":
        # Frames with a moving pattern
        frames = []
        for i in range(num_frames):
            frame = np.zeros(frame_size, dtype=np.uint8)
            cv2.circle(frame, (50 + i * 10, 50 + i * 10), 20, 255, -1)
            frames.append(frame)
        return frames
    elif pattern == "empty":
        # All frames are blank
        return [np.zeros(frame_size, dtype=np.uint8) for _ in range(num_frames)]
    else:
        raise ValueError("Unknown pattern type")


# Tests for initial_keyframes

def test_initial_keyframes_basic():
    # Test with a simple moving pattern
    video_frames = create_synthetic_video(num_frames=10, pattern="moving")
    
    keyframes = initial_keyframes(video_frames, threshold=10)
    
    assert len(keyframes) <= 3, "Number of keyframes exceeds the limit."
    for keyframe in keyframes:
        assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
        assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
        assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


def test_initial_keyframes_static():
    # Test with static frames
    video_frames = create_synthetic_video(num_frames=10, pattern="static")
    
    keyframes = initial_keyframes(video_frames, threshold=10)
    
    assert len(keyframes) <= 3, "Number of keyframes exceeds the limit."
    assert len(keyframes) == 1, "Static frames should produce only one keyframe."
    for keyframe in keyframes:
        assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
        assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
        assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


def test_initial_keyframes_empty():
    # Test with empty frames
    video_frames = create_synthetic_video(num_frames=10, pattern="empty")
    
    keyframes = initial_keyframes(video_frames, threshold=10)
    
    assert len(keyframes) == 0, "Empty frames should produce no keyframes."


def test_initial_keyframes_varying_threshold():
    # Test with varying thresholds
    video_frames = create_synthetic_video(num_frames=10, pattern="moving")
    
    for threshold in [1, 5, 10, 20]:
        keyframes = initial_keyframes(video_frames, threshold=threshold)
        assert len(keyframes) <= 3, f"Number of keyframes exceeds limit for threshold {threshold}."
        for keyframe in keyframes:
            assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
            assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
            assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


def test_initial_keyframes_no_variation():
    # Test with identical frames
    video_frames = create_synthetic_video(num_frames=10, pattern="static")
    
    keyframes = initial_keyframes(video_frames, threshold=3)
    
    assert len(keyframes) == 1, "Identical frames should produce one keyframe."
    for keyframe in keyframes:
        assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
        assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
        assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


def test_initial_keyframes_insufficient_frames():
    # Test with fewer than 3 frames
    video_frames = create_synthetic_video(num_frames=2, pattern="moving")
    
    keyframes = initial_keyframes(video_frames, threshold=3)
    
    assert len(keyframes) <= 2, "Number of keyframes should not exceed the number of frames."
    for keyframe in keyframes:
        assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
        assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
        assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


def test_initial_keyframes_large_dataset():
    # Test with a large number of frames
    video_frames = create_synthetic_video(num_frames=100, pattern="moving")
    
    keyframes = initial_keyframes(video_frames, threshold=3)
    
    assert len(keyframes) <= 3, "Number of keyframes exceeds the limit."
    for keyframe in keyframes:
        assert "points" in keyframe and "descriptors" in keyframe, "Keyframe missing required data."
        assert keyframe["points"].shape[0] > 0, "Keyframe has no points."
        assert keyframe["descriptors"].shape[0] > 0, "Keyframe has no descriptors."


if __name__ == "__main__":
    pytest.main()