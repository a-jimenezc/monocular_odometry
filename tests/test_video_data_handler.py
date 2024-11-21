import pytest
from src.video_data_handler import VideoDataHandler

@pytest.fixture
def video_file():
    """Fixture for the test video file path."""
    return "test_data/vid3.mp4"

def test_initialization(video_file):
    """Test initialization and metadata of the VideoDataHandler."""
    handler = VideoDataHandler(video_file, grayscale=False)
    assert handler.frame_width > 0, "Frame width should be greater than 0"
    assert handler.frame_height > 0, "Frame height should be greater than 0"
    handler.release()

def test_iteration(video_file):
    """Test iteration through the video frames."""
    handler = VideoDataHandler(video_file, grayscale=False)
    
    frame_count = 0
    for frame in handler:
        assert frame is not None, "Frame should not be None"
        assert frame.shape[0] == handler.frame_height, "Frame height mismatch"
        assert frame.shape[1] == handler.frame_width, "Frame width mismatch"
        frame_count += 1
        if frame_count > 10:  # Limit the number of frames tested
            break
    
    assert frame_count > 0, "Should iterate over at least one frame"
    handler.release()

def test_grayscale_conversion(video_file):
    """Test if the grayscale option works correctly."""
    handler = VideoDataHandler(video_file, grayscale=True)
    
    for frame in handler:
        assert len(frame.shape) == 2, "Frame should be grayscale (2D)"
        break  # Test only the first frame
    
    handler.release()

def test_end_of_stream(video_file):
    """Test if StopIteration is raised at the end of the stream."""
    handler = VideoDataHandler(video_file, grayscale=False)
    
    with pytest.raises(StopIteration):
        for _ in handler:
            pass
        next(handler)  # Attempting to read after the stream ends
    
    handler.release()

def test_invalid_source():
    """Test handling of an invalid video source."""
    with pytest.raises(ValueError, match="Cannot open video source"):
        VideoDataHandler("non_existent_video.mp4", grayscale=False)
