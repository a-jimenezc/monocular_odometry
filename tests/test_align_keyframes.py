import numpy as np
import pytest
from src.initialization import align_keyframes, keyframe_matcher  # Replace `your_module` with the actual module name


def test_align_keyframes():
    """
    Test the `align_keyframes` function with handcrafted keyframes.
    """

    # Handcrafted keyframe data
    keyframe0 = {
        "points": np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),  # 4 points
        "descriptors": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32),  # 4 descriptors
    }

    keyframe1 = {
        "points": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),  # Overlaps with 1 point from keyframe0
        "descriptors": np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32),  # Corresponding descriptors
    }

    keyframe2 = {
        "points": np.array([[2, 2], [3, 3], [4, 4], [5, 5]]),  # Overlaps with 1 point from keyframe1
        "descriptors": np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=np.float32),  # Corresponding descriptors
    }

    # Perform alignment
    aligned_keyframe0, aligned_keyframe1, aligned_keyframe2 = align_keyframes(keyframe0, keyframe1, keyframe2)

    # Validate the output
    # Only common points across all keyframes should be retained
    assert aligned_keyframe0["points"].shape[0] == 2, "Incorrect number of aligned points in keyframe0."
    assert aligned_keyframe1["points"].shape[0] == 2, "Incorrect number of aligned points in keyframe1."
    assert aligned_keyframe2["points"].shape[0] == 2, "Incorrect number of aligned points in keyframe2."

    # Validate retained points and descriptors
    np.testing.assert_array_equal(aligned_keyframe0["points"], np.array([[2, 2], [3, 3]]))
    np.testing.assert_array_equal(aligned_keyframe1["points"], np.array([[2, 2], [3, 3]]))
    np.testing.assert_array_equal(aligned_keyframe2["points"], np.array([[2, 2], [3, 3]]))

    np.testing.assert_array_equal(aligned_keyframe0["descriptors"], np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32))
    np.testing.assert_array_equal(aligned_keyframe1["descriptors"], np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32))
    np.testing.assert_array_equal(aligned_keyframe2["descriptors"], np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32))

    print("Test passed for align_keyframes!")

def test_align_keyframes_basic():
    """
    Test basic alignment with partial overlaps.
    """
    keyframe0 = {
        "points": np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32),
        "descriptors": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
    }
    keyframe1 = {
        "points": np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32),
        "descriptors": np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
    }
    keyframe2 = {
        "points": np.array([[2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32),
        "descriptors": np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=np.float32),
    }

    aligned_keyframe0, aligned_keyframe1, aligned_keyframe2 = align_keyframes(keyframe0, keyframe1, keyframe2)

    assert aligned_keyframe0["points"].shape[0] == 2, "Incorrect number of aligned points."
    np.testing.assert_array_equal(aligned_keyframe0["points"], np.array([[2, 2], [3, 3]], dtype=np.float32))

def test_align_keyframes_full_overlap():
    """
    Test alignment where all points are common between keyframes.
    """
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    descriptors = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

    keyframe0 = {"points": points, "descriptors": descriptors}
    keyframe1 = {"points": points, "descriptors": descriptors}
    keyframe2 = {"points": points, "descriptors": descriptors}

    aligned_keyframe0, aligned_keyframe1, aligned_keyframe2 = align_keyframes(keyframe0, keyframe1, keyframe2)

    assert aligned_keyframe0["points"].shape[0] == 4, "All points should be retained."
    np.testing.assert_array_equal(aligned_keyframe0["points"], points)

def test_align_keyframes_no_overlap():
    """
    Test alignment where no points are common between keyframes.
    """
    keyframe0 = {
        "points": np.array([[0, 0], [1, 1]], dtype=np.float32),
        "descriptors": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    keyframe1 = {
        "points": np.array([[2, 2], [3, 3]], dtype=np.float32),
        "descriptors": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
    }
    keyframe2 = {
        "points": np.array([[4, 4], [5, 5]], dtype=np.float32),
        "descriptors": np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32),
    }

    with pytest.raises(ValueError):
        align_keyframes(keyframe0, keyframe1, keyframe2)

def test_align_keyframes_insufficient_matches():
    """
    Test alignment with insufficient matches.
    """
    keyframe0 = {
        "points": np.array([[0, 0]], dtype=np.float32),
        "descriptors": np.array([[0.1, 0.2]], dtype=np.float32),
    }
    keyframe1 = {
        "points": np.array([[1, 1]], dtype=np.float32),
        "descriptors": np.array([[0.3, 0.4]], dtype=np.float32),
    }
    keyframe2 = {
        "points": np.array([[2, 2]], dtype=np.float32),
        "descriptors": np.array([[0.5, 0.6]], dtype=np.float32),
    }

    with pytest.raises(ValueError):
        align_keyframes(keyframe0, keyframe1, keyframe2)

def test_align_keyframes_empty_keyframes():
    """
    Test alignment with empty keyframes.
    """
    keyframe0 = {"points": np.empty((0, 2), dtype=np.float32), "descriptors": np.empty((0, 2), dtype=np.float32)}
    keyframe1 = {"points": np.empty((0, 2), dtype=np.float32), "descriptors": np.empty((0, 2), dtype=np.float32)}
    keyframe2 = {"points": np.empty((0, 2), dtype=np.float32), "descriptors": np.empty((0, 2), dtype=np.float32)}

    with pytest.raises(ValueError):
        align_keyframes(keyframe0, keyframe1, keyframe2)

def test_align_keyframes_sanity_check_failure():
    """
    Test sanity check failure due to mismatched points.
    """
    keyframe0 = {
        "points": np.array([[0, 0], [1, 1]], dtype=np.float32),
        "descriptors": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    keyframe1 = {
        "points": np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
        "descriptors": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
    }
    keyframe2 = {
        "points": np.array([[2, 2], [3, 3]], dtype=np.float32),
        "descriptors": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
    }

    with pytest.raises(ValueError):
        align_keyframes(keyframe0, keyframe1, keyframe2)

if __name__ == "__main__":
    pytest.main()
