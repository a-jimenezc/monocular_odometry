import pytest
import numpy as np
from src.initialization import keyframe_matcher

# Helper functions for generating synthetic data
def create_keyframe(num_points=100):
    """
    Create a synthetic keyframe with random points and descriptors.

    Args:
    - num_points: Number of points and descriptors to generate.

    Returns:
    - Keyframe dictionary with "points" and "descriptors".
    """
    points = np.random.uniform(0, 640, (num_points, 2))  # Random 2D points
    descriptors = np.random.uniform(0, 256, (num_points, 128)).astype(np.float32)  # Random descriptors
    return {"points": points, "descriptors": descriptors}

def create_matching_keyframes(num_matches=50, num_outliers1=20, num_outliers2=20):
    """
    Create two keyframes with overlapping points and additional outliers.

    Args:
    - num_matches: Number of matching points between the keyframes.
    - num_outliers1: Additional points in keyframe1.
    - num_outliers2: Additional points in keyframe2.

    Returns:
    - keyframe1, keyframe2: Two synthetic keyframes with overlapping and unique points.
    """
    # Generate shared matching points
    matching_points = np.random.uniform(0, 640, (num_matches, 2))
    matching_descriptors = np.random.uniform(0, 256, (num_matches, 128)).astype(np.float32)

    # Generate unique points for each keyframe
    outlier_points1 = np.random.uniform(0, 640, (num_outliers1, 2))
    outlier_descriptors1 = np.random.uniform(0, 256, (num_outliers1, 128)).astype(np.float32)

    outlier_points2 = np.random.uniform(0, 640, (num_outliers2, 2))
    outlier_descriptors2 = np.random.uniform(0, 256, (num_outliers2, 128)).astype(np.float32)

    # Combine to form keyframes
    keyframe1 = {
        "points": np.vstack((matching_points, outlier_points1)),
        "descriptors": np.vstack((matching_descriptors, outlier_descriptors1))
    }
    keyframe2 = {
        "points": np.vstack((matching_points, outlier_points2)),
        "descriptors": np.vstack((matching_descriptors, outlier_descriptors2))
    }
    return keyframe1, keyframe2


# Tests for keyframe_matcher

def test_keyframe_matcher_basic():
    keyframe1, keyframe2 = create_matching_keyframes(num_matches=50)
    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    # Check that matched keyframes have the same number of points
    assert matched_keyframe1["points"].shape == matched_keyframe2["points"].shape, \
        "Number of matched points should be the same."
    assert matched_keyframe1["descriptors"].shape == matched_keyframe2["descriptors"].shape, \
        "Number of matched descriptors should be the same."
    
    # Verify that at least some matches were found
    assert matched_keyframe1["points"].shape[0] > 0, "No matches found."


def test_keyframe_matcher_no_matches():
    # Keyframes with no overlap in descriptors
    keyframe1 = create_keyframe(num_points=50)
    keyframe2 = create_keyframe(num_points=50)
    keyframe2["descriptors"] += 1000  # Ensure descriptors are disjoint

    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    # No matches should be found
    assert matched_keyframe1["points"].shape[0] == 0, "Unexpected matches found."
    assert matched_keyframe2["points"].shape[0] == 0, "Unexpected matches found."


def test_keyframe_matcher_with_outliers():
    keyframe1, keyframe2 = create_matching_keyframes(num_matches=50, num_outliers1=20, num_outliers2=30)
    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    # Check that matched keyframes have fewer points than the total in the input keyframes
    assert matched_keyframe1["points"].shape[0] <= keyframe1["points"].shape[0], \
        "Matched keyframe1 has more points than expected."
    assert matched_keyframe2["points"].shape[0] <= keyframe2["points"].shape[0], \
        "Matched keyframe2 has more points than expected."


def test_keyframe_matcher_empty_keyframes():
    # Empty keyframes
    keyframe1 = {"points": np.empty((0, 2)), "descriptors": np.empty((0, 128))}
    keyframe2 = {"points": np.empty((0, 2)), "descriptors": np.empty((0, 128))}

    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    # No matches should be found
    assert matched_keyframe1["points"].shape[0] == 0, "Matches found in empty keyframes."
    assert matched_keyframe2["points"].shape[0] == 0, "Matches found in empty keyframes."


def test_keyframe_matcher_invalid_keyframes():
    # Invalid descriptors
    keyframe1 = {"points": np.random.uniform(0, 640, (50, 2)), "descriptors": None}
    keyframe2 = {"points": np.random.uniform(0, 640, (50, 2)), "descriptors": None}

    with pytest.raises(TypeError):
        keyframe_matcher(keyframe1, keyframe2)


def test_keyframe_matcher_single_point():
    # Keyframes with a single point and descriptor
    keyframe1 = {"points": np.array([[320, 240]]), "descriptors": np.random.uniform(0, 256, (1, 128)).astype(np.float32)}
    keyframe2 = {"points": np.array([[320, 240]]), "descriptors": np.random.uniform(0, 256, (1, 128)).astype(np.float32)}

    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    # Verify that the single point matches
    assert matched_keyframe1["points"].shape[0] <= 1, "Too many matches for single-point keyframes."
    assert matched_keyframe2["points"].shape[0] <= 1, "Too many matches for single-point keyframes."


if __name__ == "__main__":
    pytest.main()
