import pytest
import numpy as np
import cv2
from src.initialization import compute_essential_matrix, keyframe_matcher  # Replace `your_module` with your actual module name


# Helper function to create synthetic keyframes
def create_synthetic_keyframe(points, descriptors=None):
    """
    Create a synthetic keyframe with feature points and descriptors.

    Args:
    - points: List or array of 2D points.
    - descriptors: Descriptors corresponding to the points (optional).

    Returns:
    - A keyframe dictionary with points and descriptors.
    """
    if descriptors is None:
        descriptors = np.random.rand(len(points), 128).astype(np.float32)  # Example 128-dim descriptors
    return {"points": np.array(points), "descriptors": np.array(descriptors)}


# Helper function to create synthetic intrinsic matrix
def create_intrinsic_matrix():
    return np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])


# Helper function to generate matched keyframes with noise
def generate_matched_keyframes(num_points=20, noise=0.0):
    points1 = np.random.rand(num_points, 2) * 640  # Random 2D points in a 640x480 frame
    points2 = points1 + np.random.normal(0, noise, points1.shape)  # Add noise for variation
    descriptors1 = np.random.rand(num_points, 128).astype(np.float32)
    descriptors2 = np.random.rand(num_points, 128).astype(np.float32)
    keyframe1 = create_synthetic_keyframe(points1, descriptors1)
    keyframe2 = create_synthetic_keyframe(points2, descriptors2)
    return keyframe1, keyframe2


# Test cases for `compute_essential_matrix`
def test_compute_essential_matrix_basic():
    K = create_intrinsic_matrix()
    keyframe1, keyframe2 = generate_matched_keyframes(num_points=50, noise=1.0)

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K)

    assert E.shape == (3, 3), "Essential matrix must be 3x3."
    assert np.isfinite(E).all(), "Essential matrix contains invalid values."
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."
    assert t.shape == (3, 1), "Translation vector must be 3x1."
    assert len(inlier_keyframe1["points"]) == len(inlier_keyframe2["points"]), "Inlier counts must match."


def test_compute_essential_matrix_insufficient_points():
    K = create_intrinsic_matrix()
    keyframe1, keyframe2 = generate_matched_keyframes(num_points=5)  # Insufficient points

    with pytest.raises(cv2.error):
        compute_essential_matrix(keyframe1, keyframe2, K)


def test_compute_essential_matrix_degenerate_configuration():
    K = create_intrinsic_matrix()
    points = np.random.rand(8, 2) * 640  # 8 identical points
    keyframe1 = create_synthetic_keyframe(points)
    keyframe2 = create_synthetic_keyframe(points)

    with pytest.raises(cv2.error):
        compute_essential_matrix(keyframe1, keyframe2, K)


def test_compute_essential_matrix_noisy_data():
    K = create_intrinsic_matrix()
    keyframe1, keyframe2 = generate_matched_keyframes(num_points=50, noise=5.0)

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K)

    assert E.shape == (3, 3), "Essential matrix must be 3x3."
    assert len(inlier_keyframe1["points"]) > 0, "No inliers were detected."
    assert len(inlier_keyframe2["points"]) > 0, "No inliers were detected."


def test_compute_essential_matrix_large_dataset():
    K = create_intrinsic_matrix()
    keyframe1, keyframe2 = generate_matched_keyframes(num_points=500, noise=1.0)

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K)

    assert E.shape == (3, 3), "Essential matrix must be 3x3."
    assert len(inlier_keyframe1["points"]) > 0, "No inliers were detected."
    assert len(inlier_keyframe2["points"]) > 0, "No inliers were detected."


# Test cases for `keyframe_matcher` subfunction
def test_keyframe_matcher_basic():
    keyframe1, keyframe2 = generate_matched_keyframes(num_points=20)

    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    assert len(matched_keyframe1["points"]) == len(matched_keyframe2["points"]), "Number of matched points must be equal."
    assert matched_keyframe1["descriptors"].shape == matched_keyframe2["descriptors"].shape, "Descriptor shapes must match."


def test_keyframe_matcher_no_matches():
    keyframe1 = create_synthetic_keyframe(points=np.random.rand(10, 2), descriptors=np.random.rand(10, 128))
    keyframe2 = create_synthetic_keyframe(points=np.random.rand(10, 2), descriptors=np.random.rand(10, 128) + 1000)  # No overlap

    matched_keyframe1, matched_keyframe2 = keyframe_matcher(keyframe1, keyframe2)

    assert len(matched_keyframe1["points"]) == 0, "No matches expected."
    assert len(matched_keyframe2["points"]) == 0, "No matches expected."


if __name__ == "__main__":
    pytest.main()
