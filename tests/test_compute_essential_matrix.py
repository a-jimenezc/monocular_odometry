import pytest
import numpy as np
import cv2
from src.initialization import compute_essential_matrix, keyframe_matcher

# Helper function to create synthetic keyframes
def create_synthetic_keyframe(num_points=100, image_size=(640, 480), noise_level=0.0):
    """
    Create a synthetic keyframe with random points.

    Args:
    - num_points: Number of keypoints to generate.
    - image_size: Size of the synthetic image (width, height).
    - noise_level: Level of noise to add to the points.

    Returns:
    - A synthetic keyframe containing random points and dummy descriptors.
    """
    points = np.random.randint(0, image_size[0], (num_points, 2))
    if noise_level > 0:
        points = np.clip(
    np.round(points + np.random.normal(0, noise_level, points.shape)),
    0,
    image_size[0]).astype(int)

    descriptors = np.random.rand(num_points, 128).astype(np.float32)  # Dummy SIFT-like descriptors
    return {"points": points, "descriptors": descriptors}


# Test cases for compute_essential_matrix

def test_compute_essential_matrix_valid_inputs():
    """
    Test with two valid keyframes.
    """
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])  # Intrinsic matrix

    # Create synthetic keyframes with overlapping points
    keyframe1 = create_synthetic_keyframe(num_points=100)
    keyframe2 = create_synthetic_keyframe(num_points=100)

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0)

    # Validate results
    assert E.shape == (3, 3), "Essential matrix should be 3x3."
    assert R.shape == (3, 3), "Rotation matrix should be 3x3."
    assert t.shape == (3, 1), "Translation vector should be 3x1."
    assert len(inlier_keyframe1["points"]) == len(inlier_keyframe2["points"]), "Inliers should be matched."

    # Essential matrix should be rank 2
    assert np.linalg.matrix_rank(E) == 2, "Essential matrix should have rank 2."


def test_compute_essential_matrix_no_matches():
    """
    Test with keyframes that have no matching points.
    """
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    keyframe1 = create_synthetic_keyframe(num_points=100)
    keyframe2 = create_synthetic_keyframe(num_points=100)

    # Move keyframe2 points far away to avoid matches
    keyframe2["points"] += 1000

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0)

    # Validate results
    assert E is not None, "Essential matrix should still be computed."
    assert len(inlier_keyframe1["points"]) < 40, "Unexpected number of inliers for no matches."
    assert len(inlier_keyframe2["points"]) < 40, "Unexpected number of inliers for no matches."


def test_compute_essential_matrix_degenerate_case():
    """
    Test with insufficient points.
    """
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    # Create keyframes with too few points
    keyframe1 = create_synthetic_keyframe(num_points=3)  # Less than 5 points
    keyframe2 = create_synthetic_keyframe(num_points=3)

    with pytest.raises(ValueError, match="Insufficient points for essential matrix computation"):
        compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0)


def test_compute_essential_matrix_noisy_inputs():
    """
    Test with noisy inputs.
    """
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    # Create noisy keyframes
    keyframe1 = create_synthetic_keyframe(num_points=100, noise_level=5.0)
    keyframe2 = create_synthetic_keyframe(num_points=100, noise_level=5.0)

    E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0)

    # Validate results
    assert E.shape == (3, 3), "Essential matrix should be 3x3."
    assert len(inlier_keyframe1["points"]) > 0, "Some inliers should be detected."


def test_compute_essential_matrix_invalid_intrinsics():
    """
    Test with invalid intrinsic matrix.
    """
    K = np.zeros((3, 3))  # Invalid intrinsic matrix

    keyframe1 = create_synthetic_keyframe(num_points=100)
    keyframe2 = create_synthetic_keyframe(num_points=100)

    with pytest.raises(ValueError):
        compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=1.0)


def test_compute_essential_matrix_varying_ransac_threshold():
    """
    Test with different RANSAC thresholds.
    """
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    keyframe1 = create_synthetic_keyframe(num_points=100)
    keyframe2 = create_synthetic_keyframe(num_points=100)

    for threshold in [0.1, 1.0, 10.0]:
        E, R, t, inlier_keyframe1, inlier_keyframe2 = compute_essential_matrix(keyframe1, keyframe2, K, ransac_threshold=threshold)
        assert E.shape == (3, 3), f"Essential matrix should be 3x3 for threshold {threshold}."
        assert len(inlier_keyframe1["points"]) == len(inlier_keyframe2["points"]), "Inliers should be matched."


if __name__ == "__main__":
    pytest.main()
