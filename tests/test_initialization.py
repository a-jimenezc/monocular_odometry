import pytest
import numpy as np
import cv2
from src.initialization import pose_initialization, bundle_adjustment

# Fixtures
@pytest.fixture
def sample_keyframes():
    """Generate mock keyframes with consistent points and descriptors."""
    keypoints_1 = np.random.rand(100, 2) * 100  # 100 random 2D points for keyframe 1
    keypoints_2 = keypoints_1 + np.random.normal(0, 0.5, keypoints_1.shape)  # Slightly shifted
    keypoints_3 = keypoints_1 + np.random.normal(0, 0.5, keypoints_1.shape)

    descriptors = np.random.rand(100, 128)  # Random SIFT-like descriptors

    return [
        {"points": keypoints_1, "descriptors": descriptors},
        {"points": keypoints_2, "descriptors": descriptors},
        {"points": keypoints_3, "descriptors": descriptors},
    ]

@pytest.fixture
def sample_intrinsics():
    """Provide a sample intrinsic matrix."""
    return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Typical pinhole camera model

@pytest.fixture
def sample_poses():
    """Provide sample initial poses."""
    R = np.eye(3)
    t = np.zeros(3)
    return [{"R": R, "t": t}, {"R": R, "t": t}, {"R": R, "t": t}]

@pytest.fixture
def sample_points_3d():
    """Generate random 3D points."""
    return np.random.rand(100, 3) * 10  # 100 points in a 10x10x10 cube

# Tests
def test_keyframe_structure(sample_keyframes):
    """Test if keyframes have consistent points and descriptors."""
    for kf in sample_keyframes:
        assert "points" in kf, "Keyframe missing 'points'."
        assert "descriptors" in kf, "Keyframe missing 'descriptors'."
        assert kf["points"].shape[0] == kf["descriptors"].shape[0], (
            "Number of points and descriptors do not match."
        )

def test_pose_initialization_structure(sample_intrinsics, sample_keyframes):
    """Test the output structure of pose_initialization."""
    optimized_poses, optimized_points = pose_initialization(sample_intrinsics, sample_keyframes)

    # Check pose structure
    for pose in optimized_poses:
        assert "R" in pose and "t" in pose, "Pose missing 'R' or 't'."
        assert pose["R"].shape == (3, 3), "Rotation matrix has incorrect shape."
        assert pose["t"].shape == (3,), "Translation vector has incorrect shape."

    # Check 3D points
    assert optimized_points.shape[1] == 3, "3D points are not Nx3."

def test_common_inliers(sample_intrinsics, sample_keyframes):
    """Test if the common inliers are consistently filtered."""
    _, inliers1 = cv2.findFundamentalMat(sample_keyframes[0]["points"], sample_keyframes[1]["points"], cv2.FM_RANSAC)
    _, inliers2 = cv2.findFundamentalMat(sample_keyframes[0]["points"], sample_keyframes[2]["points"], cv2.FM_RANSAC)

    common_inliers = inliers1.ravel() & inliers2.ravel()
    pts1_common = sample_keyframes[0]["points"][common_inliers > 0]
    pts2_common = sample_keyframes[1]["points"][common_inliers > 0]
    pts3_common = sample_keyframes[2]["points"][common_inliers > 0]

    assert len(pts1_common) == len(pts2_common) == len(pts3_common), "Common inliers count mismatch."

def test_bundle_adjustment(sample_intrinsics, sample_poses, sample_points_3d, sample_keyframes):
    """Test bundle adjustment for consistent output."""
    optimized_poses, optimized_points = bundle_adjustment(
        sample_intrinsics, sample_poses, sample_points_3d, sample_keyframes
    )

    # Check optimized poses
    for pose in optimized_poses:
        assert "R" in pose and "t" in pose, "Optimized pose missing 'R' or 't'."
        assert pose["R"].shape == (3, 3), "Optimized rotation matrix has incorrect shape."
        assert pose["t"].shape == (3,), "Optimized translation vector has incorrect shape."

    # Check optimized 3D points
    assert optimized_points.shape[1] == 3, "Optimized 3D points are not Nx3."

def test_insufficient_keyframes(sample_intrinsics, sample_keyframes):
    """Test for failure when fewer than 3 keyframes are provided."""
    with pytest.raises(ValueError):
        pose_initialization(sample_intrinsics, sample_keyframes[:2])  # Only 2 keyframes

def test_no_inliers_case(sample_intrinsics):
    """Test behavior when no inliers are found."""
    keyframes = [
        {"points": np.random.rand(10, 2) * 100, "descriptors": np.random.rand(10, 128)},
        {"points": np.random.rand(10, 2) * 100, "descriptors": np.random.rand(10, 128)},
        {"points": np.random.rand(10, 2) * 100, "descriptors": np.random.rand(10, 128)},
    ]
    with pytest.raises(AssertionError):
        pose_initialization(sample_intrinsics, keyframes)

def test_large_noise_in_points(sample_intrinsics, sample_keyframes):
    """Test pose initialization under noisy points."""
    noisy_keyframes = [
        {"points": kf["points"] + np.random.normal(0, 10, kf["points"].shape), "descriptors": kf["descriptors"]}
        for kf in sample_keyframes
    ]
    optimized_poses, optimized_points = pose_initialization(sample_intrinsics, noisy_keyframes)

    # Ensure output is still structured correctly
    assert len(optimized_poses) == 3, "Incorrect number of optimized poses."
    assert optimized_points.shape[1] == 3, "Optimized 3D points are not Nx3."
    