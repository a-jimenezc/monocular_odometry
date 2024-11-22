import pytest
import numpy as np
import cv2
from scipy.optimize import least_squares
from src.bundle_adjustment import project, reprojection_error, bundle_adjustment


# Helper functions for creating test data
def create_test_camera_pose():
    R = np.eye(3)  # Identity rotation
    t = np.array([0, 0, -5])  # Translation along Z-axis
    return {"R": R, "t": t}


def create_test_intrinsic():
    return np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])


def create_test_3d_points(num_points=5):
    return np.random.uniform(-1, 1, (num_points, 3)) + np.array([0, 0, 5])


def create_test_keyframes(camera_pose, K, points_3d):
    projected_points = project(points_3d, camera_pose, K)
    return [{"points": projected_points}]


# Tests for `project`
def test_project_basic():
    K = create_test_intrinsic()
    camera_pose = create_test_camera_pose()
    points_3d = create_test_3d_points()

    projected_points = project(points_3d, camera_pose, K)

    assert projected_points.shape == (points_3d.shape[0], 2), "Projection output shape mismatch."
    assert not np.any(np.isnan(projected_points)), "Projection contains NaN values."


def test_project_degenerate_case():
    K = create_test_intrinsic()
    camera_pose = create_test_camera_pose()
    points_3d = np.zeros((1, 3))  # A single point at the origin

    projected_points = project(points_3d, camera_pose, K)

    assert projected_points.shape == (1, 2), "Degenerate case projection shape mismatch."
    assert np.all(projected_points >= 0), "Projected points should be valid."


def test_project_invalid_pose():
    K = create_test_intrinsic()
    camera_pose = {"R": np.eye(3), "t": np.array([0, 0])}  # Invalid translation vector
    points_3d = create_test_3d_points()

    with pytest.raises(ValueError):
        project(points_3d, camera_pose, K)


# Tests for `reprojection_error`
def test_reprojection_error():
    K = create_test_intrinsic()
    camera_pose = create_test_camera_pose()
    points_3d = create_test_3d_points()
    keyframes = create_test_keyframes(camera_pose, K, points_3d)

    # Flattened parameters for the camera and points
    rvec, _ = cv2.Rodrigues(camera_pose["R"])
    tvec = camera_pose["t"]
    params = np.hstack((rvec.ravel(), tvec, points_3d.ravel()))

    num_cameras = 1
    num_points = points_3d.shape[0]

    residuals = reprojection_error(params, num_cameras, num_points, K, keyframes)

    assert residuals.shape[0] == num_points * 2, "Residuals should have 2 entries per point."
    assert not np.any(np.isnan(residuals)), "Residuals contain NaN values."


def test_reprojection_error_mismatch():
    K = create_test_intrinsic()
    camera_pose = create_test_camera_pose()
    points_3d = create_test_3d_points()
    keyframes = create_test_keyframes(camera_pose, K, points_3d)

    rvec, _ = cv2.Rodrigues(camera_pose["R"])
    tvec = camera_pose["t"]
    params = np.hstack((rvec.ravel(), tvec, points_3d.ravel()))

    # Introduce mismatch in keyframe points
    keyframes[0]["points"] = np.random.uniform(0, 640, (10, 2))

    num_cameras = 1
    num_points = points_3d.shape[0]

    with pytest.raises(ValueError):
        reprojection_error(params, num_cameras, num_points, K, keyframes)


# Tests for `bundle_adjustment`
def test_bundle_adjustment_basic():
    K = create_test_intrinsic()
    camera_poses = [create_test_camera_pose()]
    points_3d = create_test_3d_points()
    keyframes = create_test_keyframes(camera_poses[0], K, points_3d)

    optimized_camera_poses, optimized_points_3d = bundle_adjustment(K, camera_poses, points_3d, keyframes)

    assert len(optimized_camera_poses) == len(camera_poses), "Number of optimized poses mismatch."
    assert optimized_points_3d.shape == points_3d.shape, "Optimized points shape mismatch."


def test_bundle_adjustment_noisy_observations():
    K = create_test_intrinsic()
    camera_poses = [create_test_camera_pose()]
    points_3d = create_test_3d_points()
    keyframes = create_test_keyframes(camera_poses[0], K, points_3d)

    # Add noise to observations
    keyframes[0]["points"] += np.random.normal(0, 0.5, keyframes[0]["points"].shape)

    optimized_camera_poses, optimized_points_3d = bundle_adjustment(K, camera_poses, points_3d, keyframes)

    assert len(optimized_camera_poses) == len(camera_poses), "Number of optimized poses mismatch."
    assert optimized_points_3d.shape == points_3d.shape, "Optimized points shape mismatch."


def test_bundle_adjustment_invalid_data():
    K = create_test_intrinsic()
    camera_poses = [create_test_camera_pose()]
    points_3d = create_test_3d_points()
    keyframes = create_test_keyframes(camera_poses[0], K, points_3d)

    # Remove keyframe data
    keyframes[0]["points"] = np.array([])

    with pytest.raises(ValueError):
        bundle_adjustment(K, camera_poses, points_3d, keyframes)


if __name__ == "__main__":
    pytest.main()
    