import numpy as np
import pytest
import cv2
from src.initialization import triangulate_points

def test_triangulate_points():
    """
    Test the `triangulate_points` function with simple, handcrafted data.
    """
    # Define two projection matrices (P1 and P2)
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])  # Camera 1 at origin
    P2 = np.array([[1, 0, 0, -1],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])  # Camera 2 translated along X-axis by -1

    # Define 3D points in world space
    points_3d_gt = np.array([
        [1, 1, 5],  # 3D point at (1, 1, 5)
        [2, 2, 10],  # 3D point at (2, 2, 10)
    ])

    # Project the 3D points into the two camera views
    def project_point(P, point):
        point_homogeneous = np.append(point, 1)  # Convert to homogeneous
        projection = P @ point_homogeneous
        return projection[:2] / projection[2]  # Normalize by z

    pts1 = np.array([project_point(P1, p) for p in points_3d_gt])  # Camera 1 projections
    pts2 = np.array([project_point(P2, p) for p in points_3d_gt])  # Camera 2 projections

    # Call the triangulate_points function
    points_3d_estimated = triangulate_points(P1, P2, pts1, pts2)

    # Assert the results are close to the ground truth
    assert points_3d_estimated.shape == points_3d_gt.shape, "Output shape mismatch."
    np.testing.assert_allclose(points_3d_estimated, points_3d_gt, atol=1e-6, err_msg="Triangulated points do not match ground truth.")

    print("Triangulation test passed!")

# Run the test
if __name__ == "__main__":
    test_triangulate_points()
