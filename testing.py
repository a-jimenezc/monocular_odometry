import numpy as np
import cv2
import numpy as np

# Define 3D points in world coordinates
points_3d = np.array([
    [1, 1, 5],
    [2, 2, 10],
    [-1, -1, 7],
    [3, -2, 15]
])  # Shape: (4, 3)

# Define intrinsic matrix (same for all views)
K = np.array([
    [1000, 0, 320],
    [0, 1000, 240],
    [0, 0, 1]
])  # Intrinsic matrix

# Define camera poses
pose1 = {"R": np.eye(3), "t": np.array([0, 0, 0])}  # Camera at origin
pose2 = {"R": np.array([[0.866, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.866]]), "t": np.array([-2, 0, 0])}  # Rotated and translated
pose3 = {"R": np.array([[0.707, 0, 0.707], [0, 1, 0], [-0.707, 0, 0.707]]), "t": np.array([1, -1, 1])}  # Rotated and translated

# Compute projection matrices
def compute_projection_matrix(K, pose):
    R, t = pose["R"], pose["t"].reshape(-1, 1)
    R_inv = R.T
    t_inv = -R_inv @ t
    return K @ np.hstack((R_inv, t_inv))

P1 = compute_projection_matrix(K, pose1)
P2 = compute_projection_matrix(K, pose2)
P3 = compute_projection_matrix(K, pose3)

# Convert 3D points to homogeneous coordinates
points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Shape: (4, 4)

# Project points into each view
projections1 = P1 @ points_homogeneous.T
projections2 = P2 @ points_homogeneous.T
projections3 = P3 @ points_homogeneous.T

# Normalize by z-coordinate to get 2D points
points_2d_view1 = (projections1[:2] / projections1[2]).T  # Shape: (4, 2)
points_2d_view2 = (projections2[:2] / projections2[2]).T  # Shape: (4, 2)
points_2d_view3 = (projections3[:2] / projections3[2]).T  # Shape: (4, 2)

# Print results
print("3D Points:")
print(points_3d)

print("\n2D Points in View 1:")
print(points_2d_view1.tolist())

print("\n2D Points in View 2:")
print(points_2d_view2.tolist())

print("\n2D Points in View 3:")
print(points_2d_view3.tolist())


# Mock keyframes with points and descriptors
keyframes = [
    {"points": np.array(points_2d_view1.tolist()),
        "descriptors": np.array([[0.2, 0.3], [0.1, 0.2], [0.3, 0.4], [0.6, 0.5]]).astype(np.float32)},
    {"points": np.array(points_2d_view2.tolist()),
        "descriptors": np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]).astype(np.float32)},
    {"points": np.array(points_2d_view3.tolist()),
        "descriptors": np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]).astype(np.float32)}
]
keyframe1, keyframe2, keyframe3 = keyframes[-3:]

# Match points between keyframes
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_12 = bf.match(keyframe1["descriptors"], keyframe2["descriptors"])
matches_23 = bf.match(keyframe2["descriptors"], keyframe3["descriptors"])

# Filter matches present across all three keyframes
common_matches = {}
for m in matches_12:
    common_matches[m.trainIdx] = {"query_idx_12": m.queryIdx, "train_idx_23": None}

for m in matches_23:
    if m.queryIdx in common_matches:
        common_matches[m.queryIdx]["train_idx_23"] = m.trainIdx

valid_matches = []
for train_idx_12, match_data in common_matches.items():
    if match_data["train_idx_23"] is not None:
        valid_matches.append((match_data["query_idx_12"], train_idx_12, match_data["train_idx_23"]))
print(valid_matches)
# Extract aligned 2D points
points1 = np.array([keyframe1["points"][query_idx_12] for query_idx_12, _, _ in valid_matches])
points2 = np.array([keyframe2["points"][train_idx1] for _, train_idx1, _ in valid_matches])
points3 = np.array([keyframe3["points"][train_idx_23] for _, _, train_idx_23 in valid_matches])
# Display the results
print("Aligned Points in Keyframe 1:")
print(points1)
print("Aligned Points in Keyframe 2:")
print(points2)
print("Aligned Points in Keyframe 3:")
print(points3)
print('len points', len(points1))
points_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
print('triangulated points', len(points_homogeneous.T))

