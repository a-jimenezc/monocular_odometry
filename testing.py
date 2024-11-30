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
        "descriptors": np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.6, 0.5]]).astype(np.float32)},
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

#points_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
# Triangulate between View 1 & View 2
points_homogeneous_12 = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
points_3d_12 = points_homogeneous_12[:3] / points_homogeneous_12[3]

# Triangulate between View 1 & View 3
points_homogeneous_13 = cv2.triangulatePoints(P1, P3, points1.T, points3.T)
points_3d_13 = points_homogeneous_13[:3] / points_homogeneous_13[3]

# Combine the results (example: using average of two triangulations)
points_3d_combined = (points_3d_12 + points_3d_13) / 2

print("Triangulated 3D Points (Combined):")
print(points_3d_combined.T)


#print('triangulated points', len(points_homogeneous.T))
#print((points_homogeneous[:3] / points_homogeneous[3]).T)

import cv2
import numpy as np

def main():
    # Intrinsic matrix K
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Define 3D points in the world
    points_3d = np.array([
        [1, 1, 5],
        [2, 0, 5],
        [0, -1, 5],
        [-1, -1, 5],
        [-1, -2, 10],
        [-2, -3, 9],
        [-2, -5, 1],
        [3, -1, 9],
    ], dtype=np.float32)

    # Define two camera poses
    R1 = np.eye(3)  # First camera: Identity rotation
    t1 = np.zeros((3, 1))  # First camera: No translation

    R2, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))  # Slight rotation for the second camera
    t2 = np.array([[1.0], [0.5], [0.2]])  # Slight translation for the second camera

    # Project the 3D points into two views

    
    #testing bundle adjustment projection
    from src.bundle_adjustment import project as project_points

    points_view1, P1 = project_points(points_3d, {'R' : R1, 't' : t1}, K)
    points_view2, P2 = project_points(points_3d, {'R' : R2, 't' : t2}, K)
    #points_homogeneous = cv2.triangulatePoints(P1, P2, points_view1.T, points_view2.T)
    #points_3d = points_homogeneous[:3] / points_homogeneous[3]
    #print('points_3d', points_3d.T)
    print('points_view1', points_view1)
    from src.initialization import triangulate_points
    points_3d = triangulate_points(P1, P2, points_view1,  points_view2)# already handling traspose
    print('points_3d', points_3d)
    # Estimate the fundamental matrix
    ransac_threshold = 1.0
    F, inliers = cv2.findFundamentalMat(points_view1, points_view2, cv2.FM_RANSAC, ransac_threshold)

    if F is None or inliers is None:
        print("Fundamental matrix estimation failed.")
        return

    # Compute Essential Matrix
    E = K.T @ F @ K

    # Recover pose from the Essential Matrix
    retval, R_recovered, t_recovered, mask = cv2.recoverPose(E, points_view1, points_view2, K)
    R_inv = R_recovered.T
    t_inv = -R_inv @ t_recovered
    # Print results
    print("Original Rotation R2:\n", R2)
    print("Recovered Rotation R:\n", R_inv)
    print("Original Translation t2:\n", t2.ravel())
    print("Recovered Translation t:\n", t_inv.ravel())
    print("Number of inliers:", inliers)

if __name__ == "__main__":
    main()



import numpy as np
import cv2

# Example of hardcoded keyframes
keyframe0 = {
    "points": np.array([
        [100, 200],  # Point 0
        [150, 250],  # Point 1
        [200, 300],  # Point 2
        [250, 350],  # Point 3
    ]),
    "descriptors": np.array([
        [0.1, 0.58, 0.3, 0.4],  # Descriptor 0
        [0.2, 0.3, 0.4, 0.5],  # Descriptor 1
        [0.3, 0.4, 0.5, 0.6],  # Descriptor 2
        [0.4, 0.5, 0.6, 0.7],  # Descriptor 3
    ], dtype=np.float32),
}

keyframe1 = {
    "points": np.array([
        [150, 250],  # Point 1 (matches keyframe0 Point 1)
        [200, 300],  # Point 2 (matches keyframe0 Point 2)
        [300, 400],  # Point 4 (new point)
        [400, 500],  # Point 5 (new point)
    ]),
    "descriptors": np.array([
        [0.1, 0.2, 0.3, 0.4],  # Descriptor 0
        [0.2, 0.3, 0.4, 0.5],  # Descriptor 1
        [0.4, 0.5, 0.6, 0.7],  # Descriptor 2
        [0.4, 0.5, 0.6, 0.7],  # Descriptor 3
    ], dtype=np.float32),
}

keyframe2 = {
    "points": np.array([
        [200, 300],  # Point 2 (matches keyframe1 Point 2)
        [300, 400],  # Point 4 (matches keyframe1 Point 4)
        [500, 600],  # Point 6 (new point)
        [600, 700],  # Point 7 (new point)
    ]),
    "descriptors": np.array([
        [0.1, 0.2, 0.3, 0.4],  # Descriptor 0
        [0.2, 0.3, 0.4, 0.5],  # Descriptor 1
        [0.3, 0.4, 0.5, 0.6],  # Descriptor 2
        [0.4, 0.5, 0.6, 0.7],  # Descriptor 3
    ], dtype=np.float32),
}

def test_align_keyframes():
    from src.initialization import align_keyframes, keyframe_matcher  # Replace `your_module` with actual module name

    try:
        matched_keyframe0_a, matched_keyframe1, matched_keyframe2 = align_keyframes(keyframe0, keyframe1, keyframe2)

        print("Aligned Keyframe 0 (with Keyframe 1):")
        print("Points:", matched_keyframe0_a["points"])
        print("Descriptors:", matched_keyframe0_a["descriptors"])

        print("\nAligned Keyframe 1:")
        print("Points:", matched_keyframe1["points"])
        print("Descriptors:", matched_keyframe1["descriptors"])

        print("\nAligned Keyframe 2:")
        print("Points:", matched_keyframe2["points"])
        print("Descriptors:", matched_keyframe2["descriptors"])

    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    test_align_keyframes()



