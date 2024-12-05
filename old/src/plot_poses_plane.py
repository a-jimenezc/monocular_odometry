import numpy as np
import matplotlib.pyplot as plt

def plot_poses(poses, plane='xz'):
    """
    Plot the poses in a specified 2D plane.

    Args:
    - poses: List of pose dictionaries, each containing {"R": rotation matrix, "t": translation vector}.
    - plane: The plane to plot ('xz', 'yz', or 'xy').

    Returns:
    - A 2D plot of the poses.
    """
    if not poses:
        print("No poses to plot.")
        return

    # Extract translations
    translations = np.array([pose["t"].flatten() for pose in poses])  # Shape: (N, 3)

    # Select plane to plot
    if plane == 'xz':
        x, y = translations[:, 0], translations[:, 2]
        xlabel, ylabel = "X", "Z"
    elif plane == 'yz':
        x, y = translations[:, 1], translations[:, 2]
        xlabel, ylabel = "Y", "Z"
    elif plane == 'xy':
        x, y = translations[:, 0], translations[:, 1]
        xlabel, ylabel = "X", "Y"
    else:
        raise ValueError("Invalid plane. Choose from 'xz', 'yz', or 'xy'.")

    # Plot the poses
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', label="Camera poses")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Camera Poses in the {xlabel}-{ylabel} Plane")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

