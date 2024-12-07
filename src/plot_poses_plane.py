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
    translations = np.array([pose.t.flatten() for pose in poses])  # Shape: (N, 3)

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

def plot_camera_poses(poses, ax=None, scale=0.1):
    """
    Plot a list of camera poses in 3D space.
    
    Args:
        poses (list of CamPose): List of camera poses to plot.
        ax (mpl_toolkits.mplot3d.Axes3D, optional): Existing 3D axis to plot on. If None, creates a new figure.
        scale (float): Scale of the orientation axes drawn for each pose.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot each camera pose
    for pose in poses:
        # Translation vector
        t = pose.t
        ax.scatter(t[0], t[1], t[2], c='r', label='Camera' if 'Camera' not in ax.get_legend_handles_labels()[1] else "")
        
        # Plot orientation axes (scaled by `scale`)
        R = pose.R
        for i, color in enumerate(['r', 'g', 'b']):  # x=red, y=green, z=blue
            axis = R[:, i] * scale
            ax.quiver(
                t[0], t[1], t[2], axis[0], axis[1], axis[2],
                color=color, label=f'{color}-axis' if f'{color}-axis' not in ax.get_legend_handles_labels()[1] else ""
            )

    # Set labels and aspect
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Camera Poses")
    ax.legend()
    ax.grid(True)
    plt.show()
