import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(points_3d, title="3D Point Cloud"):
    """
    Plots a 3D point cloud using Matplotlib.

    Args:
    - points_3d: NumPy array of shape (N, 3) representing the 3D points.
    - title: Title for the plot.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    # Scatter plot
    ax.scatter(x, y, z, c=z, cmap='viridis', s=5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set equal scaling
    max_range = (x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    max_extent = max(max_range)
    for axis, center in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (z.min() + z.max()) / 2]):
        axis([center - max_extent / 2, center + max_extent / 2])
    
    plt.show()
    