import os
from typing import Optional

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


def load_as_array(file_path: str) -> tuple[NDArray[np.float64], bool]:
    """Load data from an .xyz file into a numpy array.

    Args:
        file_path: Path to the .xyz file.

    Returns:
        A tuple containing:
            - data_array: Raw data as np array, of shape (n, 6) or (n, 8)
              if labels are available.
            - labels_available: Boolean indicating whether per-point labels
              are available.
    """
    data_array = np.loadtxt(file_path, comments='//')
    labels_available = data_array.shape[1] == 8
    return data_array, labels_available


def load_as_o3d_cloud(
    file_path: str
) -> tuple[o3d.geometry.PointCloud, bool, Optional[NDArray[np.float64]]]:
    """Load data from an .xyz file into an Open3D point cloud object.

    Args:
        file_path: Path to the .xyz file.

    Returns:
        A tuple containing:
            - pc: Open3D PointCloud object.
            - labels_available: Boolean indicating whether labels are available.
            - labels: Label data array if available, None otherwise.
    """
    data, labels_available = load_as_array(file_path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
    labels: Optional[NDArray[np.float64]] = None
    if labels_available:
        labels = data[:, 6:]
    return pc, labels_available, labels


def save_data_as_xyz(data: NDArray[np.float64], path: str) -> None:
    """Save data in .xyz format.

    Args:
        data: Numpy array containing point cloud data.
        path: Output file path.
    """
    with open(path, 'w') as f:
        f.write("//X Y Z R G B class instance\n")
        np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])


if __name__ == '__main__':
    """Example usage of the functions defined in this file."""
    # Please specify where the data set is located:
    data_dir = '/home/.../'

    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]

    # Load the first scan in the data set for example:
    pointcloud, labels_available, labels = load_as_o3d_cloud(
        data_dir + scans[0])
    # Visualize the point cloud:
    o3d.visualization.draw_geometries([pointcloud])
