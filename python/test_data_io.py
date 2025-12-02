import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from data_io import load_as_array, load_as_o3d_cloud, save_data_as_xyz


@pytest.fixture
def sample_data_without_labels() -> np.ndarray:
    """Create sample point cloud data without labels (6 columns)."""
    return np.array([
        [1.0, 2.0, 3.0, 255, 128, 0],
        [4.0, 5.0, 6.0, 0, 255, 128],
        [7.0, 8.0, 9.0, 128, 0, 255],
    ])


@pytest.fixture
def sample_data_with_labels() -> np.ndarray:
    """Create sample point cloud data with labels (8 columns)."""
    return np.array([
        [1.0, 2.0, 3.0, 255, 128, 0, 1, 0],
        [4.0, 5.0, 6.0, 0, 255, 128, 2, 1],
        [7.0, 8.0, 9.0, 128, 0, 255, 1, 0],
    ])


@pytest.fixture
def temp_xyz_file_without_labels(
    sample_data_without_labels: np.ndarray,
) -> Path:
    """Create a temporary .xyz file without labels."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.xyz', delete=False
    ) as f:
        f.write("//X Y Z R G B\n")
        np.savetxt(
            f,
            sample_data_without_labels,
            fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d']
        )
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def temp_xyz_file_with_labels(sample_data_with_labels: np.ndarray) -> Path:
    """Create a temporary .xyz file with labels."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.xyz', delete=False
    ) as f:
        f.write("//X Y Z R G B class instance\n")
        np.savetxt(
            f,
            sample_data_with_labels,
            fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d', '%d', '%d']
        )
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


class TestLoadAsArray:
    """Tests for load_as_array function."""

    def test_load_without_labels(
        self,
        temp_xyz_file_without_labels: Path,
        sample_data_without_labels: np.ndarray,
    ) -> None:
        """Test loading data without labels."""
        data, labels_available = load_as_array(
            str(temp_xyz_file_without_labels))

        assert data.shape == (3, 6)
        assert not labels_available
        np.testing.assert_array_almost_equal(data, sample_data_without_labels)

    def test_load_with_labels(
        self,
        temp_xyz_file_with_labels: Path,
        sample_data_with_labels: np.ndarray,
    ) -> None:
        """Test loading data with labels."""
        data, labels_available = load_as_array(str(temp_xyz_file_with_labels))

        assert data.shape == (3, 8)
        assert labels_available
        np.testing.assert_array_almost_equal(data, sample_data_with_labels)

    def test_load_nonexistent_file(self) -> None:
        """Test loading a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            load_as_array("nonexistent_file.xyz")


class TestLoadAsO3dCloud:
    """Tests for load_as_o3d_cloud function."""

    def test_load_without_labels(
        self,
        temp_xyz_file_without_labels: Path,
        sample_data_without_labels: np.ndarray,
    ) -> None:
        """Test loading point cloud without labels."""
        pc, labels_available, labels = load_as_o3d_cloud(
            str(temp_xyz_file_without_labels)
        )

        assert isinstance(pc, o3d.geometry.PointCloud)
        assert not labels_available
        assert labels is None

        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)

        expected_points = sample_data_without_labels[:, 0:3]
        expected_colors = sample_data_without_labels[:, 3:6] / 255

        np.testing.assert_array_almost_equal(points, expected_points)
        np.testing.assert_array_almost_equal(colors, expected_colors)

    def test_load_with_labels(
        self,
        temp_xyz_file_with_labels: Path,
        sample_data_with_labels: np.ndarray,
    ) -> None:
        """Test loading point cloud with labels."""
        pc, labels_available, labels = load_as_o3d_cloud(
            str(temp_xyz_file_with_labels)
        )

        assert isinstance(pc, o3d.geometry.PointCloud)
        assert labels_available
        assert labels is not None

        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)

        expected_points = sample_data_with_labels[:, 0:3]
        expected_colors = sample_data_with_labels[:, 3:6] / 255
        expected_labels = sample_data_with_labels[:, 6:]

        np.testing.assert_array_almost_equal(points, expected_points)
        np.testing.assert_array_almost_equal(colors, expected_colors)
        np.testing.assert_array_almost_equal(labels, expected_labels)

    def test_color_normalization(
        self, temp_xyz_file_without_labels: Path
    ) -> None:
        """Test that colors are correctly normalized to [0, 1] range."""
        pc, _, _ = load_as_o3d_cloud(str(temp_xyz_file_without_labels))
        colors = np.asarray(pc.colors)

        assert np.all(colors >= 0)
        assert np.all(colors <= 1)


class TestSaveDataAsXyz:
    """Tests for save_data_as_xyz function."""

    def test_save_data(self, sample_data_without_labels: np.ndarray) -> None:
        """Test saving data to .xyz file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xyz', delete=False
        ) as f:
            temp_path = Path(f.name)

        try:
            save_data_as_xyz(sample_data_without_labels, str(temp_path))

            assert temp_path.exists()

            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert lines[0].strip() == "//X Y Z R G B class instance"

            loaded_data = np.loadtxt(temp_path, comments='//')
            np.testing.assert_array_almost_equal(
                loaded_data, sample_data_without_labels
            )
        finally:
            temp_path.unlink()

    def test_save_and_reload_roundtrip(
        self, sample_data_without_labels: np.ndarray
    ) -> None:
        """Test that saved data can be reloaded correctly."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xyz', delete=False
        ) as f:
            temp_path = Path(f.name)

        try:
            save_data_as_xyz(sample_data_without_labels, str(temp_path))
            reloaded_data, _ = load_as_array(str(temp_path))

            np.testing.assert_array_almost_equal(
                reloaded_data, sample_data_without_labels
            )
        finally:
            temp_path.unlink()


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(
        self, sample_data_without_labels: np.ndarray
    ) -> None:
        """Test complete save/load workflow."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xyz', delete=False
        ) as f:
            temp_path = Path(f.name)

        try:
            # Save data (only 6 columns supported)
            save_data_as_xyz(sample_data_without_labels, str(temp_path))

            # Load as array
            array_data, labels_available = load_as_array(str(temp_path))
            assert labels_available is False
            np.testing.assert_array_almost_equal(
                array_data, sample_data_without_labels
            )

            # Load as point cloud
            pc, labels_available, labels = load_as_o3d_cloud(str(temp_path))
            assert isinstance(pc, o3d.geometry.PointCloud)
            assert len(pc.points) == len(sample_data_without_labels)
            assert labels is None
        finally:
            temp_path.unlink()
