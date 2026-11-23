"""
Integration testing with the CLI
"""

from pathlib import Path
from click.testing import CliRunner
from PIL import Image
import pytest

from cli.cli import cli
from mylib.image_classifier import DEFAULT_CLASSES


@pytest.fixture(name="sample_image_path")
def fixture_sample_image_path(tmp_path: Path) -> Path:
    """Create a temporary image file for CLI integration tests."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (64, 48), color=(100, 140, 180))
    img.save(img_path, format="JPEG")
    return img_path


def test_help():
    """Tests the command-line interface help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


def test_predict_cli(sample_image_path: Path):
    """Tests the CLI predict command with a deterministic seed."""
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", str(sample_image_path), "--seed", "7"])
    assert result.exit_code == 0
    assert result.output.strip() in DEFAULT_CLASSES


def test_resize_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI resize command saves the resized image."""
    out_path = tmp_path / "resized.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["resize", str(sample_image_path), "32", "24", "--output", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.is_file()
    with Image.open(out_path) as saved:
        assert saved.size == (32, 24)


def test_resize_invalid_size_cli(sample_image_path: Path, tmp_path: Path):
    """Tests that invalid resize dimensions print an error message."""
    out_path = tmp_path / "bad_resize.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["resize", str(sample_image_path), "0", "24", "--output", str(out_path)],
    )
    assert result.exit_code == 0
    assert "Error:" in result.output


def test_normalize_orientation_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI normalize-orientation command."""
    out_path = tmp_path / "oriented.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["normalize-orientation", str(sample_image_path), "-o", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.is_file()


def test_adjust_contrast_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI adjust-contrast command."""
    out_path = tmp_path / "contrast.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adjust-contrast",
            str(sample_image_path),
            "--factor",
            "1.3",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.is_file()


def test_adjust_brightness_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI adjust-brightness command."""
    out_path = tmp_path / "brightness.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "adjust-brightness",
            str(sample_image_path),
            "--factor",
            "1.15",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.is_file()


def test_equalize_histogram_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI equalize-histogram command."""
    out_path = tmp_path / "equalized.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["equalize-histogram", str(sample_image_path), "-o", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.is_file()


def test_apply_gaussian_blur_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI apply-gaussian-blur command."""
    out_path = tmp_path / "blurred.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "apply-gaussian-blur",
            str(sample_image_path),
            "--radius",
            "2.0",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0
    assert out_path.is_file()


def test_to_grayscale_rgb_cli(sample_image_path: Path, tmp_path: Path):
    """Tests the CLI to-grayscale-rgb command."""
    out_path = tmp_path / "gray.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["to-grayscale-rgb", str(sample_image_path), "-o", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.is_file()
