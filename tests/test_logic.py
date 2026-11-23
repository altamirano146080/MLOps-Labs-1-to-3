"""
Unit testing of the application's logic
"""

from pathlib import Path
from PIL import Image
import pytest

from mylib.image_classifier import (
    DEFAULT_CLASSES,
    adjust_brightness,
    adjust_contrast,
    apply_gaussian_blur,
    equalize_histogram,
    normalize_orientation,
    predict,
    resize,
    save_image,
    to_grayscale_rgb,
)


@pytest.fixture(name="sample_image_path")
def fixture_sample_image_path(tmp_path: Path) -> Path:
    """Create a temporary RGB image file for testing."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (64, 48), color=(120, 150, 180))
    img.save(img_path, format="JPEG")
    return img_path


def test_predict_returns_valid_class(sample_image_path: Path):
    """Tests that predict returns one of the configured class names."""
    result = predict(sample_image_path, classes=DEFAULT_CLASSES, seed=42)
    assert result in DEFAULT_CLASSES


def test_predict_empty_classes_raises(sample_image_path: Path):
    """Tests that predict raises ValueError when classes is empty."""
    with pytest.raises(ValueError):
        predict(sample_image_path, classes=())


def test_predict_missing_file_raises(tmp_path: Path):
    """Tests that predict raises FileNotFoundError for a missing path."""
    missing = tmp_path / "nonexistent.jpg"
    with pytest.raises(FileNotFoundError):
        predict(missing)


def test_resize_dimensions(sample_image_path: Path):
    """Tests resizing an image to target width and height."""
    resized = resize(sample_image_path, 32, 24)
    assert resized.size == (32, 24)
    assert resized.mode == "RGB"


def test_resize_invalid_dimensions_raises(sample_image_path: Path):
    """Tests that non-positive dimensions raise ValueError."""
    with pytest.raises(ValueError):
        resize(sample_image_path, 0, 24)
    with pytest.raises(ValueError):
        resize(sample_image_path, 32, -5)


def test_normalize_orientation(sample_image_path: Path):
    """Tests orientation normalization returns an RGB image."""
    result = normalize_orientation(sample_image_path)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (64, 48)


def test_adjust_contrast_and_invalid_factor(sample_image_path: Path):
    """Tests contrast adjustment and rejects non-positive factors."""
    result = adjust_contrast(sample_image_path, factor=1.5)
    assert result.mode == "RGB"
    assert result.size == (64, 48)
    with pytest.raises(ValueError):
        adjust_contrast(sample_image_path, factor=0.0)


def test_adjust_brightness_and_invalid_factor(sample_image_path: Path):
    """Tests brightness adjustment and rejects non-positive factors."""
    result = adjust_brightness(sample_image_path, factor=1.2)
    assert result.mode == "RGB"
    assert result.size == (64, 48)
    with pytest.raises(ValueError):
        adjust_brightness(sample_image_path, factor=-1.0)


def test_equalize_histogram(sample_image_path: Path):
    """Tests histogram equalization returns an RGB image of same size."""
    result = equalize_histogram(sample_image_path)
    assert result.mode == "RGB"
    assert result.size == (64, 48)


def test_apply_gaussian_blur_and_invalid_radius(sample_image_path: Path):
    """Tests Gaussian blur and rejects negative radius."""
    result = apply_gaussian_blur(sample_image_path, radius=1.5)
    assert result.mode == "RGB"
    assert result.size == (64, 48)
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_image_path, radius=-0.5)


def test_to_grayscale_rgb(sample_image_path: Path):
    """Tests grayscale conversion returns a 3-channel RGB image."""
    result = to_grayscale_rgb(sample_image_path)
    assert result.mode == "RGB"
    assert result.size == (64, 48)


def test_save_image(sample_image_path: Path, tmp_path: Path):
    """Tests saving a PIL image to a new nested directory."""
    resized = resize(sample_image_path, 16, 16)
    out_file = tmp_path / "nested" / "out.png"
    saved_path = save_image(resized, out_file)
    assert saved_path.is_file()


def test_load_invalid_type_raises():
    """Tests that passing an unsupported object raises TypeError."""
    with pytest.raises(TypeError):
        predict(12345)  # type: ignore[arg-type]


def test_normalize_orientation_missing_file_raises(tmp_path: Path):
    """Tests that normalize_orientation raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        normalize_orientation(tmp_path / "missing.jpg")
