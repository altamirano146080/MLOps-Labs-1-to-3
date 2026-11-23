"""
Image classifier library: the business logic shared by the CLI and the API.
"""

from pathlib import Path
import random
from typing import Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

DEFAULT_CLASSES: tuple[str, ...] = ("cat", "dog", "bird")
ImageInput = Union[Image.Image, str, Path]


def _load_rgb_image(image: ImageInput) -> Image.Image:
    """Load an image from a path or return a copy converted to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")
    raise TypeError(
        f"Unsupported image type: {type(image).__name__}. "
        "Expected a PIL Image, str, or pathlib.Path."
    )


def save_image(image: Image.Image, output_path: Union[str, Path]) -> Path:
    """Save a PIL image to disk and return the resolved output path."""
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def predict(
    image: ImageInput,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    seed: int | None = None,
) -> str:
    """Return a randomly chosen class name for a given image."""
    _load_rgb_image(image)
    if not classes:
        raise ValueError("classes must not be empty.")
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(list(classes))


def resize(image: ImageInput, width: int, height: int) -> Image.Image:
    """Return a resized copy of the input image with dimensions (width, height)."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    rgb_image = _load_rgb_image(image)
    return rgb_image.resize((width, height), Image.Resampling.LANCZOS)


## Image preprocessing methods


def normalize_orientation(image: ImageInput) -> Image.Image:
    """Apply EXIF orientation metadata so the image is upright in RGB."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as raw_img:
            return ImageOps.exif_transpose(raw_img).convert("RGB")
    rgb_image = _load_rgb_image(image)
    return ImageOps.exif_transpose(rgb_image).convert("RGB")


def adjust_contrast(image: ImageInput, factor: float = 1.2) -> Image.Image:
    """Adjust image contrast by a positive factor and return RGB."""
    if factor <= 0:
        raise ValueError("factor must be greater than 0.")
    rgb_image = _load_rgb_image(image)
    return ImageEnhance.Contrast(rgb_image).enhance(factor)


def adjust_brightness(image: ImageInput, factor: float = 1.1) -> Image.Image:
    """Adjust image brightness by a positive factor and return RGB."""
    if factor <= 0:
        raise ValueError("factor must be greater than 0.")
    rgb_image = _load_rgb_image(image)
    return ImageEnhance.Brightness(rgb_image).enhance(factor)


def equalize_histogram(image: ImageInput) -> Image.Image:
    """Equalize RGB image histogram to balance lighting and contrast."""
    rgb_image = _load_rgb_image(image)
    return ImageOps.equalize(rgb_image)


def apply_gaussian_blur(image: ImageInput, radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur with a non-negative radius and return RGB."""
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")
    rgb_image = _load_rgb_image(image)
    return rgb_image.filter(ImageFilter.GaussianBlur(radius=radius))


def to_grayscale_rgb(image: ImageInput) -> Image.Image:
    """Convert image to grayscale and back to 3-channel RGB."""
    rgb_image = _load_rgb_image(image)
    return ImageOps.grayscale(rgb_image).convert("RGB")
