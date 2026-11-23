"""
Main CLI or app entry point for image classification and preprocessing.
"""

from pathlib import Path
import click

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


def _default_output_path(input_path: str, suffix: str) -> Path:
    """Build a default output file path alongside the input image."""
    src = Path(input_path)
    return src.with_name(f"{src.stem}_{suffix}{src.suffix or '.jpg'}")


@click.group()
def cli():
    """Main CLI to run image prediction, resizing, and preprocessing."""


@cli.command("predict")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--classes",
    default=",".join(DEFAULT_CLASSES),
    show_default=True,
    help="Comma-separated list of candidate class names.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Optional integer seed for reproducible random selection.",
)
def predict_cli(image_path, classes, seed):
    """Predict a random class for an image.

    Example:
        uv run python -m cli.cli predict sample.jpg
    """
    class_list = tuple(c.strip() for c in classes.split(",") if c.strip())
    try:
        result = predict(image_path, classes=class_list, seed=seed)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(result, fg="green"))


@cli.command("resize")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output file path (defaults to <name>_resized<ext>).",
)
def resize_cli(image_path, width, height, output):
    """Resize an image to WIDTH and HEIGHT pixels.

    Example:
        uv run python -m cli.cli resize sample.jpg 224 224
    """
    try:
        resized_img = resize(image_path, width, height)
        dest = save_image(
            resized_img, output or _default_output_path(image_path, "resized")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved resized image to {dest}", fg="green"))


@cli.command("normalize-orientation")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def normalize_orientation_cli(image_path, output):
    """Normalize image orientation using EXIF tags.

    Example:
        uv run python -m cli.cli normalize-orientation sample.jpg
    """
    try:
        processed = normalize_orientation(image_path)
        dest = save_image(
            processed, output or _default_output_path(image_path, "oriented")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved oriented image to {dest}", fg="green"))


@cli.command("adjust-contrast")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--factor", type=float, default=1.2, show_default=True)
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def adjust_contrast_cli(image_path, factor, output):
    """Adjust image contrast by a positive factor.

    Example:
        uv run python -m cli.cli adjust-contrast sample.jpg --factor 1.4
    """
    try:
        processed = adjust_contrast(image_path, factor=factor)
        dest = save_image(
            processed, output or _default_output_path(image_path, "contrast")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved contrast-adjusted image to {dest}", fg="green"))


@cli.command("adjust-brightness")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--factor", type=float, default=1.1, show_default=True)
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def adjust_brightness_cli(image_path, factor, output):
    """Adjust image brightness by a positive factor.

    Example:
        uv run python -m cli.cli adjust-brightness sample.jpg --factor 1.2
    """
    try:
        processed = adjust_brightness(image_path, factor=factor)
        dest = save_image(
            processed, output or _default_output_path(image_path, "brightness")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved brightness-adjusted image to {dest}", fg="green"))


@cli.command("equalize-histogram")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def equalize_histogram_cli(image_path, output):
    """Equalize image histogram across RGB channels.

    Example:
        uv run python -m cli.cli equalize-histogram sample.jpg
    """
    try:
        processed = equalize_histogram(image_path)
        dest = save_image(
            processed, output or _default_output_path(image_path, "equalized")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved equalized image to {dest}", fg="green"))


@cli.command("apply-gaussian-blur")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--radius", type=float, default=1.5, show_default=True)
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def apply_gaussian_blur_cli(image_path, radius, output):
    """Apply Gaussian blur with a non-negative radius.

    Example:
        uv run python -m cli.cli apply-gaussian-blur sample.jpg --radius 2.0
    """
    try:
        processed = apply_gaussian_blur(image_path, radius=radius)
        dest = save_image(
            processed, output or _default_output_path(image_path, "blurred")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved blurred image to {dest}", fg="green"))


@cli.command("to-grayscale-rgb")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None)
def to_grayscale_rgb_cli(image_path, output):
    """Convert image to grayscale and back to 3-channel RGB.

    Example:
        uv run python -m cli.cli to-grayscale-rgb sample.jpg
    """
    try:
        processed = to_grayscale_rgb(image_path)
        dest = save_image(
            processed, output or _default_output_path(image_path, "grayscale")
        )
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        return
    click.echo(click.style(f"Saved grayscale RGB image to {dest}", fg="green"))


# Main entry point
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
