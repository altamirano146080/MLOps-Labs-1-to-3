"""
Integration testing with the CLI
"""

from pathlib import Path
import runpy
import sys
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


def test_resize_default_output_cli(sample_image_path: Path):
    """Tests resizing without --output creates the default <stem>_resized.jpg file."""
    expected_out = sample_image_path.parent / "sample_resized.jpg"
    runner = CliRunner()
    try:
        result = runner.invoke(cli, ["resize", str(sample_image_path), "32", "24"])
        assert result.exit_code == 0
        assert expected_out.is_file()
    finally:
        if expected_out.exists():
            expected_out.unlink()


def test_predict_empty_classes_cli(sample_image_path: Path):
    """Tests that empty --classes prints an error message instead of crashing."""
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", str(sample_image_path), "--classes", " , "])
    assert result.exit_code == 0
    assert "Error:" in result.output


def test_preprocess_commands_invalid_params_cli(sample_image_path: Path):
    """Tests error handling in CLI preprocessing commands for invalid parameters."""
    runner = CliRunner()

    res_contrast = runner.invoke(
        cli, ["adjust-contrast", str(sample_image_path), "--factor", "0"]
    )
    assert res_contrast.exit_code == 0
    assert "Error:" in res_contrast.output

    res_brightness = runner.invoke(
        cli, ["adjust-brightness", str(sample_image_path), "--factor", "-1"]
    )
    assert res_brightness.exit_code == 0
    assert "Error:" in res_brightness.output

    res_blur = runner.invoke(
        cli, ["apply-gaussian-blur", str(sample_image_path), "--radius", "-2"]
    )
    assert res_blur.exit_code == 0
    assert "Error:" in res_blur.output


@pytest.mark.parametrize(
    "command",
    ["normalize-orientation", "equalize-histogram", "to-grayscale-rgb"],
)
def test_preprocess_commands_invalid_image_cli(tmp_path: Path, command: str):
    """Tests error handling in CLI preprocessing commands for invalid image files."""
    bad_file = tmp_path / "not_an_image.jpg"
    bad_file.write_bytes(b"not an image")
    runner = CliRunner()
    result = runner.invoke(cli, [command, str(bad_file)])
    assert result.exit_code == 0
    assert "Error:" in result.output


def test_cli_main_entrypoint(monkeypatch, capsys):
    """Verify that running cli.cli as __main__ invokes the Click CLI."""
    monkeypatch.setattr(sys, "argv", ["cli", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("cli.cli", run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Show this message and exit." in captured.out
