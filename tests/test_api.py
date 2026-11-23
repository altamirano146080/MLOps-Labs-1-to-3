"""
Integration testing with the API
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from api.api import app
from mylib.image_classifier import DEFAULT_CLASSES


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes():
    """Create an in-memory JPEG byte stream for multipart upload tests."""
    img = Image.new("RGB", (64, 48), color=(120, 150, 180))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def test_home_endpoint(client):
    """Verify that the endpoint / returns the home page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Image Classifier" in response.text


def test_health_endpoint(client):
    """Verify that the health check answers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_default_classes(client, sample_jpeg_bytes):
    """Verify that /predict returns a valid default class for an uploaded image."""
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["class"] in DEFAULT_CLASSES


def test_predict_endpoint_custom_classes(client, sample_jpeg_bytes):
    """Verify that /predict respects custom comma-separated class names."""
    custom_classes = "car,bike,bus"
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"classes": custom_classes},
    )
    assert response.status_code == 200
    assert response.json()["class"] in {"car", "bike", "bus"}


def test_predict_endpoint_empty_classes(client, sample_jpeg_bytes):
    """Verify that empty classes are rejected with HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"classes": " , "},
    )
    assert response.status_code == 400
    assert "classes" in response.json()["detail"]


def test_predict_endpoint_invalid_image(client):
    """Verify that non-image file content is rejected with HTTP 400."""
    response = client.post(
        "/predict",
        files={
            "file": ("not_an_image.jpg", io.BytesIO(b"not a real image"), "image/jpeg")
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or corrupted image file"


def test_predict_endpoint_missing_file(client):
    """Verify that missing file upload returns HTTP 422."""
    response = client.post("/predict")
    assert response.status_code == 422
    assert "detail" in response.json()


def test_resize_endpoint_success(client, sample_jpeg_bytes):
    """Verify that /resize returns a JPEG image with requested dimensions."""
    response = client.post(
        "/resize",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"width": "32", "height": "24"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as output_img:
        assert output_img.size == (32, 24)


@pytest.mark.parametrize("width, height", [("0", "24"), ("32", "-5")])
def test_resize_endpoint_invalid_dimensions(client, sample_jpeg_bytes, width, height):
    """Verify that non-positive width or height returns HTTP 400."""
    response = client.post(
        "/resize",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"width": width, "height": height},
    )
    assert response.status_code == 400


def test_resize_endpoint_non_integer_dimensions(client, sample_jpeg_bytes):
    """Verify that non-integer form values return HTTP 422."""
    response = client.post(
        "/resize",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"width": "abc", "height": "24"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_resize_endpoint_missing_parameters(client, sample_jpeg_bytes):
    """Verify that missing form parameters return HTTP 422."""
    response = client.post(
        "/resize",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"width": "32"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "method, extra_data",
    [
        ("normalize-orientation", {}),
        ("ADJUST-CONTRAST", {"factor": "1.3"}),
        ("adjust-brightness", {"factor": "1.1"}),
        ("equalize-histogram", {}),
        ("apply-gaussian-blur", {"radius": "1.5"}),
        ("to-grayscale-rgb", {}),
    ],
)
def test_preprocess_endpoint_methods(client, sample_jpeg_bytes, method, extra_data):
    """Verify that /preprocess supports all configured preprocessing methods."""
    payload = {"method": method, **extra_data}
    response = client.post(
        "/preprocess",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data=payload,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as output_img:
        assert output_img.size == (64, 48)


def test_preprocess_endpoint_invalid_method(client, sample_jpeg_bytes):
    """Verify that an unknown preprocessing method returns HTTP 400."""
    response = client.post(
        "/preprocess",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"method": "unknown-filter"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid preprocessing method"


def test_preprocess_endpoint_invalid_factor(client, sample_jpeg_bytes):
    """Verify that a non-positive contrast factor returns HTTP 400."""
    response = client.post(
        "/preprocess",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"method": "adjust-contrast", "factor": "0"},
    )
    assert response.status_code == 400


def test_predict_endpoint_empty_file(client):
    """Verify that an empty uploaded file is rejected with HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_preprocess_endpoint_invalid_radius(client, sample_jpeg_bytes):
    """Verify that a negative blur radius returns HTTP 400."""
    response = client.post(
        "/preprocess",
        files={"file": ("sample.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        data={"method": "apply-gaussian-blur", "radius": "-1.0"},
    )
    assert response.status_code == 400
