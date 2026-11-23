"""
REST API of the image classifier built with FastAPI.

Endpoints:
    GET  /            home page with a link to the interactive documentation
    GET  /health      health check (Docker HEALTHCHECK, CI smoke test, cloud platforms)
    POST /predict     predict a class for an uploaded image (form data: file, classes)
    POST /resize      resize an uploaded image to width and height (form data: file, width, height)
    POST /preprocess  preprocess an uploaded image (form data: file, method, factor, radius)
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from mylib.image_classifier import (
    DEFAULT_CLASSES,
    adjust_brightness,
    adjust_contrast,
    apply_gaussian_blur,
    equalize_histogram,
    normalize_orientation,
    predict,
    resize,
    to_grayscale_rgb,
)

PREPROCESS_OPERATIONS = {
    "normalize-orientation": normalize_orientation,
    "adjust-contrast": adjust_contrast,
    "adjust-brightness": adjust_brightness,
    "equalize-histogram": equalize_histogram,
    "apply-gaussian-blur": apply_gaussian_blur,
    "to-grayscale-rgb": to_grayscale_rgb,
}

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Image Classifier using FastAPI",
    description="API to predict, resize, and preprocess images using mylib.image_classifier",
    version="1.1.0",
)

# The templates folder is located relative to this file, so the API works
# whatever the working directory is (locally, in Docker, or in tests)
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)


async def _read_uploaded_image(file: UploadFile) -> Image.Image:
    """Read an uploaded binary file and open it as an RGB PIL image."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        with Image.open(io.BytesIO(contents)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid or corrupted image file"
        ) from exc


def _image_to_jpeg_response(image: Image.Image) -> StreamingResponse:
    """Encode a PIL image as JPEG in memory and return a streaming response."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page with a link to the interactive documentation (/docs)."""
    return templates.TemplateResponse(request, "home.html")


@app.get("/health")
def health():
    """Health check: it answers while the API is running."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    classes: str = Form(",".join(DEFAULT_CLASSES)),
):
    """
    Predict a class label for an uploaded image.

    - file: binary image file (multipart/form-data)
    - classes: comma-separated candidate class names
    """
    image = await _read_uploaded_image(file)
    class_tuple = tuple(c.strip() for c in classes.split(",") if c.strip())
    try:
        predicted_class = predict(image, classes=class_tuple)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"class": predicted_class}


@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
):
    """
    Resize an uploaded image to the requested width and height in pixels.

    - file: binary image file (multipart/form-data)
    - width, height: target positive integer dimensions
    """
    image = await _read_uploaded_image(file)
    try:
        resized_image = resize(image, width, height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _image_to_jpeg_response(resized_image)


@app.post("/preprocess")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    method: str = Form("normalize-orientation"),
    factor: float = Form(1.2),
    radius: float = Form(1.5),
):
    """
    Apply a preprocessing step to an uploaded image.

    - file: binary image file (multipart/form-data)
    - method: normalize-orientation, adjust-contrast, adjust-brightness,
      equalize-histogram, apply-gaussian-blur, or to-grayscale-rgb
    - factor: enhancement factor for contrast or brightness
    - radius: blur radius for Gaussian blur
    """
    normalized_method = method.strip().lower()
    operation = PREPROCESS_OPERATIONS.get(normalized_method)
    if operation is None:
        raise HTTPException(status_code=400, detail="Invalid preprocessing method")

    image = await _read_uploaded_image(file)
    try:
        if normalized_method in ("adjust-contrast", "adjust-brightness"):
            processed = operation(image, factor=factor)
        elif normalized_method == "apply-gaussian-blur":
            processed = operation(image, radius=radius)
        else:
            processed = operation(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _image_to_jpeg_response(processed)


# Entry point (for direct execution only: python -m api.api)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
