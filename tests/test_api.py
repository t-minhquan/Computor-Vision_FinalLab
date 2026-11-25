from hashlib import sha256
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.emotion_recognition.svm_fer_classifier import EMOTIONS
from src.preprocessing.face_crop import MAX_UPLOAD_BYTES


def test_frontend_and_assets_are_served(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert '<html lang="vi">' in response.text
        assert 'id="upload-form"' in response.text
        assert response.headers["cache-control"] == "no-store"
        static_dir = Path(__file__).resolve().parents[1] / "src/api/static"
        for asset in ("app.js", "style.css"):
            version = sha256((static_dir / asset).read_bytes()).hexdigest()[:12]
            assert f'/static/{asset}?v={version}' in response.text
            assert client.get(f"/static/{asset}?v={version}").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        script = client.get("/static/app.js")
        assert script.status_code == 200
        assert "/api/v1/predict" in script.text


def test_health_and_missing_model(tmp_path, image_bytes):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").json() == {"status": "ok", "model_ready": False}
        assert client.get("/api/v1/emotions").json() == {"emotions": list(EMOTIONS)}
        response = client.post("/api/v1/predict", files={"file": ("face.png", image_bytes)})
        assert response.status_code == 503
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_predict_and_upload_errors(trained_model, image_bytes, monkeypatch):
    app = create_app(trained_model)
    with TestClient(app) as client:
        monkeypatch.setattr(
            app.state.pipeline.detector,
            "crop",
            lambda image: (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0, 128, 128)),
        )
        assert client.get("/health").json()["model_ready"]
        response = client.post("/api/v1/predict", files={"file": ("face.png", image_bytes)})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["emotion"] in EMOTIONS
        assert result["face_box"] == {"x": 0, "y": 0, "width": 128, "height": 128}
        assert result["latency_ms"] >= 0
        assert client.post("/api/v1/predict").status_code == 422
        for content in (b"", b"bad image"):
            assert (
                client.post("/api/v1/predict", files={"file": ("bad.png", content)}).status_code
                == 422
            )
        assert (
            client.post(
                "/api/v1/predict", files={"file": ("large.png", b"x" * (MAX_UPLOAD_BYTES + 1))}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/v1/predict",
                files={"file": ("large.png", b"x" * (MAX_UPLOAD_BYTES + 100_000))},
            ).status_code
            == 413
        )


def test_streamed_upload_limit_without_content_length(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        response = client.post(
            "/api/v1/predict",
            content=iter([b"x" * 1024] * 6000),
            headers={"content-type": "multipart/form-data; boundary=test"},
        )
        assert response.status_code == 413


def test_no_face_error_is_returned(trained_model, image_bytes, monkeypatch):
    app = create_app(trained_model)

    def fail(image):
        raise ValueError("Expected exactly one face; found 0.")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.pipeline.detector, "crop", fail)
        response = client.post("/api/v1/predict", files={"file": ("face.png", image_bytes)})
        assert response.status_code == 422
        assert "found 0" in response.json()["detail"]


def test_invalid_model_fails_startup(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"invalid model")
    with pytest.raises(Exception):
        with TestClient(create_app(path)):
            pass


def test_cropped_image_mode_bypasses_detection_but_validates_image(
    trained_model, image_bytes, monkeypatch
):
    app = create_app(trained_model)

    def no_detector(image):
        raise ValueError("Expected exactly one face; found 0.")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.pipeline.detector, "crop", no_detector)
        upload = {"file": ("face.png", image_bytes, "image/png")}
        assert client.post("/api/v1/predict", files=upload).status_code == 422
        response = client.post("/api/v1/predict?cropped_faces=true", files=upload)
        assert response.status_code == 200, response.text
        assert response.json()["face_box"] == {"x": 0, "y": 0, "width": 128, "height": 128}
        assert client.post("/api/v1/predict?cropped_faces=invalid", files=upload).status_code == 422
        assert (
            client.post(
                "/api/v1/predict?cropped_faces=true",
                files={"file": ("bad.png", b"bad image", "image/png")},
            ).status_code
            == 422
        )
