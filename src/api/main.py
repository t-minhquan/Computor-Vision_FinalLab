import os
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.emotion_recognition.svm_fer_classifier import DEFAULT_MODEL_PATH, EMOTIONS
from src.pipeline import EmotionPipeline
from src.preprocessing.face_crop import MAX_UPLOAD_BYTES


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Prediction(BaseModel):
    emotion: str
    face_box: FaceBox
    latency_ms: float


def create_app(model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured = os.environ.get("FER_MODEL_PATH")
        path = model_path
        if path is None:
            path = Path(configured) if configured else DEFAULT_MODEL_PATH
        if configured and model_path is None and not path.is_file():
            raise RuntimeError("FER_MODEL_PATH does not point to a model file.")
        app.state.pipeline = EmotionPipeline(path)
        yield

    app = FastAPI(
        title="Facial Emotion Recognition API",
        version="0.1.0",
        lifespan=lifespan,
        # Include a small allowance for multipart headers; image bytes are checked separately.
        max_body_size=MAX_UPLOAD_BYTES + 64 * 1024,
        form_limits={"max_files": 1, "max_fields": 0},
    )
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        for asset in ("app.js", "style.css"):
            version = sha256((static_dir / asset).read_bytes()).hexdigest()[:12]
            html = html.replace(f'"/static/{asset}"', f'"/static/{asset}?v={version}"')
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_ready": request.app.state.pipeline.ready}

    @app.get("/api/v1/emotions")
    def emotions():
        return {"emotions": list(EMOTIONS)}

    @app.post(
        "/api/v1/predict",
        response_model=Prediction,
        responses={
            413: {"description": "Upload too large"},
            422: {"description": "Invalid image or face count"},
            503: {"description": "Model not trained"},
        },
    )
    def predict(request: Request, file: UploadFile, cropped_faces: bool = False):
        pipeline = request.app.state.pipeline
        if not pipeline.ready:
            raise HTTPException(503, "No trained model. Train a model and restart the backend.")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 5 MiB limit.")
        try:
            return pipeline.predict(content, cropped_faces=cropped_faces)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
