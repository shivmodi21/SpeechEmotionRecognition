import os
import tempfile
from pathlib import Path
import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from .predictor import predict_emotion


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
PROCESSING_TIMEOUT = 60  # 60 seconds
COPY_CHUNK_SIZE = 1024 * 1024  # 1 MB


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Speech Emotion Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, FLAC, OGG, M4A, or AAC.",
        )

    temp_path = None
    total_size = 0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = temp.name

            while chunk := await file.read(COPY_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file must be smaller than {MAX_FILE_SIZE // (1024 * 1024)} MB.",
                    )
                temp.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

        result = await asyncio.wait_for(
            run_in_threadpool(predict_emotion, temp_path),
            timeout=PROCESSING_TIMEOUT,
        )
        return {"filename": file.filename, **result}

    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Audio processing timed out. Please try a shorter audio file.",
        ) from exc
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process this audio file: {exc}") from exc
    finally:
        await file.close()
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
