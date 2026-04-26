from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.core.config import settings
from app.schemas.responses import RecognitionScore, RecognizeResponse
from app.services.recognizer_service import RecognizerService, get_recognizer_service

router = APIRouter(tags=["recognize"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
    svc: RecognizerService = Depends(get_recognizer_service),
) -> RecognizeResponse:
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in settings.allowed_audio_suffixes:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported suffix {suffix}")
    payload = await file.read()
    try:
        label, distance, top = svc.recognize(payload, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"recognition failed: {exc}")
    return RecognizeResponse(
        label=label,
        distance=distance,
        top_k=[RecognitionScore(label=l, distance=d) for l, d in top],
    )
