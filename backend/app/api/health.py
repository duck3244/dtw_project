from fastapi import APIRouter

from app.core.config import settings
from app.schemas.responses import HealthResponse
from app.services.recognizer_service import accel_available, get_recognizer_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_recognizer_service()
    return HealthResponse(
        status="ok",
        backend=svc.backend_name,
        sample_rate=settings.sample_rate,
        accel_available=accel_available(),
    )
