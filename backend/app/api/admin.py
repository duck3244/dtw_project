from fastapi import APIRouter, Depends

from app.schemas.responses import SnapshotResponse
from app.services.recognizer_service import RecognizerService, get_recognizer_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/snapshot", response_model=SnapshotResponse)
def create_snapshot(svc: RecognizerService = Depends(get_recognizer_service)) -> SnapshotResponse:
    path = svc.snapshot()
    return SnapshotResponse(path=str(path))
