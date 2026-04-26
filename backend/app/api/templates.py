from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.responses import (
    AddTemplateResponse,
    LabelDetail,
    TemplateInfo,
    TemplatesResponse,
)
from app.services.recognizer_service import (
    RecognizerService,
    TemplateNotFound,
    get_recognizer_service,
)

router = APIRouter(prefix="/templates", tags=["templates"])


def _validate_audio(file: UploadFile) -> None:
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in settings.allowed_audio_suffixes:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported suffix {suffix}")


@router.get("", response_model=TemplatesResponse)
def list_templates(svc: RecognizerService = Depends(get_recognizer_service)) -> TemplatesResponse:
    counts = svc.list_templates()
    items = [TemplateInfo(label=l, count=c) for l, c in sorted(counts.items())]
    return TemplatesResponse(labels=items, total=sum(counts.values()))


@router.get("/{label}", response_model=LabelDetail)
def label_detail(label: str, svc: RecognizerService = Depends(get_recognizer_service)) -> LabelDetail:
    ids = svc.list_label_ids(label)
    if not ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"label {label!r} not found")
    return LabelDetail(label=label, template_ids=ids)


@router.post("", response_model=AddTemplateResponse, status_code=status.HTTP_201_CREATED)
async def add_template(
    label: str = Form(..., min_length=1, max_length=64),
    file: UploadFile = File(...),
    svc: RecognizerService = Depends(get_recognizer_service),
) -> AddTemplateResponse:
    _validate_audio(file)
    payload = await file.read()
    try:
        tid, count = svc.add_template(label, payload)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"failed to register template: {exc}")
    return AddTemplateResponse(label=label, template_id=tid, count=count)


@router.delete("/{label}", status_code=status.HTTP_204_NO_CONTENT)
def delete_label(label: str, svc: RecognizerService = Depends(get_recognizer_service)) -> None:
    n = svc.remove_label(label)
    if n == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"label {label!r} not found")


@router.delete("/{label}/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(
    label: str,
    template_id: str,
    svc: RecognizerService = Depends(get_recognizer_service),
) -> None:
    try:
        svc.remove_template(label, template_id)
    except TemplateNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
