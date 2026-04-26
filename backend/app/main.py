import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure root logging so our `dtw.*` loggers reach uvicorn stdout —
# uvicorn only configures `uvicorn.*` itself.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

from app.api import admin, evaluate, health, recognize, templates  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.middleware.body_size import BodySizeLimitMiddleware  # noqa: E402
from app.services.recognizer_service import get_recognizer_service  # noqa: E402

log = logging.getLogger("dtw.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    svc = get_recognizer_service()
    t0 = time.perf_counter()
    # Three cold paths to warm:
    #   1) librosa MFCC plumbing (~600 ms first call) via extract_from_array
    #   2) numba-jitted compute_dtw (~100 ms first call)
    #   3) numba-jitted compute_dtw_normalized (the actual recognize hot path)
    silence = np.zeros(settings.sample_rate, dtype=np.float32)
    feats = svc._extractor.extract_from_array(silence)
    rng = np.random.default_rng(0)
    other = rng.standard_normal(feats.shape).astype(np.float32)
    svc._dtw.compute_dtw(feats, other)
    svc._dtw.compute_dtw_normalized(feats, other)
    log.info(
        "warmup ok (backend=%s) in %.0f ms",
        svc.backend_name,
        (time.perf_counter() - t0) * 1000,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_audio_bytes,
        route_overrides={
            f"{settings.api_prefix}/evaluate": settings.max_evaluate_batch_bytes,
        },
    )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(templates.router, prefix=settings.api_prefix)
    app.include_router(recognize.router, prefix=settings.api_prefix)
    app.include_router(evaluate.router, prefix=settings.api_prefix)
    app.include_router(admin.router, prefix=settings.api_prefix)

    return app


app = create_app()
