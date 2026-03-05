from fastapi import FastAPI
from app.routes import router
from app.config import settings


app = FastAPI(title="Enterprise Fraud Detection API")

app.include_router(router)

from app.logger import log_event

log_event(
    "system_startup",
    {
        "model_version_id":settings.model_version_id,
    },
)