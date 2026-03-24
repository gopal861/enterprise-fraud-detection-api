import time
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.schemas import (
    TransactionInput,
    PredictionResponse,
    PredictionWithExplanationResponse,
    MetricsResponse,
)
from app.ml import FraudModelService
from app.llm import ExplanationService
from app.db import log_prediction, fetch_metrics
from app.logger import log_event

router = APIRouter()

# Initialize services once
model_service = FraudModelService()
llm_service = ExplanationService()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_version_id": settings.model_version_id,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    start_time = time.time()

    try:
        prob, prediction = model_service.predict(
            transaction.transaction_time_seconds,
            transaction.amount,
            transaction.v,
        )
    except Exception as e:
        log_event(
            "prediction_error",
            {
                "model_version_id": settings.model_version_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(settings.model_version_id, prob, prediction, latency_ms)

    log_event(
        "prediction_made",
        {
            "model_version_id": settings.model_version_id,
            "fraud_probability": prob,
            "prediction": prediction,
            "latency_ms": latency_ms,
        },
    )

    return {
        "fraud_probability": prob,
        "prediction": prediction,
        "threshold": model_service.serialized_threshold,
        "latency_ms": latency_ms,
    }


@router.post(
    "/predict-with-explanation",
    response_model=PredictionWithExplanationResponse,
)
def predict_with_explanation(transaction: TransactionInput):
    start_time = time.time()

    try:
        prob, prediction = model_service.predict(
            transaction.transaction_time_seconds,
            transaction.amount,
            transaction.v,
        )
    except Exception as e:
        log_event(
            "prediction_error",
            {
                "model_version_id": settings.model_version_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))

    explanation = llm_service.generate_explanation(
        prob,
        prediction,
        model_service.serialized_threshold,
    )

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(settings.model_version_id, prob, prediction, latency_ms)

    log_event(
        "prediction_with_explanation",
        {
            "model_version_id": settings.model_version_id,
            "fraud_probability": prob,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "llm_total_calls": llm_service.total_llm_calls,
        },
    )

    return {
        "fraud_probability": prob,
        "prediction": prediction,
        "threshold": model_service.serialized_threshold,
        "latency_ms": latency_ms,
        "explanation": explanation,
    }


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    metrics_data = fetch_metrics(settings.model_version_id)
    llm_usage = llm_service.get_llm_usage()

    log_event(
        "metrics_requested",
        {
            "model_version_id": settings.model_version_id,
        },
    )

    return {
        "model_version_id": settings.model_version_id,
        "total_predictions": metrics_data["total_predictions"],
        "fraud_predictions": metrics_data["fraud_predictions"],
        "fraud_rate": metrics_data["fraud_rate"],
        "average_latency_ms": metrics_data["average_latency_ms"],
        "llm_total_calls": llm_usage["total_llm_calls"],
        "llm_total_prompt_tokens": llm_usage["total_prompt_tokens"],
        "llm_total_completion_tokens": llm_usage["total_completion_tokens"],
        "llm_total_tokens": llm_usage["total_tokens"],
    }

@router.get("/")
def root():
    return {
        "status": "running",
        "service": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/predict",
            "/predict-with-explanation",
            "/metrics"
        ]
    }