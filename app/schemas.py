from pydantic import BaseModel, Field
from typing import List


class TransactionInput(BaseModel):
    transaction_time_seconds: int
    amount: float
    v: List[float] = Field(..., min_length=28, max_length=28)


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    threshold: float
    latency_ms: float


class PredictionWithExplanationResponse(PredictionResponse):
    explanation: str


class MetricsResponse(BaseModel):
    model_version_id: int
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    average_latency_ms: float | None
    llm_total_calls: int
    llm_total_prompt_tokens: int
    llm_total_completion_tokens: int
    llm_total_tokens: int