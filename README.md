# Enterprise Fraud Detection API

A production-grade, real-time fraud detection system that classifies financial transactions using a trained machine learning model and generates human-readable explanations via GPT-4o-mini. Deployed on AWS EC2 with AWS RDS PostgreSQL as the backend database.


- Served at: `https://fraud-detection-api.duckdns.org`

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Request & Response](#request--response)
- [ML Model](#ml-model)
- [LLM Explanation Layer](#llm-explanation-layer)
- [Performance](#performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [AWS Deployment](#aws-deployment)
- [Training Pipeline](#training-pipeline)
- [Model Metrics](#model-metrics)
- [Known Limitations](#known-limitations)

---

## Overview

This system exposes a REST API that accepts a financial transaction and returns a real-time fraud decision. It runs two distinct inference paths:

- **`/predict`** — ML-only inference. Mean latency **663ms**. Returns fraud probability, binary decision, and threshold used.
- **`/predict-with-explanation`** — ML inference followed by a GPT-4o-mini explanation. Mean latency **2,455ms**. Returns everything above plus a plain-English explanation suitable for fraud analysts and audit logs.

All predictions are logged to AWS RDS PostgreSQL for auditability. Every system event is emitted as structured JSON to stdout for log aggregation.

---

## Architecture

```
Client
  │
  ▼  HTTPS
┌─────────────────────────────────────────────────────┐
│                  AWS EC2 Instance                   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         Docker Container (port 8000)         │   │
│  │                                              │   │
│  │   FastAPI  ──►  ML Service (sklearn)         │   │
│  │               ──►  LLM Service (OpenAI API)  │   │
│  └────────────────────────┬─────────────────────┘   │
│                           │                         │
└───────────────────────────┼─────────────────────────┘
                            │  psycopg2
                            ▼
               ┌────────────────────────┐
               │      AWS RDS           │
               │   PostgreSQL           │
               │                        │
               │  model_versions        │
               │  predictions           │
               │  transactions          │
               └────────────────────────┘
                            │
                            │  (external)
                            ▼
                   ┌─────────────────┐
                   │   OpenAI API    │
                   │  gpt-4o-mini    │
                   └─────────────────┘
```

**Deployment:**
- Compute: AWS EC2
- Database: AWS RDS PostgreSQL
- Container runtime: Docker
- Served at: `https://fraud-detection-api.duckdns.org`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check. Returns active model version ID. |
| `POST` | `/predict` | ML inference. Returns fraud probability and binary decision. |
| `POST` | `/predict-with-explanation` | ML inference + GPT-4o-mini explanation. |
| `GET` | `/metrics` | Aggregate prediction stats and LLM token usage. |

---

## Request & Response

### `POST /predict`

**Request body:**

```json
{
  "transaction_time_seconds": 406,
  "amount": 149.62,
  "v": [-1.3598, -0.0727, 2.5363, 1.3781, -0.3383,
         0.4623,  0.2395,  0.0986,  0.3637,  0.0907,
        -0.5515, -0.6177, -0.9913, -0.3111,  1.4681,
        -0.4703,  0.2079,  0.0257,  0.4039,  0.2514,
        -0.0183,  0.2778, -0.1104,  0.0669,  0.1285,
        -0.1891,  0.1335, -0.0210]
}
```

| Field | Type | Constraint | Description |
|-------|------|-----------|-------------|
| `transaction_time_seconds` | `int` | required | Seconds elapsed since dataset epoch |
| `amount` | `float` | required | Transaction amount |
| `v` | `List[float]` | exactly 28 elements | PCA-transformed features V1–V28 |

**Response:**

```json
{
  "fraud_probability": 0.0023,
  "prediction": 0,
  "threshold": 0.8,
  "latency_ms": 634.2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fraud_probability` | `float` | Model confidence score [0.0 – 1.0] |
| `prediction` | `int` | `0` = legitimate, `1` = fraud |
| `threshold` | `float` | Decision threshold applied (0.8) |
| `latency_ms` | `float` | End-to-end request latency |

---

### `POST /predict-with-explanation`

Same request body as `/predict`.

**Response:**

```json
{
  "fraud_probability": 0.9987,
  "prediction": 1,
  "threshold": 0.8,
  "latency_ms": 2341.5,
  "explanation": "The fraud probability for this transaction is 99.87%, which significantly exceeds the decision threshold of 80%. As a result, the transaction has been classified as fraudulent. This high probability indicates strong model confidence that the transaction exhibits patterns consistent with fraud."
}
```

---

### `GET /metrics`

**Response:**

```json
{
  "model_version_id": 1,
  "total_predictions": 5000,
  "fraud_predictions": 39,
  "fraud_rate": 0.0078,
  "average_latency_ms": 1917.4,
  "llm_total_calls": 122,
  "llm_total_prompt_tokens": 10272,
  "llm_total_completion_tokens": 7330,
  "llm_total_tokens": 17602
}
```

> **Note:** `llm_*` fields are in-memory counters scoped to the current process lifetime. They reset on container restart.

---

### `GET /health`

```json
{
  "status": "ok",
  "model_version_id": 1
}
```

---

### Error Responses

| Code | Cause |
|------|-------|
| `422` | Request validation failed — wrong types, missing fields, or `v` length ≠ 28 |
| `500` | ML inference error — logged to stdout with event type `prediction_error` |

---

## ML Model

The model is a scikit-learn `Pipeline` serialized to `.pkl` and loaded at container startup.

```
Input (31 features)
    └─► StandardScaler          normalize to zero mean, unit variance
    └─► LogisticRegression      predict_proba → fraud probability
    └─► threshold = 0.8         int(prob >= 0.8) → 0 or 1
```

### Model Configuration

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression |
| Solver | lbfgs |
| Penalty | L2 |
| Regularization C | 1.0 |
| `class_weight` | balanced |
| Decision threshold | 0.8 |
| Features | 31 |
| Training convergence | 40 iterations |

### Features

| Feature | Engineering |
|---------|-------------|
| `amount` | Passed through raw |
| `hour` | `(transaction_time_seconds // 3600) % 24` |
| `log_amount` | `log(amount)`; `0` if amount is zero |
| `v1` – `v28` | PCA-transformed features, passed through |

The model receives 31 features total. Feature engineering (`hour`, `log_amount`) runs at inference time per request inside `FraudModelService.engineer_features()`.

### Model Integrity Verification

At every startup, the system cross-validates the model bundle (`.pkl`) against the `model_versions` database record. If the stored threshold or feature list does not match what is inside the `.pkl`, the process raises a `RuntimeError` and exits before serving any traffic. This prevents misconfigured models from silently serving wrong predictions.

### Class Imbalance

The training dataset (284,807 transactions, ~0.17% fraud) is severely imbalanced. The model uses `class_weight="balanced"`, which weights fraud cases approximately **289×** more than legitimate cases during training:

```
weight(fraud)  = 284,807 / (2 × 492)     ≈ 289×
weight(legit)  = 284,807 / (2 × 284,315) ≈ 0.5×
```

This prevents the model from defaulting to "always predict legitimate" to optimize accuracy.

---

## LLM Explanation Layer

The `/predict-with-explanation` endpoint passes the model's output to GPT-4o-mini, which generates a concise, factual explanation for fraud analysts and audit purposes.

### What the LLM receives

```python
{
    "fraud_probability": 0.9987,
    "prediction": 1,
    "threshold": 0.8
}
```

The LLM does **not** receive raw transaction features, PCA components, or any account data. It translates the model's decision into plain language — it does not re-analyze the transaction.

### Prompt constraints

- `Do NOT change the decision` — LLM cannot contradict the ML model output
- `Do NOT invent new facts` — LLM cannot fabricate transaction details
- `Only explain the structured result` — restricted to what was provided
- `Keep explanation under 4 sentences` — concise for analyst consumption

### Token usage (observed — 122 calls)

| Metric | Value |
|--------|-------|
| Avg prompt tokens / call | 84 |
| Avg completion tokens / call | 60 |
| Avg total tokens / call | 144 |
| Total tokens (122 calls) | 17,602 |

---

## Performance

All figures measured against the live deployed API.

### Latency Percentiles

```
                              P50      P75      P90      P95      P99
──────────────────────────────────────────────────────────────────────
/predict       (sequential)   597ms    629ms    674ms    756ms   2603ms
/predict       (10 workers)  1731ms   2044ms   2719ms   3079ms   3782ms
/predict-with-explanation    2325ms   2713ms   3108ms   3366ms   3984ms
```

- Sequential: 200 requests, 1 worker, 0 errors
- Concurrent: 5,000 requests, 10 threads, 0 errors
- LLM: 50 requests, 1 worker, 0 errors

### Throughput (single Uvicorn worker)

```
/predict                  ~1.5 req/sec
/predict-with-explanation ~0.4 req/sec
```

### Latency Breakdown (`/predict`)

The dominant cost is **PostgreSQL connection establishment**, not ML inference:

```
ML inference (feature eng + scaler + LR)   ~5ms
PostgreSQL connect + INSERT + close        ~260ms
Network round-trip to RDS                  ~300ms
─────────────────────────────────────────────────
Total sequential mean                      ~663ms
```

### Concurrency Degradation

| Workers | Mean Latency | Factor |
|---------|-------------|--------|
| 1 (sequential) | 663ms | baseline |
| 10 (concurrent) | 1,917ms | 2.9× |

The 2.9× degradation under 10 workers is caused by per-request RDS connection establishment without a connection pool. See [Known Limitations](#known-limitations).

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| API framework | FastAPI | 0.111.0 |
| ASGI server | Uvicorn | 0.30.1 |
| Schema validation | Pydantic | 2.7.4 |
| ML library | scikit-learn | 1.5.0 |
| Numerical | NumPy | 1.26.4 |
| Data processing | pandas | 2.2.2 |
| Model serialization | joblib | 1.4.2 |
| Database driver | psycopg2-binary | 2.9.9 |
| LLM client | openai SDK | 1.35.13 |
| Runtime | Python | 3.12-slim |
| Containerization | Docker | — |
| Compute | AWS EC2 | — |
| Database | AWS RDS PostgreSQL | — |

---

## Project Structure

```
fraud_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, router registration, startup log
│   ├── routes.py            # All API endpoints, service singletons
│   ├── schemas.py           # Pydantic request/response models
│   ├── ml.py                # Model load, integrity check, feature engineering, inference
│   ├── llm.py               # OpenAI client, explanation generation, token tracking
│   ├── prompts.py           # LLM prompt construction
│   ├── db.py                # PostgreSQL: model metadata, prediction logging, metrics
│   ├── config.py            # Environment variable loading via Pydantic Settings
│   └── logger.py            # Structured JSON logging to stdout
│
├── models/
│   └── fraud_model_YYYYMMDD_HHMMSS.pkl
│
├── scripts/
│   ├── load_data.py         # Ingest creditcard.csv into RDS transactions table
│   └── prepare_data.py      # Train model, save .pkl, print evaluation report
│
├── data/
│   └── creditcard.csv       # 284,807 transactions, 30 features + Class label
│
├── experiments/
│   ├── baseline_test.py          # 200-request sequential benchmark
│   ├── dataset_load_test.py      # 5,000-request concurrent benchmark with true labels
│   ├── llm_evaluation.py         # LLM endpoint latency and output evaluation
│   └── failure_analysis.py       # Extract false positives and false negatives
│
├── Dockerfile
├── requirements.txt
├── architecture.md
├── design.md
└── README.md
```

---

## Environment Variables

All configuration is read from environment variables at startup. The process exits immediately if any variable is missing.

| Variable | Required | Description |
|----------|----------|-------------|
| `DB_NAME` | ✅ | RDS PostgreSQL database name |
| `DB_USER` | ✅ | RDS database user |
| `DB_PASSWORD` | ✅ | RDS database password |
| `DB_HOST` | ✅ | RDS endpoint hostname |
| `DB_PORT` | ✅ | RDS port (typically `5432`) |
| `MODEL_VERSION_ID` | ✅ | ID of the model version to load from `model_versions` table |
| `OPENAI_API_KEY` | ✅ | OpenAI API key for GPT-4o-mini |

---

## Local Setup

### Prerequisites

- Docker
- PostgreSQL instance (local or RDS)
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/your-org/fraud-detection-api.git
cd fraud-detection-api
```

### 2. Set up the database

Connect to your PostgreSQL instance and create the required tables:

```sql
CREATE TABLE model_versions (
    id                  SERIAL PRIMARY KEY,
    model_path          TEXT,
    threshold           NUMERIC,
    feature_columns     TEXT,
    training_fraud_rate NUMERIC
);

CREATE TABLE predictions (
    id                SERIAL PRIMARY KEY,
    model_version_id  INTEGER,
    fraud_probability NUMERIC,
    prediction        INTEGER,
    latency_ms        NUMERIC
);

CREATE TABLE transactions (
    id                        SERIAL PRIMARY KEY,
    transaction_time_seconds  INTEGER,
    amount                    FLOAT,
    v1  FLOAT, v2  FLOAT, v3  FLOAT, v4  FLOAT, v5  FLOAT, v6  FLOAT, v7  FLOAT,
    v8  FLOAT, v9  FLOAT, v10 FLOAT, v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT,
    v15 FLOAT, v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT, v21 FLOAT,
    v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT, v26 FLOAT, v27 FLOAT, v28 FLOAT,
    is_fraud BOOLEAN
);
```

### 3. Train the model

```bash
# Ingest dataset into the transactions table
python scripts/load_data.py

# Train and save model
python scripts/prepare_data.py
```

The training script will output evaluation metrics and save the model to `models/fraud_model_{timestamp}.pkl`.

### 4. Register the model in the database

```sql
INSERT INTO model_versions (model_path, threshold, feature_columns, training_fraud_rate)
VALUES (
    'models/fraud_model_YYYYMMDD_HHMMSS.pkl',
    0.8,
    'amount,hour,log_amount,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28',
    0.0017
);
```

Note the `id` of the inserted row — this is your `MODEL_VERSION_ID`.

### 5. Build and run the container

```bash
docker build -t fraud-detection-api .

docker run -d \
  -p 8000:8000 \
  -e DB_NAME=fraud_detection \
  -e DB_USER=your_user \
  -e DB_PASSWORD=your_password \
  -e DB_HOST=your_rds_endpoint \
  -e DB_PORT=5432 \
  -e MODEL_VERSION_ID=1 \
  -e OPENAI_API_KEY=sk-... \
  fraud-detection-api
```

### 6. Verify the deployment

```bash
curl https://localhost:8000/health
```

Expected:

```json
{ "status": "ok", "model_version_id": 1 }
```

---

## AWS Deployment

### EC2

1. Launch an EC2 instance (Amazon Linux 2 or Ubuntu 22.04 recommended)
2. Install Docker on the instance
3. Transfer or pull the Docker image
4. Inject environment variables via EC2 user data, AWS Systems Manager Parameter Store, or AWS Secrets Manager
5. Run the container with `-p 8000:8000`
6. Configure the EC2 security group to allow inbound traffic on port 8000 (or 443 if behind a load balancer)

### RDS

1. Create an RDS PostgreSQL instance in the same VPC as the EC2 instance
2. Set the RDS security group to allow inbound connections from the EC2 instance's security group on port 5432
3. Use the RDS endpoint hostname as `DB_HOST`
4. The database does **not** need to be publicly accessible — EC2-to-RDS traffic stays within the VPC

### Environment Variable Management

Do not pass secrets as plain Docker `-e` flags in production. Use one of:

- **AWS Systems Manager Parameter Store** — store each variable as a SecureString parameter and inject at container startup
- **AWS Secrets Manager** — store DB credentials and OpenAI key as secrets, retrieve via SDK at startup
- **EC2 IAM Instance Profile** — grant the EC2 instance permission to read from Parameter Store or Secrets Manager without hardcoding credentials

### Model Update on EC2

```bash
# 1. Retrain locally or on EC2
python scripts/prepare_data.py

# 2. Register the new model version in RDS
# INSERT INTO model_versions ...

# 3. Rebuild the image (model is baked in via COPY models ./models)
docker build -t fraud-detection-api .

# 4. Stop the running container
docker stop <container_id>

# 5. Start the new container with the updated MODEL_VERSION_ID
docker run -d \
  -p 8000:8000 \
  -e MODEL_VERSION_ID=2 \
  ... \
  fraud-detection-api
```

---

## Training Pipeline

The training pipeline is a standalone offline process — it does not interact with the serving API.

```
creditcard.csv
    │
    ▼  scripts/load_data.py
INSERT into RDS transactions table
    │
    ▼  scripts/prepare_data.py
SELECT * FROM transactions
    │
engineer features: hour, log_amount
    │
train_test_split(test_size=0.2, stratify=y, random_state=42)
    │
Pipeline.fit(X_train, y_train)
  StandardScaler → LogisticRegression(balanced, max_iter=2000)
    │
evaluate on test set → prints confusion matrix + classification report
    │
joblib.dump({pipeline, threshold=0.8, features}) → models/fraud_model_{timestamp}.pkl
```

**Stratified split** preserves the 0.17% fraud rate in both train and test sets. Without stratification, the test set could contain zero fraud cases on a dataset this imbalanced.

**`random_state=42`** ensures reproducible train/test splits across training runs for consistent evaluation comparison.

---

## Model Metrics

Evaluated against 5,000 real transactions from the creditcard dataset.
True fraud prevalence in sample: **6 of 5,000 (0.12%)**.

| Metric | Value |
|--------|-------|
| Accuracy | 99.26% |
| Precision | 10.26% |
| Recall | 66.67% |
| F1 Score | 17.78% |

```
Confusion Matrix (5,000 transactions)
──────────────────────────────────────
                Predicted 0   Predicted 1
Actual 0           4,959           35     ← 35 false positives
Actual 1               2            4     ← 2 false negatives, 4 caught
```

**Accuracy is not the primary metric here.** A model that predicts "legitimate" on every transaction would achieve 99.88% accuracy. The meaningful signal is recall — the system caught 4 of 6 real fraud cases (66.7%) — and precision — 10.3% of fraud alerts corresponded to actual fraud.

The low precision (high false positive rate) is a consequence of the severe class imbalance and the bimodal probability distribution: the model fires alerts confidently (probability > 0.82) on 35 legitimate transactions, suggesting a specific cluster of legitimate transactions that share surface-level similarity with fraud patterns in PCA space.

---

## Known Limitations

### No Database Connection Pooling

Each request to RDS opens and closes a new TCP connection. Under 10 concurrent workers, this caused a **2.9× latency increase** (663ms → 1,917ms mean). The fix is `psycopg2.pool.ThreadedConnectionPool` — not yet implemented.

### Single Uvicorn Worker

The container starts one Uvicorn worker process. All requests are handled sequentially within one event loop. For higher throughput, add `--workers 4` to the Uvicorn command in the Dockerfile.








