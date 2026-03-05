# Design Document — Enterprise Fraud Detection API

This document explains the engineering decisions behind every major component of this system. It is written for engineers who want to understand not just what the system does, but why it was built the way it was, what was considered and rejected, and where the current design has known limitations.

---

## Table of Contents

1. [System Design Goals](#1-system-design-goals)
2. [ML Model Design](#2-ml-model-design)
3. [Feature Engineering Design](#3-feature-engineering-design)
4. [Threshold Design](#4-threshold-design)
5. [Class Imbalance Strategy](#5-class-imbalance-strategy)
6. [API Layer Design](#6-api-layer-design)
7. [Dual Inference Endpoints](#7-dual-inference-endpoints)
8. [Schema Validation Design](#8-schema-validation-design)
9. [Service Initialization Design](#9-service-initialization-design)
10. [Model Integrity Verification](#10-model-integrity-verification)
11. [Database Design](#11-database-design)
12. [LLM Explanation Design](#12-llm-explanation-design)
13. [Prompt Engineering Design](#13-prompt-engineering-design)
14. [Logging Design](#14-logging-design)
15. [Configuration Design](#15-configuration-design)
16. [Containerization Design](#16-containerization-design)
17. [Training Pipeline Design](#17-training-pipeline-design)
18. [Known Limitations and What to Fix Next](#18-known-limitations-and-what-to-fix-next)

---

## 1. System Design Goals

Before any technology choice was made, the following goals were established:

**Correctness over throughput.** A fraud system that returns wrong answers fast is worse than one that returns correct answers slower. Every design decision prioritizes prediction correctness and auditability first.

**Auditability.** Every prediction is logged to the database with its probability, decision, and latency. Every system event is emitted as structured JSON. The system must be able to answer the question: "What did the model predict for transaction X at time T?"

**Operational simplicity.** The system is designed to be deployable by a single engineer. No service mesh, no orchestration platform, no external caches. The dependency graph is: FastAPI → PostgreSQL → OpenAI API. That is intentional.

**Fail loudly at startup, fail gracefully at request time.** If the model file is missing, the database is unreachable, or the configuration is incomplete, the process should refuse to start rather than serve bad predictions silently. At request time, a single failed explanation should not take down the prediction.

---

## 2. ML Model Design

### Why Logistic Regression

The model is `LogisticRegression` from scikit-learn, wrapped in a `Pipeline` with `StandardScaler`. This is not a limitation — it is a deliberate choice for this specific problem.

The input features `V1–V28` are the result of a PCA transformation applied to the original transaction data. PCA decorrelates the features and reduces them to orthogonal components ordered by variance. When the input space is already decorrelated and linearly structured by PCA, Logistic Regression is theoretically appropriate — it models the log-odds of fraud as a linear combination of the PCA components, which is exactly what PCA was designed to support.

More complex models — gradient boosted trees, neural networks — would add training complexity and serving overhead without a clear accuracy benefit when the feature space is already a clean linear decomposition. Logistic Regression on PCA features is interpretable, fast to serve (sub-millisecond inference), and deterministic.

**Model converged in 40 iterations** against a `max_iter` of 2000. The model was not struggling — it found a solution quickly. Increasing model complexity was not warranted by the convergence behavior.

**Inference cost is effectively zero.** The entire prediction pipeline — feature engineering, StandardScaler transform, matrix multiply, sigmoid — runs in under 5ms in-process. The observed 663ms request latency has nothing to do with the model. It is entirely network and database I/O.

### Why Not Tree-Based Models

XGBoost and Random Forest are strong choices for tabular fraud detection in general. The reason they were not chosen here is the input space: V1–V28 are PCA components, not raw features. Tree-based models split on feature thresholds, which makes less geometric sense in a PCA space where each dimension represents a linear combination of the original variables. Logistic Regression's decision boundary is a hyperplane in the PCA space — the correct structure for this data representation.

### Why Not a Neural Network

A neural network on 31 features with ~285,000 training samples would be severe overkill. The risk of overfitting to the minority class pattern is high. Training time, serialization complexity, and serving overhead are all higher. There is no accuracy ceiling being hit with Logistic Regression that would justify the engineering cost.

### Why `lbfgs` Solver

`lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is the default solver for multi-class and well-conditioned binary classification problems in scikit-learn. It handles L2 regularization natively, scales well to ~31 features, and converges stably. It was not changed from the default because the default was correct for this problem size.

### Why L2 Regularization with C=1.0

L2 regularization prevents any single PCA component from dominating the prediction. `C=1.0` is the default regularization strength — medium regularization. Given that PCA components are already orthogonal and scaled similarly, strong regularization was not needed. The model was not showing signs of overfitting (it converged cleanly in 40 iterations), so the default was kept.

---

## 3. Feature Engineering Design

### Raw API Input vs Model Input

The API accepts three types of input: `transaction_time_seconds`, `amount`, and `v[1..28]`. The model requires 31 features. The gap is bridged by `engineer_features()` in `ml.py`, which runs at inference time per request.

```
API Input (30 values)          Model Input (31 features)
─────────────────────────      ──────────────────────────
transaction_time_seconds  →    hour = (seconds // 3600) % 24
amount                    →    amount (passed through)
amount                    →    log_amount = log(amount)
v[0..27]                  →    v1..v28 (passed through)
```

### Why `hour` Instead of Raw Seconds

`transaction_time_seconds` in the creditcard dataset is seconds elapsed since the start of the dataset collection period — not a wall-clock timestamp. The model cannot learn anything useful from absolute elapsed seconds because the offset is dataset-specific and meaningless to a generalization about fraud.

Hour of day (0–23) is the meaningful signal. Fraud patterns in financial transactions are time-of-day dependent — certain hours have elevated fraud rates. Extracting hour via `(seconds // 3600) % 24` converts an arbitrary offset into a cyclical behavioral signal the model can use.

### Why `log_amount`

Transaction amounts in the creditcard dataset have an extremely right-skewed distribution. A few high-value transactions dominate the raw `amount` scale. Without transformation, the model would assign disproportionate weight to the magnitude of large transactions rather than learning the fraud signal.

`log(amount)` compresses the scale, makes the distribution more symmetric, and allows the model to treat a $500 transaction differently from a $5,000 transaction in a way that reflects the actual information content rather than the raw magnitude.

`amount` is also retained as a raw feature because there are fraud patterns specifically at round numbers, specific low amounts (testing transactions), and specific high amounts that the log transform would compress. Both representations carry different signals.

### Why Feature Engineering at Inference Time

`engineer_features()` is called on every request inside `FraudModelService.predict()`. It is not precomputed or cached. This is intentional: the feature engineering is deterministic, stateless, and takes less than 0.1ms. The overhead is negligible. Caching at this level would add complexity with no measurable benefit.

---

## 4. Threshold Design

### The Threshold is 0.8 and It Is Hardcoded

The decision threshold separating "fraud" from "legitimate" is `0.8` — hardcoded in `prepare_data.py` at training time, stored in the `.pkl` bundle, registered in the `model_versions` table, verified at startup against both sources, and confirmed in every LLM explanation.

**Why 0.8 specifically:** In fraud detection, false negatives (missed fraud) and false positives (legitimate transactions flagged) have asymmetric costs. False negatives mean real fraud gets through. False positives mean legitimate customers are inconvenienced or blocked. The threshold of 0.8 was set to minimize false positives — the system only fires a fraud alert when the model is highly confident.

**The probability distribution supports this.** From the 5,000-request load test, the fraud probability output is bimodal:

```
Probability range    Count (of 5,000)
─────────────────    ────────────────
< 0.1                4,002  (80.0%)
0.1 – 0.5              878  (17.6%)
0.5 – 0.8               81   (1.6%)
0.8 – 0.9               17   (0.3%)
> 0.9                   22   (0.4%)
```

The model produces almost no predictions in the 0.5–0.8 range. Most transactions are predicted near 0 (highly legitimate) or near 1 (highly suspicious). This bimodal behavior means the threshold location within 0.5–0.9 has a relatively small effect on the total prediction count — the model has already made up its mind before the threshold is applied.

**The known cost of this threshold:** 2 real fraud transactions were missed in the 5,000-sample test (fraud probabilities of 0.279 and 0.672 — both below threshold). Lowering the threshold to 0.6 would have caught the second missed fraud but would likely increase false positives beyond the already high 35. This is a precision-recall tradeoff that was resolved in favor of precision.

---

## 5. Class Imbalance Strategy

### The Problem

The creditcard dataset is severely imbalanced: approximately 492 fraud cases in 284,807 transactions — a fraud rate of 0.17%. Without intervention, a model that predicts "legitimate" for every transaction achieves 99.83% accuracy while being completely useless for fraud detection.

### Why `class_weight="balanced"`

scikit-learn's `class_weight="balanced"` instructs the model to weight each class inversely proportional to its frequency during training:

```
weight(fraud)  = n_samples / (2 × n_fraud)  ≈ 289×
weight(legit)  = n_samples / (2 × n_legit)  ≈ 0.5×
```

The model is trained as if each fraud case appeared 289 times compared to a legitimate case appearing once. This forces the model to pay attention to the minority class instead of ignoring it to optimize accuracy.

**Why not SMOTE or oversampling:** Synthetic minority oversampling (SMOTE) generates artificial fraud samples by interpolating between existing fraud cases in feature space. For PCA-transformed features, this interpolation is geometrically valid, but it introduces synthetic data that may not represent real fraud patterns. `class_weight="balanced"` achieves the same mathematical effect (adjusting the loss function to emphasize the minority class) without modifying the training data. It is simpler, reproducible, and does not risk overfitting to synthetic samples.

**Why not undersampling:** Discarding 99.8% of legitimate transactions would drastically reduce the training set and throw away real signal about what legitimate transactions look like. The model needs to understand legitimate transaction patterns well to avoid false positives.

### Why Accuracy is a Misleading Metric Here

The system achieves 99.26% accuracy on the 5,000-sample test. This number is almost entirely explained by the class distribution, not model quality. If the model predicted "legitimate" for every single transaction, accuracy would be 4,994/5,000 = 99.88% — higher than what was achieved. The meaningful metrics are:

```
Recall:     66.7%  — catches 4 of 6 real fraud cases
Precision:  10.3%  — of 39 fraud alerts, only 4 were real fraud
F1:         17.8%  — harmonic mean, reflects the precision-recall imbalance
```

The low F1 is a known consequence of the severe class imbalance, the bimodal probability distribution, and the threshold placement. It is not a hidden problem — it is a fully characterized limitation documented here.

---

## 6. API Layer Design

### Why FastAPI

FastAPI was chosen over Flask and Django REST Framework for three reasons:

**Automatic schema validation.** FastAPI integrates with Pydantic natively. Defining `TransactionInput` with `v: List[float] = Field(..., min_length=28, max_length=28)` gives automatic HTTP 422 validation with zero additional code. In Flask, the same validation would require manual checking, custom error handling, or an additional library like Marshmallow.

**Automatic documentation.** FastAPI generates OpenAPI docs at `/docs` from the type annotations and Pydantic models. This is not a development convenience — it is an operational asset. Anyone calling the API can inspect the request/response schema without reading source code.

**Synchronous execution is sufficient.** FastAPI supports both sync and async routes. The ML inference and database calls are synchronous. Running them as `def` (not `async def`) is the correct choice — FastAPI will run sync routes in a threadpool, which prevents blocking the event loop. For this workload, there is no benefit to converting to async.

### Why Uvicorn

Uvicorn is the standard ASGI server for FastAPI. It is production-capable for moderate traffic. The `Dockerfile` starts Uvicorn with `--host 0.0.0.0 --port 8000`, binding to all interfaces for container compatibility. A `--workers N` flag was not added — see Known Limitations.

---

## 7. Dual Inference Endpoints

### Why Two Separate Endpoints

The system exposes two prediction endpoints:

```
POST /predict                     →  ML inference only  (~663ms mean)
POST /predict-with-explanation    →  ML inference + LLM (~2,454ms mean)
```

This is a deliberate architectural separation, not an oversight.

**Different callers have different needs.** Automated fraud screening systems processing thousands of transactions need the binary prediction fast. A fraud analyst reviewing a flagged transaction needs a human-readable explanation. These are different consumers with different latency tolerance and different data requirements.

**LLM latency must not contaminate the fast path.** The OpenAI API call adds ~1,791ms of overhead. If explanations were generated for every prediction, every caller would pay this cost regardless of whether they need it. The separation ensures the fast path remains fast.

**Failure isolation.** If OpenAI's API is degraded, `/predict` continues operating normally. Only `/predict-with-explanation` is affected. If the endpoints were merged, an LLM outage would degrade all predictions. The current design handles this correctly — `generate_explanation()` catches all exceptions and returns a fallback string, so even the explanation endpoint degrades gracefully rather than returning HTTP 500.

### Latency Profile by Endpoint

```
Endpoint                        P50      P75      P90      P95      P99
─────────────────────────────   ──────   ──────   ──────   ──────   ──────
/predict (sequential)           597ms    629ms    674ms    756ms    2603ms
/predict (10 concurrent)        1731ms   2044ms   2719ms   3079ms   3782ms
/predict-with-explanation       2325ms   2713ms   3108ms   3366ms   3984ms
```

The P99 spike in sequential `/predict` (2,603ms against a P95 of 756ms) is a known anomaly consistent with occasional PostgreSQL connection establishment delays on a self-hosted server.

---

## 8. Schema Validation Design

### Why Pydantic v2

Pydantic v2 (2.7.4) provides validation at the API boundary before any application code runs. Malformed requests are rejected at the framework level with descriptive errors, never reaching `FraudModelService`.

The critical constraint is `v: List[float] = Field(..., min_length=28, max_length=28)`. This enforces that the V-feature vector is exactly 28 elements — matching V1–V28 in the training data. A request with 27 or 29 features returns HTTP 422 immediately. Without this, a mismatched feature vector would reach `engineer_features()` and fail at a `RuntimeError` inside ML code, which is the wrong layer to be catching API contract violations.

### Why the Schema Validates Length But Not Value Range

The V-features are PCA components. Their meaningful range depends on the distribution of the training data — it is not a fixed interval. Enforcing, say, `ge=-10, le=10` would be an arbitrary constraint that could reject valid transactions from unusual periods. Pydantic validates structure (type, length), not semantics. Semantic validation belongs to the model.

---

## 9. Service Initialization Design

### Why Singletons, Not Per-Request Loading

Both `FraudModelService` and `ExplanationService` are instantiated once at module load time in `routes.py` and shared across all requests for the lifetime of the process:

```python
model_service = FraudModelService()    # runs once at startup
llm_service = ExplanationService()     # runs once at startup
```

`FraudModelService.__init__()` performs a database query, a filesystem read (`joblib.load`), and an integrity check. This takes hundreds of milliseconds. If this ran on every request, the ML endpoint latency would increase by 500–1,000ms on every call — worse than the current observed latency with the database write included.

The model pipeline itself (StandardScaler + LogisticRegression) holds a large in-memory object. Loading and deserializing it per request is not only slow — it is memory-wasteful, causing repeated garbage collection pressure.

**Thread safety:** Logistic Regression inference in scikit-learn is read-only. The `pipeline.predict_proba()` call does not mutate any state on the pipeline object. Multiple threads calling `model_service.predict()` concurrently is safe. The in-memory LLM token counters in `ExplanationService` are updated without a lock — under concurrent writes, counter values may be slightly off, but this is an acceptable tradeoff for a monitoring metric. It does not affect predictions.

---

## 10. Model Integrity Verification

### The Problem It Solves

The model is registered in two places: the `model_versions` database table (stores path, threshold, feature list) and the `.pkl` file on disk (stores the actual pipeline, threshold, feature list). These two records can drift independently. A developer might update the DB threshold without retraining. Someone might swap the `.pkl` file without updating the DB record.

If the system served predictions under a threshold mismatch, every prediction would be silently wrong — either too aggressive or too permissive — with no indication of the problem.

### The Solution

`_validate_model_integrity()` runs at startup before the API accepts any traffic:

```python
if float(self.db_threshold) != float(self.serialized_threshold):
    raise RuntimeError("Threshold mismatch between DB and model file")

if self.db_feature_columns != self.serialized_features:
    raise RuntimeError("Feature list mismatch between DB and model file")
```

If either check fails, the process exits with a clear error message. The system refuses to start rather than serve wrong predictions. This is the correct failure mode: loud and early.

**Why this check is at startup, not per-request:** The model is loaded once. Checking integrity per request would re-read the DB on every prediction call, adding unnecessary latency and database load for a check that only matters when the model file changes.

---

## 11. Database Design

### Why PostgreSQL

PostgreSQL is the storage layer for three distinct purposes: model registry, prediction audit log, and raw training data. A single relational database serves all three because the access patterns are simple, the data volumes are manageable, and the operational overhead of a multi-database setup is not justified.

### Table Design Rationale

**`model_versions`** is a model registry. It stores the metadata needed to load and validate a model: path, threshold, feature list, and training fraud rate. The `MODEL_VERSION_ID` environment variable controls which model version the API serves. Swapping to a different model requires only changing this environment variable and restarting — no code change.

**`predictions`** is an immutable audit log. Every prediction is appended with its probability, binary decision, and latency. This table answers operational questions: What is the average latency over the last hour? What fraction of predictions are flagged as fraud? Are fraud rates trending up? The `fetch_metrics()` function runs three aggregation queries against this table.

**`transactions`** stores the raw creditcard dataset after ingestion via `load_data.py`. It is the training data source — `prepare_data.py` reads from it to train the model. It is not accessed during inference.

### Why Per-Request Connections Without Pooling

This is the most significant known design limitation, documented honestly here.

`db.py` opens a new PostgreSQL connection for every database operation and closes it when done. There is no connection pool. Under 10 concurrent workers, this caused a 2.9× latency increase (663ms → 1,917ms mean) because connection establishment under contention becomes serialized.

The reason this exists as-is: `psycopg2`'s `ThreadedConnectionPool` adds roughly 15 lines of code and a pool configuration decision (min/max connections). It was not added in the initial implementation. The performance impact is real and documented, and connection pooling is the highest-priority fix for scaling this system.

### Why `log_prediction()` Silently Swallows Exceptions

```python
except Exception:
    pass
```

The prediction audit log write is designed to never fail the prediction itself. If the database is temporarily unreachable during a prediction request, the API should still return the fraud decision to the caller. The alternative — failing the prediction if the audit log write fails — would cause a database outage to take down the prediction service.

This is a reasonable tradeoff. The cost is that failed audit log writes are invisible — no log event, no counter, no alert. The correct fix is to emit a `log_event("audit_log_failure", {...})` inside the except block rather than silently passing. Currently it is silent. That is a known gap.

---

## 12. LLM Explanation Design

### Why GPT-4o-mini

`gpt-4o-mini` was chosen over `gpt-4o` or `gpt-3.5-turbo` for this task. The explanation task is narrow and well-structured: given three numbers (probability, prediction, threshold), generate a four-sentence explanation. This does not require the reasoning depth of `gpt-4o`. `gpt-4o-mini` handles this task correctly, produces explanations at the required quality level, and costs significantly less per call.

Verified token cost per call: **144 tokens average** (84 input, 60 output). This is a small, predictable footprint per explanation.

### Why Temperature 0.2

Temperature controls the randomness of the LLM output. `temperature=0.0` produces deterministic outputs — identical inputs produce identical outputs. `temperature=1.0` produces highly varied outputs.

The explanation task requires consistency, not creativity. Two predictions with the same probability and the same threshold should produce similar explanations. Temperature 0.2 introduces enough variation to avoid robotic repetition while keeping explanations factual and stable. Higher temperatures would risk the LLM wandering into speculation or varying the explanation tone in ways that could confuse fraud analysts.

### Why the LLM Receives Only Three Values

The LLM prompt receives only: `fraud_probability`, `prediction`, and `threshold`. It does not receive:
- The 28 PCA components (V1–V28)
- The transaction amount
- The hour of day
- Any customer or account information

This is intentional. PCA components are mathematically transformed features — they have no direct interpretable meaning to a human or an LLM. Passing V3 = -1.35 to a language model and asking it to explain the fraud risk from that number would produce hallucinated interpretations. The LLM would invent meanings for decorrelated PCA dimensions it cannot actually interpret.

By restricting the LLM to only the model's output (probability, decision, threshold), the explanations are always grounded in what the model actually decided. The LLM translates the prediction result into plain language — it does not reinterpret the transaction.

---

## 13. Prompt Engineering Design

### The Full Prompt

```
You are explaining fraud detection results.

Rules:
- Do NOT change the decision.
- Do NOT invent new facts.
- Only explain the structured result.
- Keep explanation under 4 sentences.
- Be concise and factual.

Structured Result:
{fraud_probability, prediction, threshold}
```

### Why Explicit Prohibitions

The most important lines in the prompt are the prohibitions: "Do NOT change the decision" and "Do NOT invent new facts."

Without the first prohibition, an LLM given a prediction of 1 (fraud) and a probability of 0.82 might say "While the model flagged this as fraud, the probability is relatively low, suggesting the transaction may be legitimate." This contradicts the model's decision and is dangerous in a fraud system — it introduces the LLM as a second opinion that can silently override the ML model.

Without the second prohibition, the LLM might fabricate reasons like "This transaction was flagged because of an unusual IP address" when no such information was provided. In a fraud system used for auditing, hallucinated facts are a compliance liability.

The prohibitions are not guardrails added from caution — they define the exact task boundary. The LLM is an explainer, not a decision-maker.

### Why "Under 4 Sentences"

Fraud analysts reviewing decisions need concise explanations. Long explanations increase cognitive load and obscure the key information. Four sentences is enough to state the probability, explain the threshold relationship, give the resulting decision, and add one line of context. More would be noise.

### Verified Behavior

From the 50-request LLM evaluation, all sampled explanations correctly reported the probability and threshold values, did not contradict the prediction, and stayed within the length constraint. The prompt guardrails are functioning as designed.

---

## 14. Logging Design

### Why Structured JSON Logging

Every system event is emitted as a JSON object to stdout:

```json
{
  "timestamp": "2026-03-02T22:59:14.123456",
  "event_type": "prediction_made",
  "model_version_id": 1,
  "fraud_probability": 0.9987,
  "prediction": 1,
  "latency_ms": 634.2
}
```

Free-text log lines like `"Prediction made: 0.9987"` are unqueryable. Structured JSON can be ingested directly by log aggregation systems (Datadog, Loki, CloudWatch Logs Insights, Elasticsearch) and queried with filters. In a containerized deployment, stdout is the correct output target — the container runtime captures it, and the log aggregator picks it up from there.

### Why a Named Logger with Handler Guard

The logger is named `"fraud_system"` and uses `if not logger.handlers` before adding a handler. This prevents duplicate log lines when Uvicorn's hot reload re-imports the module without restarting the process. Without the guard, every reload would add another handler to the logger, causing each event to be printed N times. This is a common Python logging mistake — it is handled correctly here.

### What Is Logged vs What Is Not

| Event | Logged |
|-------|--------|
| System startup | ✅ with model_version_id |
| Every prediction (/predict) | ✅ with probability, decision, latency |
| Every prediction (/predict-with-explanation) | ✅ with LLM call count |
| ML inference errors | ✅ with error string |
| LLM API errors | ❌ silently returns fallback |
| DB write failures (audit log) | ❌ silently swallowed |
| Invalid request (HTTP 422) | ❌ not emitted by application code |

The two missing log events (LLM errors, DB audit failures) are known gaps. They should emit warning-level log events so operators can detect degraded subsystems without having to infer from missing records.

---

## 15. Configuration Design

### Why All Config at Import Time

`config.py` reads all seven environment variables into a `Settings` Pydantic model when the module is first imported — before the API accepts any traffic. Missing variables raise `RuntimeError` immediately.

This is fail-fast design. The alternative — reading config lazily on first request — means a misconfigured container serves requests until someone triggers the specific code path that needs the missing variable. With eager loading, a missing `OPENAI_API_KEY` is caught at container startup, not after the first call to `/predict-with-explanation`.

### Why Pydantic `BaseModel` for Settings

Pydantic validates that `MODEL_VERSION_ID` can be cast to `int` (`int(get_env("MODEL_VERSION_ID"))`). A `Settings` model provides type coercion and validation for free. If someone sets `MODEL_VERSION_ID=abc` in the environment, the error is caught at startup with a clear message, not at the database query level with a cryptic psycopg2 type error.

---

## 16. Containerization Design

### Why `python:3.12-slim`

`python:3.12-slim` is the minimal Debian-based Python image. The full `python:3.12` image includes development tools, documentation, and test utilities that have no place in a production container. `slim` reduces image size while retaining `apt-get` for system-level dependency installation.

`python:3.12-alpine` was not used. Alpine uses `musl libc` instead of `glibc`. `psycopg2-binary` and several NumPy/SciPy wheels are compiled against `glibc` — they either fail to install on Alpine or require compiling from source, which significantly complicates the build process.

### Why `build-essential` and `libpq-dev` Are Installed

`psycopg2-binary` includes a pre-compiled PostgreSQL client binary, but it still requires `libpq-dev` (the PostgreSQL client library headers and runtime) to link against at install time in the Docker environment. `build-essential` provides the C compiler needed during pip installation of any package with C extensions. These are necessary build dependencies.

### Why `requirements.txt` Is Copied Before Application Code

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
```

Docker builds layers. If application code changes but `requirements.txt` does not, the `pip install` layer is pulled from cache. This is the standard Docker layer optimization pattern. Without it, every code change would re-download and re-install all Python dependencies — adding minutes to every build.

### Why the Model Is Baked Into the Image

`COPY models ./models` copies the `.pkl` file into the Docker image at build time. The model is part of the container, not mounted from an external volume.

**Advantage:** The container is fully self-contained. No external object storage dependency, no startup download latency, no risk of serving a container that can't reach its model file.

**Known cost:** A new model version requires rebuilding the image and redeploying. In a high-frequency retraining environment, this creates deployment friction. The correct evolution is to store models in object storage (S3 or equivalent) and download the appropriate version at container startup based on `MODEL_VERSION_ID`. This was not implemented in the current version.

---

## 17. Training Pipeline Design

### Why the Training Data Flows Through PostgreSQL

The raw dataset is ingested into the `transactions` table via `load_data.py` and read back for training via `prepare_data.py`. The training data does not go directly from CSV to model training.

This indirection provides a durable, queryable training data source. Future training runs can filter, join, or augment the transaction data using SQL without modifying the Python training code. It also means the training data has the same access controls, backup behavior, and audit trail as the rest of the system's data.

### Why Stratified Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

`stratify=y` ensures that the 0.17% fraud rate is preserved in both the training set and test set. Without stratification, random sampling on a severely imbalanced dataset could produce a test set with zero fraud cases — making evaluation meaningless. With 284,807 samples and ~492 fraud cases, an unstratified 20% test split might contain 80–120 fraud cases by chance. Stratification guarantees approximately 98 fraud cases in the test set (20% of 492), providing a stable evaluation baseline.

`random_state=42` ensures reproducibility. The same train/test split is produced on every run, making evaluation results comparable across training iterations.

### Why the Pipeline Includes StandardScaler

Logistic Regression is sensitive to feature scale. `amount` has values ranging from near zero to thousands. `hour` ranges from 0 to 23. `log_amount` ranges from 0 to ~8. V1–V28 (PCA components) have varying scales depending on the explained variance of each component.

Without scaling, the logistic regression solver would assign larger weights to high-magnitude features not because they are more predictive, but because they are numerically larger. `StandardScaler` normalizes each feature to zero mean and unit variance, ensuring the optimizer treats all features on equal footing.

The scaler is **fit on the training data only** and applied to both training and test data. Fitting the scaler on the full dataset before the split would constitute data leakage — the test set statistics would influence the training normalization. The Pipeline ensures this is handled correctly: `scaler.fit_transform(X_train)` and `scaler.transform(X_test)`.

---




