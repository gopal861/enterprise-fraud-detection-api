import psycopg2
from contextlib import contextmanager
from app.config import settings


@contextmanager
def get_connection():
    conn = psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_model_metadata(model_version_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model_path, threshold, feature_columns, training_fraud_rate
            FROM model_versions
            WHERE id = %s;
            """,
            (model_version_id,),
        )
        row = cursor.fetchone()
        cursor.close()

    if row is None:
        raise RuntimeError(f"Model version {model_version_id} not found in DB")

    return {
        "model_path": row[0],
        "threshold": row[1],
        "feature_columns": row[2],
        "training_fraud_rate": row[3],
    }


def log_prediction(model_version_id, prob, prediction, latency_ms):
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (
                    model_version_id,
                    fraud_probability,
                    prediction,
                    latency_ms
                )
                VALUES (%s, %s, %s, %s);
                """,
                (model_version_id, prob, prediction, latency_ms),
            )
            conn.commit()
            cursor.close()
    except Exception:
        pass


def fetch_metrics(model_version_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_version_id = %s;",
            (model_version_id,),
        )
        total_predictions = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE model_version_id = %s AND prediction = 1;
            """,
            (model_version_id,),
        )
        fraud_predictions = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT AVG(latency_ms)
            FROM predictions
            WHERE model_version_id = %s;
            """,
            (model_version_id,),
        )
        avg_latency = cursor.fetchone()[0]

        cursor.close()

    fraud_rate = (
        fraud_predictions / total_predictions
        if total_predictions > 0
        else 0
    )

    return {
        "total_predictions": total_predictions,
        "fraud_predictions": fraud_predictions,
        "fraud_rate": fraud_rate,
        "average_latency_ms": avg_latency,
    }