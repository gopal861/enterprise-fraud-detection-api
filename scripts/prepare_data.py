import os
import psycopg2
import pandas as pd
import math
import joblib
from datetime import datetime

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load environment variables from .env
load_dotenv()


DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


MODEL_OUTPUT_DIR = r"C:\Users\IP\Desktop\fraud_project\models"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_transactions() -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM transactions;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = (df["transaction_time_seconds"] // 3600) % 24

    df["log_amount"] = df["amount"].apply(
        lambda x: 0 if x == 0 else math.log(x)
    )

    return df


def train_and_save_model(df: pd.DataFrame):

    feature_columns = (
        ["amount", "hour", "log_amount"]
        + [f"v{i}" for i in range(1, 29)]
    )

    X = df[feature_columns]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_probs = pipeline.predict_proba(X_test)[:, 1]

    threshold = 0.8
    y_pred = (y_probs >= threshold).astype(int)

    print("Threshold Used:", threshold)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_probs))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = f"{MODEL_OUTPUT_DIR}\\fraud_model_{timestamp}.pkl"

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "features": feature_columns,
        },
        model_path,
    )

    print("\nModel saved at:", model_path)


def main():

    df = load_transactions()

    df = engineer_features(df)

    train_and_save_model(df)


if __name__ == "__main__":
    main()