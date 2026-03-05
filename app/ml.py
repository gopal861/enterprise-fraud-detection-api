import joblib
import math
import numpy as np
from app.config import settings
from app.db import fetch_model_metadata


class FraudModelService:
    def __init__(self):
        self.model_version_id = settings.model_version_id

        # Fetch metadata from DB
        metadata = fetch_model_metadata(self.model_version_id)

        self.model_path = metadata["model_path"]
        self.db_threshold = metadata["threshold"]
        self.db_feature_columns = metadata["feature_columns"].split(",")
        self.training_fraud_rate = metadata["training_fraud_rate"]

        # Load serialized model
        model_bundle = joblib.load(self.model_path)

        self.pipeline = model_bundle["pipeline"]
        self.serialized_threshold = model_bundle["threshold"]
        self.serialized_features = model_bundle["features"]

        # Integrity checks
        self._validate_model_integrity()

    def _validate_model_integrity(self):
        if float(self.db_threshold) != float(self.serialized_threshold):
            raise RuntimeError("Threshold mismatch between DB and model file")

        if self.db_feature_columns != self.serialized_features:
            raise RuntimeError("Feature list mismatch between DB and model file")

    def engineer_features(self, transaction_time_seconds: int, amount: float, v: list):
        hour = (transaction_time_seconds // 3600) % 24
        log_amount = 0 if amount == 0 else math.log(amount)

        features = [amount, hour, log_amount] + v

        if len(features) != len(self.serialized_features):
            raise RuntimeError("Feature length mismatch during inference")

        return np.array(features).reshape(1, -1)

    def predict(self, transaction_time_seconds: int, amount: float, v: list):
        features = self.engineer_features(transaction_time_seconds, amount, v)

        prob = float(self.pipeline.predict_proba(features)[0][1])
        prediction = int(prob >= self.serialized_threshold)

        return prob, prediction