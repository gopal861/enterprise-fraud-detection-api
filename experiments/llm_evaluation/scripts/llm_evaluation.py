import pandas as pd
import requests
import time

API_URL = "https://fraud-detection-api.duckdns.org/predict-with-explanation"

df = pd.read_csv("creditcard.csv").sample(50)

results = []

for _, row in df.iterrows():

    payload = {
        "transaction_time_seconds": int(row["Time"]),
        "amount": float(row["Amount"]),
        "v": [row[f"V{i}"] for i in range(1,29)]
    }

    start = time.time()

    r = requests.post(API_URL, json=payload)

    latency = (time.time() - start) * 1000

    if r.status_code == 200:

        data = r.json()

        results.append({
            "latency_ms": latency,
            "prediction": data["prediction"],
            "fraud_probability": data["fraud_probability"],
            "explanation": data["explanation"]
        })

df_results = pd.DataFrame(results)

print("Average LLM latency:", df_results["latency_ms"].mean())

df_results.to_csv("llm_results.csv", index=False)