import requests
import random
import time
import csv

API_URL = "https://fraud-detection-api.duckdns.org/predict"
TOTAL_REQUESTS = 200

results = []

def generate_transaction():
    return {
        "transaction_time_seconds": random.randint(0, 172800),
        "amount": round(random.uniform(1, 2000), 2),
        "v": [round(random.uniform(-3, 3), 4) for _ in range(28)]
    }

for i in range(TOTAL_REQUESTS):

    payload = generate_transaction()

    start = time.time()

    response = requests.post(API_URL, json=payload)

    end = time.time()

    latency = (end - start) * 1000

    if response.status_code == 200:
        data = response.json()

        results.append({
            "request_id": i,
            "latency_ms": latency,
            "fraud_probability": data["fraud_probability"],
            "prediction": data["prediction"]
        })

    else:
        results.append({
            "request_id": i,
            "latency_ms": latency,
            "fraud_probability": None,
            "prediction": "ERROR"
        })

    print(f"Request {i+1}/{TOTAL_REQUESTS} latency: {latency:.2f} ms")

with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["request_id", "latency_ms", "fraud_probability", "prediction"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nBaseline test completed.")
print("Results saved to baseline_results.csv")