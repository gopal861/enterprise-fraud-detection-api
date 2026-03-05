import pandas as pd
import requests
import time
import csv
import threading
from queue import Queue

API_URL = "https://fraud-detection-api.duckdns.org/predict"

TOTAL_REQUESTS = 5000
CONCURRENT_WORKERS = 10

df = pd.read_csv("creditcard.csv")

# Sample rows for experiment
df = df.sample(n=TOTAL_REQUESTS, random_state=42)

results = []
lock = threading.Lock()


def build_payload(row):

    v_features = [row[f"V{i}"] for i in range(1, 29)]

    payload = {
        "transaction_time_seconds": int(row["Time"]),
        "amount": float(row["Amount"]),
        "v": v_features
    }

    return payload


def worker(queue):

    while not queue.empty():

        idx, row = queue.get()

        payload = build_payload(row)

        start = time.time()

        try:

            response = requests.post(API_URL, json=payload, timeout=10)

            latency = (time.time() - start) * 1000

            if response.status_code == 200:

                data = response.json()

                record = {
                    "request_id": idx,
                    "latency_ms": latency,
                    "fraud_probability": data["fraud_probability"],
                    "prediction": data["prediction"],
                    "true_label": int(row["Class"])
                }

            else:

                record = {
                    "request_id": idx,
                    "latency_ms": latency,
                    "fraud_probability": None,
                    "prediction": "ERROR",
                    "true_label": int(row["Class"])
                }

        except Exception:

            latency = (time.time() - start) * 1000

            record = {
                "request_id": idx,
                "latency_ms": latency,
                "fraud_probability": None,
                "prediction": "FAILED",
                "true_label": int(row["Class"])
            }

        with lock:
            results.append(record)

        queue.task_done()


queue = Queue()

for i, row in df.iterrows():
    queue.put((i, row))


threads = []

start_time = time.time()

for _ in range(CONCURRENT_WORKERS):
    t = threading.Thread(target=worker, args=(queue,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

end_time = time.time()

duration = end_time - start_time

print(f"\nTotal runtime: {duration:.2f} seconds")

rps = TOTAL_REQUESTS / duration

print(f"Throughput: {rps:.2f} requests/sec")


with open("dataset_load_test_results.csv", "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "request_id",
            "latency_ms",
            "fraud_probability",
            "prediction",
            "true_label"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("Results saved to dataset_load_test_results.csv")