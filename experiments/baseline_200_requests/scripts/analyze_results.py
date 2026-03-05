import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("baseline_results.csv")

print("\nBasic Statistics\n")
print(df.describe())

# -------- Graph 1: Latency per request --------
plt.figure(figsize=(10,5))
plt.plot(df["request_id"], df["latency_ms"])
plt.xlabel("Request Number")
plt.ylabel("Latency (ms)")
plt.title("API Latency per Request")
plt.grid(True)
plt.savefig("latency_per_request.png")

# -------- Graph 2: Latency Distribution --------
plt.figure(figsize=(8,5))
plt.hist(df["latency_ms"], bins=30)
plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.title("Latency Distribution")
plt.savefig("latency_distribution.png")

# -------- Graph 3: Fraud Probability Distribution --------
plt.figure(figsize=(8,5))
plt.hist(df["fraud_probability"], bins=30)
plt.xlabel("Fraud Probability")
plt.ylabel("Frequency")
plt.title("Fraud Probability Distribution")
plt.savefig("fraud_probability_distribution.png")

# -------- Graph 4: Prediction Class Distribution --------
prediction_counts = df["prediction"].value_counts()

plt.figure(figsize=(6,6))
prediction_counts.plot(kind="bar")
plt.xlabel("Prediction Class")
plt.ylabel("Count")
plt.title("Fraud vs Legitimate Predictions")
plt.savefig("prediction_distribution.png")

print("\nGraphs saved:")
print("latency_per_request.png")
print("latency_distribution.png")
print("fraud_probability_distribution.png")
print("prediction_distribution.png")