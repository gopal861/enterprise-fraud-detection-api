import pandas as pd
import numpy as np

df = pd.read_csv("dataset_load_test_results.csv")

lat = df["latency_ms"]

print("\nSYSTEM PERFORMANCE METRICS\n")

print("Total Requests:", len(df))

print("\nLatency Metrics (ms)")
print("Average latency:", lat.mean())
print("Median latency:", lat.median())
print("P95 latency:", np.percentile(lat,95))
print("P99 latency:", np.percentile(lat,99))
print("Max latency:", lat.max())