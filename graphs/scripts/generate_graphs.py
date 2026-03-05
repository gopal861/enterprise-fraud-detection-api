import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

df = pd.read_csv("dataset_load_test_results.csv")

# ------------------------------
# 1 Latency Distribution
# ------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["latency_ms"], bins=50)
plt.title("API Latency Distribution")
plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.savefig("latency_distribution.png")
plt.close()

# ------------------------------
# 2 Latency Percentiles
# ------------------------------

lat = df["latency_ms"]

percentiles = [
np.percentile(lat,50),
np.percentile(lat,95),
np.percentile(lat,99)
]

labels = ["P50","P95","P99"]

plt.figure(figsize=(6,4))
plt.bar(labels,percentiles)
plt.title("Latency Percentiles")
plt.ylabel("Latency (ms)")
plt.savefig("latency_percentiles.png")
plt.close()

# ------------------------------
# 3 Prediction Distribution
# ------------------------------

counts = df["prediction"].value_counts()

plt.figure(figsize=(6,4))
counts.plot(kind="bar")
plt.title("Prediction Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("prediction_distribution.png")
plt.close()

# ------------------------------
# 4 Confusion Matrix
# ------------------------------

cm = confusion_matrix(df["true_label"],df["prediction"])

plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

print("Graphs generated successfully.")