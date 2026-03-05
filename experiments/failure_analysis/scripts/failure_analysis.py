import pandas as pd

df = pd.read_csv("dataset_load_test_results.csv")

false_positives = df[(df["true_label"] == 0) & (df["prediction"] == 1)]
false_negatives = df[(df["true_label"] == 1) & (df["prediction"] == 0)]

print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

false_positives.to_csv("false_positives.csv", index=False)
false_negatives.to_csv("false_negatives.csv", index=False)