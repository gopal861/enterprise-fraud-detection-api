import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset_load_test_results.csv")

y_true = df["true_label"]
y_pred = df["prediction"]

cm = confusion_matrix(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

report = f"""
MODEL EVALUATION REPORT
=======================

Dataset size: {len(df)}

Confusion Matrix:
{cm}

Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1}
"""

print(report)

with open("model_metrics.txt","w") as f:
    f.write(report)