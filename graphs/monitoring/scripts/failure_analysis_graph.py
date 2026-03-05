import matplotlib.pyplot as plt

labels = ["False Positives", "False Negatives"]
values = [35, 2]

plt.bar(labels, values)
plt.title("Fraud Detection Errors")
plt.ylabel("Count")
plt.savefig("fraud_detection_errors.png")