import numpy as np
import pandas as pd

performance_metrics = pd.read_csv("./data/model/performance_metrics.csv")

mean_accuracy = np.mean(performance_metrics["Training Accuracy"])
std_accuracy = np.std(performance_metrics["Training Accuracy"])

print("Training")
print(mean_accuracy)
print(std_accuracy)

mean_accuracy = np.mean(performance_metrics["OOB Score"])
std_accuracy = np.std(performance_metrics["OOB Score"])

print("OOB Score")
print(mean_accuracy)
print(std_accuracy)

mean_accuracy = np.mean(performance_metrics["Testing Accuracy"])
std_accuracy = np.std(performance_metrics["Testing Accuracy"])

print("Testing")
print(mean_accuracy)
print(std_accuracy)
