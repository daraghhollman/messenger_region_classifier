"""
Get statistics for unknown regions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

regions = pd.read_csv("./data/postprocessing/regions_with_unknowns.csv")

# Keep only unknowns
unknowns = regions.loc[regions["Label"] == "Unknown"].copy()

# Identify breaks: whenever the previous row is not also Unknown, start a new group
unknowns["group"] = (unknowns.index.to_series().diff() != 1).cumsum()

# Sum durations within each consecutive group
combined = unknowns.groupby("group")["Duration (seconds)"].sum()

print(combined.quantile([0.25, 0.5, 0.75, 0.95]))

fig, axes = plt.subplots(1, 2)
duration_ax, confidence_ax = axes

duration_ax.hist(regions["Duration (seconds)"])
confidence_ax.hist(regions["Confidence"])

duration_ax.set_xlabel("Duration [seconds]")
confidence_ax.set_xlabel("Confidence [arb.]")

duration_stats = {
    "Mean": np.mean(regions["Duration (seconds)"]),
    "Median": np.median(regions["Duration (seconds)"]),
    "SD": np.std(regions["Duration (seconds)"]),
}
confidence_stats = {
    "Mean": np.mean(regions["Confidence"]),
    "Median": np.median(regions["Confidence"]),
    "SD": np.std(regions["Confidence"]),
}

print(duration_stats)
print(confidence_stats)

plt.show()
