"""
We want to look at the distribution of our entire training and testing dataset
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy.plotting import wong_colours as colours

sys.path.append("./src")

from load_data import load_reduced_data

training_data, testing_data = load_reduced_data(verbose=False)
all_data = pd.concat([training_data, testing_data])

features = all_data.columns[
    2:-2
]  # Remove start and end times of samples, along with label and boundary id

mapping = {
    "Mean |B|": "Mean |B| [nT]",
    "Mean Bx": "Mean Bx [nT]",
    "Mean By": "Mean By [nT]",
    "Mean Bz": "Mean Bz [nT]",
    "Median |B|": "Median |B| [nT]",
    "Median Bx": "Median Bx [nT]",
    "Median By": "Median By [nT]",
    "Median Bz": "Median Bz [nT]",
    "Standard Deviation |B|": "StD |B| [nT]",
    "Standard Deviation Bx": "StD Bx [nT]",
    "Standard Deviation By": "StD By [nT]",
    "Standard Deviation Bz": "StD Bz [nT]",
    "Heliocentric Distance (AU)": "$R_H$ [AU]",
    "Local Time (hrs)": "LT [hours]",
    "Latitude (deg.)": "Lat. [$^\circ$]",
    "Magnetic Latitude (deg.)": "M.Lat. [$^\circ$]",
    "Mercury Distance (radii)": r"$|\vec{X}_\text{MSM'}|$ [$R_M$]",
    "X MSM' (radii)": r"$X_\text{MSM'}$ [$R_M$]",
    "Y MSM' (radii)": r"$Y_\text{MSM'}$ [$R_M$]",
    "Z MSM' (radii)": r"$Z_\text{MSM'}$ [$R_M$]",
}

abbreviations = [mapping.get(feature, feature) for feature in features]

# For each feature, we want a histogram
fig, axes = plt.subplots(4, 7, figsize=(11.7, 8.3))

# A lot of these histograms will have the same params
hist_params = {"histtype": "step", "alpha": 0.8, "density": True}

for ax, feature, abb in zip(axes.flatten(), features, abbreviations):

    bins = np.linspace(np.min(all_data[feature]), np.max(all_data[feature]), 50)

    ax.hist(
        all_data[feature],
        label="All Reduced Data",
        color="black",
        bins=bins,
        alpha=0.2,
        density=True,
    )
    ax.hist(
        training_data[feature],
        label="Trainig Data",
        color=colours["red"],
        bins=bins,
        **hist_params,
    )
    ax.hist(
        testing_data[feature],
        label="Testing Data",
        color=colours["blue"],
        bins=bins,
        **hist_params,
    )

    ax.set_xlabel(abb)

    ax.margins(0)
    # if ax not in axes[:, 0]:
    #     ax.set_yticks([])

plt.tight_layout()
plt.savefig("./figures/data_distributions.pdf", format="pdf")
