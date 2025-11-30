"""
Script to quickly visualise the model output within the context of the data
"""

import datetime as dt

import hermpy.mag
import hermpy.utils
import matplotlib.pyplot as plt
import pandas as pd
from hermpy.plotting import wong_colours as colours

start = dt.datetime.strptime("2011-04-07", "%Y-%m-%d")
end = dt.datetime.strptime("2011-04-08", "%Y-%m-%d")

# Load MAG data
data = hermpy.mag.Load_Between_Dates(
    hermpy.utils.User.DATA_DIRECTORIES["MAG"], start, end
)

# Load model output
model_output = pd.read_csv("./data/raw_model_output.csv")
model_output["Time"] = pd.to_datetime(model_output["Time"], format="ISO8601")

# Shorten to time we want
model_output = model_output.loc[model_output["Time"].between(start, end)]


fig, axes = plt.subplots(2, 1, sharex=True)
mag_ax, probabilities_ax = axes

mag_colours = [
    colours["black"],
    colours["red"],
    colours["green"],
    colours["blue"],
]
for i, component in enumerate(["|B|", "Bx'", "By'", "Bz'"]):
    mag_ax.plot(data["date"], data[component], color=mag_colours[i], label=component)

probability_colours = [
    colours["orange"],
    colours["light blue"],
    colours["yellow"],
]
for i, label in enumerate(model_output.columns[1:]):
    probabilities_ax.plot(
        model_output["Time"],
        model_output[label],
        color=probability_colours[i],
        label=label,
    )

probabilities_ax.legend()

plt.show()
