import numpy as np
import pandas as pd

raw_model_output = pd.read_csv("./data/raw_model_output.csv")
raw_model_output["Time"] = pd.to_datetime(raw_model_output["Time"], format="ISO8601")

probability_columns = ["P(Solar Wind)", "P(Magnetosheath)", "P(Magnetosphere)"]

# Each row contains a Time value, and 3 probability values. There are sometimes time jumps, which we want to ignore.
# We can find the time difference in the following way:
raw_model_output["Time Difference"] = raw_model_output["Time"].diff()

# Define a maximum time gap to split into multiple regions
max_gap = pd.Timedelta("5 seconds")

raw_model_output["Label"] = (
    raw_model_output[probability_columns].idxmax(axis=1).str[2:-1]
)

probabilities = raw_model_output[probability_columns].to_numpy()
sorted_probabilities = -np.sort(-probabilities, axis=1)[:, :2]
raw_model_output["Probability Ratio"] = (
    sorted_probabilities[:, 1] / sorted_probabilities[:, 0]
)

# Changepoints - This is a boolean column
raw_model_output["Changepoint"] = (
    raw_model_output["Label"] != raw_model_output["Label"].shift()
) | (raw_model_output["Time Difference"] > max_gap)

# Define a region id as the cumulative sum of the changepoints
raw_model_output["Region ID"] = raw_model_output["Changepoint"].cumsum()

regions = raw_model_output.groupby("Region ID").agg(
    **{
        "Start Time": ("Time", "min"),
        "End Time": ("Time", "max"),
        "Label": ("Label", "first"),
        "Mean Probability Ratio": ("Probability Ratio", "mean"),
    }
)

regions["Duration"] = (regions["End Time"] - regions["Start Time"]).dt.total_seconds()
regions["Confidence"] = 1 - regions["Mean Probability Ratio"]

regions.to_csv("./data/postprocessing/continous_regions.csv", index=False)
