"""
We call a single region classification as the length of continuous 1-sec
classifications. We want to find a relation between the confidence in that
classification and the duration to determine some regions as unknown.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

# First we load the raw model ouput
raw_model_output = pd.read_csv("./data/raw_model_output.csv")
raw_model_output["Time"] = pd.to_datetime(raw_model_output["Time"], format="ISO8601")
probabilities = raw_model_output[
    ["P(Solar Wind)", "P(Magnetosheath)", "P(Magnetosphere)"]
].to_numpy()

# We need to find the change points to determine where regions occur, so that
# we can break this up into individual regions.
region_classification = np.argmax(probabilities, axis=1)

# We also need to find the ratio of probabilities to determine confidence later
sorted_probabilities = np.sort(probabilities, axis=1)
largest_probabilities = sorted_probabilities[:, -1]
second_largest_probabities = sorted_probabilities[:, -2]

probability_ratio = second_largest_probabities / largest_probabilities

region_mapping = np.array(
    [
        "Solar Wind",
        "Magnetosheath",
        "Magnetosphere",
    ]
)
region_labels = region_mapping[region_classification]


region_change_indices = np.where(
    region_classification[:-1] != region_classification[1:]
)[0]

# We add one to this region change index so that each index marks the start of
# a new region
region_change_indices += 1

regions: list[dict] = []
for i in tqdm(
    range(len(region_change_indices) - 1),
    desc="Finding continous regions",
    total=len(region_change_indices) - 1,
):

    current_region_index = region_change_indices[i]
    next_region_index = region_change_indices[i + 1]

    regions.append(
        {
            "Label": region_labels[current_region_index],
            "Duration": (
                raw_model_output["Time"].iloc[next_region_index]
                - raw_model_output["Time"].iloc[current_region_index]
            ).total_seconds(),
            "Confidence": 1
            - np.median(probability_ratio[current_region_index:next_region_index]),
        }
    )

pd.DataFrame(regions).to_csv("./data/postprocessing/continous_regions.csv", index=False)
