"""
Using the utilities created in src/apply_model.py, apply the model to all crossings, and save the data.
"""

import datetime as dt
import pickle

import hermpy.boundaries
import hermpy.utils
import numpy as np
import pandas as pd
import sklearn.ensemble
from tqdm import tqdm

from apply_model import get_magnetospheric_region

# Application parameters, these match what was used in the training process.
data_window_size = 10  # seconds
step_size = 1  # seconds, the frequency at which to make classifications

# How far from either crossing should we apply to. This was chosen to match
# the selection region for training data.
interval_time_buffer = dt.timedelta(minutes=10)


def main():
    # Set up data directories
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = "./data/messenger/full_cadence"
    hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"

    # Load model
    with open("./data/model/messenger_region_classifier.pkl", "rb") as f:
        model: sklearn.ensemble.RandomForestClassifier = pickle.load(f)

    # Load crossing intervals
    crossing_intervals = hermpy.boundaries.Load_Crossings(
        "./data/philpott_2020_crossing_list.xlsx", include_data_gaps=True
    )

    # For now we can just test for a handful of intervals
    crossing_intervals = crossing_intervals.iloc[:100]

    # To ensure no overlap in application to a given crossing interval, we want to
    # classify pairs of crossing intervals as one. i.e. BS_IN and MP_IN, as well as
    # MP_OUT and BS_OUT However, there are sometimes missing crossings, so we need
    # need to be careful. Based on the geometry of the orbit and the physics of the
    # system, we never expect to see MP_IN closely followed by any BS crossing.
    # Similarly, we never expect to see BS_OUT closely followed by any MP crossing.
    # If we do, it means we can't treat them as a pair, and must instead move
    # individually.
    crossing_interval_groups = pair_crossing_intervals(crossing_intervals)

    results = []
    for group in tqdm(
        crossing_interval_groups,
        desc="Applying model to crossing intervals",
        dynamic_ncols=True,
        smoothing=0,
    ):
        results.append(get_probabilities_for_group(group))

    times, probabilities = zip(*results)  # Unpack results

    # Convert lists of times and probabilities into arrays
    times = np.concatenate(times)
    probabilities = np.vstack(probabilities)

    data_to_save = {
        "Time": times,
        "P(Solar Wind)": probabilities[:, 0],
        "P(Magnetosheath)": probabilities[:, 1],
        "P(Magnetosphere)": probabilities[:, 2],
    }

    pd.DataFrame(data_to_save).to_csv("./data/raw_model_output.csv", index=False)


def pair_crossing_intervals(crossing_intervals):
    """
    Returns a list of lists (length 1 or 2)
    """
    crossing_groups = []

    crossing_index = 0
    while crossing_index < len(crossing_intervals) - 1:

        current_crossing = crossing_intervals.loc[crossing_index]
        next_crossing = crossing_intervals.loc[crossing_index + 1]

        if current_crossing["Type"] == "BS_IN":
            # We expect a magnetopause in crossing next
            match next_crossing["Type"]:
                case "MP_IN":
                    # This is as normal, we can add to our list of pairs
                    crossing_groups.append([current_crossing, next_crossing])

                    # We don't want to consider the next crossing as we have already
                    # saved it, so we add an extra to the crossing index.
                    crossing_index += 1

                case _:
                    # This is abnormal, we just want to look around the current crossing
                    crossing_groups.append([current_crossing])

        elif current_crossing["Type"] == "MP_OUT":
            # We expect a bow shock in crossing next
            match next_crossing["Type"]:
                case "BS_OUT":
                    # This is as normal, we can add to our list of pairs
                    crossing_groups.append([current_crossing, next_crossing])

                    # We don't want to consider the next crossing as we have already
                    # saved it, so we add an extra to the crossing index.
                    crossing_index += 1

                case _:
                    # This is abnormal, we just want to look around the current crossing
                    crossing_groups.append([current_crossing])

        else:
            # Otherwise, if for some reason the previous part of the crossing pair
            # didn't exist. We save this crossing on its own.

            # Ignore data gaps in this search
            if current_crossing["Type"] != "DATA_GAP":
                crossing_groups.append([current_crossing])

        crossing_index += 1

    return crossing_groups


def get_probabilities_for_group(crossing_interval_group):

    # These need to be set in here too so that they apply to each individual
    # worker instance.
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = "./data/messenger/full_cadence"
    hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"

    # Check if crossing group is a pair or individual
    group_is_pair = len(crossing_interval_group) != 1

    if group_is_pair:
        data_start_time = (
            crossing_interval_group[0]["Start Time"] - interval_time_buffer
        )
        data_end_time = crossing_interval_group[1]["End Time"] + interval_time_buffer

    else:
        data_start_time = (
            crossing_interval_group[0]["Start Time"] - interval_time_buffer
        )
        data_end_time = crossing_interval_group[0]["End Time"] + interval_time_buffer

    return get_magnetospheric_region(
        data_start_time,
        data_end_time,
        model_path="./data/model/messenger_region_classifier.pkl",
    )


if __name__ == "__main__":
    main()
