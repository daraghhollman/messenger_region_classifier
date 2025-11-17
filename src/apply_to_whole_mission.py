"""
Apply the trained model to all crossings efficiently using multiprocessing.
"""

import datetime as dt
import pickle
import sys
import warnings
from multiprocessing import Pool, cpu_count

import hermpy.boundaries
import hermpy.utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from apply_model import get_magnetospheric_region

data_window_size = 10  # seconds
step_size = 1  # seconds
interval_time_buffer = dt.timedelta(minutes=10)
model_path = "./data/model/messenger_region_classifier.pkl"
data_dir = "./data/messenger/full_cadence"
crossing_intervals_path = "./data/philpott_2020_crossing_list.xlsx"
output_path = "./data/raw_model_output.csv"

_model = None

warnings.filterwarnings(
    "ignore",
    message=".*setting n_jobs=1.*",
    category=UserWarning,
)


def main():
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = max(1, cpu_count() // 2)
    print(f"Using {n_jobs} worker processes")

    # Global hermpy setup (needed before load)
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = data_dir
    hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"

    # Load crossings
    print("Loading crossing intervals...")
    crossings = hermpy.boundaries.Load_Crossings(
        crossing_intervals_path, include_data_gaps=True
    )

    groups = pair_crossing_intervals(crossings)
    print(f"Processing {len(groups)} crossing groups...")

    # Parallel execution using imap_unordered
    classification_times = []
    region_probabilities = []

    with Pool(processes=n_jobs) as pool:
        for times, probs in tqdm(
            pool.imap_unordered(get_probabilities_for_group, groups),
            total=len(groups),
            desc="Applying model",
            dynamic_ncols=True,
        ):
            classification_times.append(times)
            region_probabilities.append(probs)

    # Concatenate results and sort
    classification_times = np.concatenate(classification_times)
    region_probabilities = np.vstack(region_probabilities)
    order = np.argsort(classification_times)
    classification_times = classification_times[order]
    region_probabilities = region_probabilities[order]

    df = pd.DataFrame(
        {
            "Time": classification_times,
            # When we created the model, the classes were sorted alphabetically
            "P(Solar Wind)": region_probabilities[:, 2],
            "P(Magnetosheath)": region_probabilities[:, 0],
            "P(Magnetosphere)": region_probabilities[:, 1],
        }
    )
    df.to_csv(output_path, index=False)

    print(f"Saved model output to {output_path}")


def get_model():
    global _model
    if _model is None:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _model


def pair_crossing_intervals(crossing_intervals):
    """
    Returns a list of lists (each length 1 or 2)
    """
    groups = []
    i = 0
    while i < len(crossing_intervals) - 1:
        current = crossing_intervals.loc[i]
        nxt = crossing_intervals.loc[i + 1]

        if current["Type"] == "BS_IN":
            if nxt["Type"] == "MP_IN":
                groups.append([current, nxt])
                i += 1
            else:
                groups.append([current])
        elif current["Type"] == "MP_OUT":
            if nxt["Type"] == "BS_OUT":
                groups.append([current, nxt])
                i += 1
            else:
                groups.append([current])
        else:
            if current["Type"] != "DATA_GAP":
                groups.append([current])
        i += 1
    return groups


def get_probabilities_for_group(group):
    """
    Processes one crossing interval group.
    Loads HermPy context + model per worker (cached),
    then runs get_magnetospheric_region().
    Returns (times, probs) for this group.
    """
    # Reinitialize hermpy context per process
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = data_dir
    hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"
    _ = get_model()  # ensure model loaded in this process

    # Check if group is pair
    if len(group) != 1:
        data_start = group[0]["Start Time"] - interval_time_buffer
        data_end = group[1]["End Time"] + interval_time_buffer
    else:
        data_start = group[0]["Start Time"] - interval_time_buffer
        data_end = group[0]["End Time"] + interval_time_buffer

    return get_magnetospheric_region(data_start, data_end)


if __name__ == "__main__":
    main()
