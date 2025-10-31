"""
Provides a function to load the model and data for a given interval of time,
determine features for that data, and return model output and class
probabilities for those times.
"""

import datetime as dt
import multiprocessing
import pathlib
import pickle

import hermpy.mag
import hermpy.utils
import numpy as np
import pandas as pd
import sklearn.ensemble
from tqdm import tqdm

from reduce_data import get_sample_features


def main():
    """
    An example of how you can setup data directories and call get_magnetospheric_region() for any time range
    """

    # Set up data directories
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = str(
        pathlib.Path(__file__).parent.parent / "data" / "messenger" / "full_cadence"
    )
    hermpy.utils.User.METAKERNEL = str(
        pathlib.Path(__file__).parent.parent
        / "SPICE"
        / "messenger"
        / "metakernel_messenger.txt"
    )

    start_time = dt.datetime(2013, 1, 1)
    end_time = dt.datetime(2013, 1, 1, 1)

    probabilities = get_magnetospheric_region(start_time, end_time)

    print(probabilities)


def get_magnetospheric_region(
    start_time: dt.datetime,
    end_time: dt.datetime,
    model_path: str = "./data/model/messenger_region_classifier.pkl",
):

    # Load model
    with open(model_path, "rb") as f:
        model: sklearn.ensemble.RandomForestClassifier = pickle.load(f)

    data_samples, is_data_missing = reduce_data(start_time, end_time)

    # Initialise array of probabilities as nan
    class_probabilities = np.full((len(is_data_missing), 3), np.nan)
    class_probabilities[~is_data_missing, :] = model.predict_proba(
        pd.DataFrame(data_samples)[model.feature_names_in_]
    )

    return class_probabilities


def reduce_data(start_time: dt.datetime, end_time: dt.datetime):

    # Load data within range
    data = hermpy.mag.Load_Between_Dates(
        hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"],
        start_time,
        end_time,
        average=None,
        no_dirs=True,
    )

    # Determine time windows for each sample to be classified
    window_size = dt.timedelta(seconds=10)
    classification_frequency = 1  # seconds

    # We determine features primarily based on the distribution of data in
    # these windows
    time_windows: list[tuple[dt.datetime, dt.datetime]] = [
        (window_start, window_start + window_size)
        for window_start in pd.date_range(
            start=start_time,
            end=end_time - window_size,  # Windows can't go outside of the data
            freq=f"{classification_frequency}s",
        )
    ]

    samples: list[dict] = []
    is_data_missing: list[bool] = []

    def init_worker(data):
        global global_data
        global_data = data

    # Iterrate through each time window
    with multiprocessing.Pool(initializer=init_worker, initargs=(data,)) as pool:

        for window_features in tqdm(
            pool.imap(get_window_features, time_windows),
            total=len(time_windows),
            desc="Calculating features",
        ):

            # We need to be able to handle missing data
            if window_features is not None:
                samples.append(window_features)
                is_data_missing.append(False)
            else:
                is_data_missing.append(True)

    # Check that we have any samples for this time
    if len(samples) == 0:
        raise ValueError("No data for provided time range")

    return samples, np.array(is_data_missing)


def get_window_features(time_windows):

    data = global_data

    # For each time window, get the features of the data within
    window_data = data.loc[data["date"].between(*time_windows)]

    return get_sample_features(window_data)


if __name__ == "__main__":
    main()
