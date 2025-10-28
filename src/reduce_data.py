"""
This script takes an input crossing intervals list (Philpott+ 2020), along with
MESSENGER MAG data, and draws samples of solar wind, magnetosheath, and
magnetosphere at random from a region surrounding (but not including) these
intervals.
"""

import csv
import datetime as dt
import multiprocessing
import os
import pathlib
import random
import shutil
import tempfile

import hermpy.boundaries
import hermpy.mag
import hermpy.trajectory
import hermpy.utils
import numpy as np
import pandas as pd
import scipy
import spiceypy as spice
from tqdm import tqdm

# GLOBALS

# Load boundary crossings
crossing_intervals = hermpy.boundaries.Load_Crossings(
    str(pathlib.Path(__file__).parent.parent / "data/philpott_2020_crossing_list.xlsx"),
    include_data_gaps=True,
    # Data gaps are included for indexing continuity but ignored in processing.
)

# Define some parameters about how the samples are drawn
sample_length = dt.timedelta(
    seconds=10
)  # Each sample describes a window of time surrounding a central point. How long should this window be?
search_distance = dt.timedelta(
    minutes=10
)  # How far from the boundary to draw samples from


def main():

    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = str(
        pathlib.Path(__file__).parent.parent / "data" / "messenger" / "full_cadence"
    )
    hermpy.utils.User.METAKERNEL = str(
        pathlib.Path(__file__).parent.parent
        / "SPICE"
        / "messenger"
        / "metakernel_messenger.txt"
    )

    # Define output files
    solar_wind_samples_path = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / "reduced"
        / "solar_wind_samples.csv"
    ).resolve()
    bow_shock_magnetosheath_samples_path = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / "reduced"
        / "bs_magnetosheath_samples.csv"
    ).resolve()
    magnetopause_magnetosheath_samples_path = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / "reduced"
        / "mp_magnetosheath_samples.csv"
    ).resolve()
    magenetosphere_samples_path = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / "reduced"
        / "magnetosphere_samples.csv"
    ).resolve()

    output_files = [
        solar_wind_samples_path,
        bow_shock_magnetosheath_samples_path,
        magnetopause_magnetosheath_samples_path,
        magenetosphere_samples_path,
    ]

    # Instatiate files
    for file in output_files:
        if not os.path.exists(file):
            os.mknod(file)

    # Loop through the crossing intervals and sample around them.
    process_items = [
        (i, crossing_interval) for i, crossing_interval in crossing_intervals.iterrows()
    ]

    sample_buffers = {
        "Solar Wind": [],
        "BS Magnetosheath": [],
        "MP Magnetosheath": [],
        "Magnetosphere": [],
    }

    with multiprocessing.Pool(
        int(input(f"Number of cores? __ / {multiprocessing.cpu_count()} "))
    ) as pool:

        for samples_taken in tqdm(
            pool.imap(process_crossing_interval, process_items),
            desc="Processing crossings",
            total=len(process_items),
        ):
            if not isinstance(samples_taken, list):
                continue

            for row in samples_taken:
                for sample in [row[0], row[1]]:
                    label = sample["Label"]

                    if label not in sample_buffers:
                        raise ValueError(f"Unknown sample label: {label}")

                    sample_buffers[label].append(list(sample.values()))

        output_paths = {
            "Solar Wind": solar_wind_samples_path,
            "BS Magnetosheath": bow_shock_magnetosheath_samples_path,
            "MP Magnetosheath": magnetopause_magnetosheath_samples_path,
            "Magnetosphere": magenetosphere_samples_path,
        }

        for label, rows in sample_buffers.items():

            output_file = output_paths[label]
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)


def get_random_sample(
    data,
    search_start,
    search_end,
    length,
    boundary_type,
    sample_location,
):

    max_attempts = 10
    attempt = 0
    sample_data = None
    while attempt < max_attempts:

        # Choose random start point of sample
        sample_start = search_start + dt.timedelta(
            seconds=random.randint(0, int((search_end - search_start).total_seconds()))
        )
        sample_end = sample_start + length

        # Index data
        sample_data = data.loc[data["date"].between(sample_start, sample_end)]

        # Test to ensure there is data
        if not sample_data.empty:
            break

        # If not, we try again with a max number of attempts
        attempt += 1

    # If all attempts fail, raise an error
    if sample_data is None or sample_data.empty:
        raise ValueError(f"No non-empty sample found after {max_attempts} attempts.")

    # Label the sample based on the type of boundary crossing interval
    labels = {
        ("BS_IN", "before"): "Solar Wind",
        ("BS_IN", "after"): "BS Magnetosheath",
        ("BS_OUT", "before"): "BS Magnetosheath",
        ("BS_OUT", "after"): "Solar Wind",
        ("MP_IN", "before"): "MP Magnetosheath",
        ("MP_IN", "after"): "Magnetosphere",
        ("MP_OUT", "before"): "Magnetosphere",
        ("MP_OUT", "after"): "MP Magnetosheath",
    }

    sample_label = labels.get((boundary_type, sample_location), "ERROR")

    # Get features
    sample_features = get_sample_features(sample_data)
    sample_features["Label"] = sample_label

    return sample_features


def get_sample_features(data):

    components = ["|B|", "Bx", "By", "Bz"]

    # Compute per-component stats as dicts
    mean = {f"Mean {c}": np.mean(data[c]) for c in components}
    median = {f"Median {c}": np.median(data[c]) for c in components}
    std = {f"Standard Deviation {c}": np.std(data[c]) for c in components}
    skew = {f"Skew {c}": scipy.stats.skew(data[c]) for c in components}
    kurtosis = {f"Kurtosis {c}": scipy.stats.kurtosis(data[c]) for c in components}

    data_middle = data.iloc[round(len(data) / 2)]

    sample_middle_position = [
        data_middle["X MSM' (radii)"],
        data_middle["Y MSM' (radii)"],
        data_middle["Z MSM' (radii)"],
    ]

    planetary_distance = np.linalg.norm(sample_middle_position)
    local_time = hermpy.trajectory.Local_Time(sample_middle_position)
    latitude = hermpy.trajectory.Latitude(sample_middle_position)
    magnetic_latitude = hermpy.trajectory.Magnetic_Latitude(sample_middle_position)

    with spice.KernelPool(hermpy.utils.User.METAKERNEL):
        et = spice.str2et(data_middle["date"].strftime("%Y-%m-%d %H:%M:%S"))
        mercury_position, _ = spice.spkpos("MERCURY", et, "J2000", "NONE", "SUN")

        # It would be quicker to calculate the heliocentric distance for all
        # samples at the end
        heliocentric_distance = np.sqrt(
            mercury_position[0] ** 2
            + mercury_position[1] ** 2
            + mercury_position[2] ** 2
        )
        heliocentric_distance = hermpy.utils.Constants.KM_TO_AU(heliocentric_distance)

    return {
        # Time identifiers
        "Sample Start": data["date"].iloc[0],
        "Sample End": data["date"].iloc[-1],
        # Magnetic field statistics
        **mean,
        **median,
        **std,
        **skew,
        **kurtosis,
        # Ephemeris
        "Heliocentric Distance (AU)": heliocentric_distance,
        "Local Time (hrs)": local_time,
        "Latitude (deg.)": latitude,
        "Magnetic Latitude (deg.)": magnetic_latitude,
        "Mercury Distance (radii)": planetary_distance,
        "X MSM' (radii)": data_middle["X MSM' (radii)"],
        "Y MSM' (radii)": data_middle["Y MSM' (radii)"],
        "Z MSM' (radii)": data_middle["Z MSM' (radii)"],
    }


def process_crossing_interval(inputs):
    i, crossing_interval = inputs

    if crossing_interval["Type"] == "DATA_GAP":
        return None  # Ignore labelled data gaps

    # We define the eariest possible sample start
    # and latest possible sample end.
    # Making sure to never go past the next boundary.
    earliest_sample_start_before = crossing_interval["Start Time"] - search_distance
    latest_sample_start_before = crossing_interval["Start Time"] - sample_length

    if i > 0:
        if earliest_sample_start_before < crossing_intervals.loc[i - 1]["End Time"]:
            earliest_sample_start_before = crossing_intervals.loc[i - 1]["End Time"]

    earliest_sample_start_after = crossing_interval["End Time"]
    latest_sample_start_after = (
        crossing_interval["End Time"] + search_distance - sample_length
    )

    if i > 0:
        if latest_sample_start_after > crossing_intervals.loc[i + 1]["Start Time"]:
            latest_sample_start_after = (
                crossing_intervals.loc[i + 1]["Start Time"] - sample_length
            )

    if earliest_sample_start_before > latest_sample_start_before:
        raise ValueError(
            f"Sample start ({earliest_sample_start_before}) is after sample end ({latest_sample_start_before})!"
        )

    if earliest_sample_start_after > latest_sample_start_after:
        raise ValueError(
            f"Sample start ({earliest_sample_start_after}) is after sample end ({latest_sample_start_after})!"
        )

    # Load data
    surrounding_data = hermpy.mag.Load_Between_Dates(
        hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"],
        earliest_sample_start_before,
        latest_sample_start_after,
        average=None,
        no_dirs=True,
    )

    samples = []
    number_of_samples = 10
    for _ in range(number_of_samples):
        # Save sample before BCI
        sample_before = get_random_sample(
            surrounding_data,
            earliest_sample_start_before,
            latest_sample_start_before,
            sample_length,
            str(crossing_interval["Type"]),
            "before",
        )

        # Save sample after
        sample_after = get_random_sample(
            surrounding_data,
            earliest_sample_start_after,
            latest_sample_start_after,
            sample_length,
            str(crossing_interval["Type"]),
            "after",
        )

        for sample in [sample_before, sample_after]:
            sample["Boundary ID"] = i

        samples.append([sample_before, sample_after])

    return samples


if __name__ == "__main__":
    main()
