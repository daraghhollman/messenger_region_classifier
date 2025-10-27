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
    include_data_gaps=False,
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

    # If some data is there already, attept to restart
    try:
        previous_indices = [
            pd.read_csv(file).iloc[-1]["Boundary ID"] for file in output_files
        ]
        last_index = np.max(previous_indices) + 1

    except pd.errors.EmptyDataError:
        last_index = -1

    # Loop through the crossing intervals and sample around them.
    process_items = [
        (i, crossing_interval) for i, crossing_interval in crossing_intervals.iterrows()
    ]

    print(f"Continuing from crossing id: {last_index + 1}")

    process_items = process_items[last_index + 1 :]

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

            # Save samples
            for row in samples_taken:
                for sample in [row[0], row[1]]:

                    match sample["Label"]:
                        case "Solar Wind":
                            output_file = solar_wind_samples_path
                        case "BS Magnetosheath":
                            output_file = bow_shock_magnetosheath_samples_path
                        case "MP Magnetosheath":
                            output_file = magnetopause_magnetosheath_samples_path
                        case "Magnetosphere":
                            output_file = magenetosphere_samples_path

                        case _:
                            raise ValueError(f"Unknown sample label: {sample['Label']}")

                    # If there is no data, only two columns:
                    # boundary id, and label
                    # This shouldn't happen!
                    if len(sample) == 2:
                        Safely_Append_Row(
                            output_file, [np.nan] * 14 + list(sample.values())
                        )
                        continue

                    # If the file doesn't exist, create it
                    if not os.path.exists(output_file):
                        os.mknod(output_file)

                        """
                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(list(sample.keys()))
                        """
                        Safely_Append_Row(output_file, list(sample.keys()))

                    else:
                        try:
                            pd.read_csv(output_file)
                        except pd.errors.EmptyDataError:
                            """
                            with open(output_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(list(sample.keys()))
                            """
                            Safely_Append_Row(output_file, list(sample.keys()))

                        """
                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(sample.values())
                        """
                        Safely_Append_Row(output_file, sample.values())


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
        raise ValueError("Attempting to process a data gap")

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


# Some fancy code to enable us to start and stop the script if needed, without losing progress
def Safely_Append_Row(output_file, sample):

    # Write to a temporary file first.
    # That way, if there are any corruptions, they won't occur in the main file
    tmp_file_name = ""
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(sample)
        tmp_file_name = tmp_file.name

    # Append the temp fiile contents atomically
    # i.e. the write happens at once, and errors can't occur from partial writes
    with (
        open(output_file, "a", newline="") as out_file,
        open(tmp_file_name, "r") as tmp_file,
    ):
        shutil.copyfileobj(tmp_file, out_file)

        # Flush python's buffer and force os to flush file to disk
        out_file.flush()
        os.fsync(out_file.fileno())

    # Clean tmp file
    os.remove(tmp_file.name)


if __name__ == "__main__":
    main()
