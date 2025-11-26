"""
We want to look at the location of each classified region, and show the spatial
distribution. Particularly, we want to show the ratio of each class in each
spatial bin.
"""

import datetime as dt
import os

import hermpy.trajectory
import hermpy.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

FORCE_UPDATE = False
REGIONS_DATA_PATH = "./data/postprocessing/regions_with_unknowns.csv"
CACHE_PATH = "./data/cache/region_maps.nc"
FIGURE_PATH = "./figures/classification_spatial_spread.pdf"


def main():

    # Test if we need to create the probability map files
    if not os.path.isfile(CACHE_PATH) or FORCE_UPDATE:
        create_maps()

    plot_maps()

    plt.savefig(FIGURE_PATH, format="pdf")


def plot_maps():

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    unflattend_axes = axes
    axes = axes.flatten()

    probability_maps = xr.load_dataset(CACHE_PATH)

    regions = ["Solar Wind", "Magnetosheath", "Magnetosphere", "Unknown"]
    for i, region_name in enumerate(regions):
        data = probability_maps[f"{region_name.replace(' ', '_').lower()}"].T

        data.values[np.where(data.values == 0)] = np.nan

        axes[i].set_title(region_name)
        mesh = axes[i].pcolormesh(
            data.coords["X"], data.coords["CYL"], data.values, vmin=0, vmax=1
        )

        axes[i].set_aspect("equal")

        if axes[i] in unflattend_axes[:, 0]:
            axes[i].set_ylabel(
                r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
            )

        if axes[i] in unflattend_axes[1]:
            axes[i].set_xlabel(r"$X_{\rm MSM'} \quad \left[ \text{R}_\text{M} \right]$")

    cbar = fig.colorbar(
        mesh, ax=axes, label="Fraction of Regions of Type", location="right"
    )

    return fig


def create_maps():
    # Load post-processed regions (i.e. including unknown regions)
    regions = pd.read_csv(REGIONS_DATA_PATH)

    regions["Start Time"] = pd.to_datetime(regions["Start Time"], format="ISO8601")
    regions["End Time"] = pd.to_datetime(regions["End Time"], format="ISO8601")

    predictions = {
        "Times": [],
        "Labels": [],
    }
    for region_start, region_end, label in tqdm(
        zip(regions["Start Time"], regions["End Time"], regions["Label"]),
        total=len(regions),
        desc="Getting region times",
    ):
        # We don't need 1 second resolution for these maps. 1 minute is
        # sufficient.
        times = pd.date_range(region_start, region_end, freq="1min")
        predictions["Times"].extend(times)
        predictions["Labels"].extend([label] * len(times))

    # Find the location of messenger at all of these times.
    print("Finding positions")
    positions = (
        hermpy.trajectory.Get_Position(
            "MESSENGER", predictions["Times"], frame="MSM", aberrate=True
        )
        / hermpy.utils.Constants.MERCURY_RADIUS_KM
    )

    predictions["X"] = positions[:, 0]
    predictions["CYL"] = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)

    predictions = pd.DataFrame(predictions)

    # Create bins
    bin_size = 0.25
    x_bins = np.arange(-5, 5 + bin_size, bin_size)  # Radii
    cyl_bins = np.arange(0, 8 + bin_size, bin_size)  # Radii

    # Make 2D histograms for each region type
    region_names = ["Solar Wind", "Magnetosheath", "Magnetosphere", "Unknown"]
    region_maps = {}
    for region_name in region_names:
        region_probability_maps = []

        # Filter observations by region
        filtered_predictions = predictions.loc[predictions["Labels"] == region_name][
            ["X", "CYL"]
        ]

        region_probability_map, _, _ = np.histogram2d(
            filtered_predictions["X"],
            filtered_predictions["CYL"],
            bins=[x_bins, cyl_bins],
        )

        region_maps.update({region_name: region_probability_map})

    # Noramlise these maps so that they actually represent probabilities
    map_totals = np.sum(
        [region_maps[region] for region in region_names],
        axis=0,
    )

    for region in region_names:
        region_maps[region] /= map_totals

    # The cleanest way to save these data is as a netcdf file.
    probability_map = xr.Dataset()

    # Save dimensions (bin centres)
    probability_map.coords["X"] = (x_bins[:-1] + x_bins[1:]) / 2
    probability_map.coords["CYL"] = (cyl_bins[:-1] + cyl_bins[1:]) / 2

    # Add other metadata
    probability_map.attrs["Date created"] = dt.datetime.today().__str__()

    # Add data
    for region_name, data in region_maps.items():
        region_id = region_name.replace(" ", "_").lower()

        probability_map[f"{region_id}"] = (("X", "CYL"), data)

    # Save to NetCDF
    probability_map.to_netcdf(CACHE_PATH)

    # Check if output saved correctly
    loaded = xr.load_dataset(CACHE_PATH)
    print(loaded)


if __name__ == "__main__":
    main()
