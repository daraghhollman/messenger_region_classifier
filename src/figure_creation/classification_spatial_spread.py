"""
We want to look at the location of each classified region, and show the spatial
distribution. Particularly, we want to show the ratio of each class in each
spatial bin.
"""

import datetime as dt

import hermpy.trajectory
import hermpy.utils
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Load post-processed regions (i.e. including unknown regions)
print("Loading region data")
regions = pd.read_csv("./data/postprocessing/regions_with_unknowns.csv")

# Shorten for testing
regions = regions.sample(frac=0.001)

regions["Start Time"] = pd.to_datetime(regions["Start Time"], format="ISO8601")
regions["End Time"] = pd.to_datetime(regions["End Time"], format="ISO8601")

# It is a little too intense to get the position at every second
predictions = {
    "Times": [],
    "Labels": [],
}
for region_start, region_end, label in tqdm(
    zip(regions["Start Time"], regions["End Time"], regions["Label"]),
    desc="Extracting times from regions",
    total=len(regions),
):
    times = pd.date_range(region_start, region_end, freq="10min")
    predictions["Times"].extend(times)
    predictions["Labels"].extend([label] * len(times))

# Find the location of messenger at all of these times.
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
region_names = (["Solar Wind", "Magnetosheath", "Magnetosphere", "Unknown"],)
region_maps = {}
for region_name in region_names:
    region_probability_maps = []

    # Filter observations by region
    filtered_predictions = predictions.loc[
        predictions["Predicted Region"] == region_name
    ][["X MSM' (radii)", "CYL MSM' (radii)"]]

    region_probability_map, _, _ = np.histogram2d(
        filtered_predictions["X MSM' (radii)"],
        filtered_predictions["CYL MSM' (radii)"],
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

    # We also don't want 0 values as they obscure the low value data
    # We do this when we plot instead
    # region_maps[region]["Mean"][np.where(region_maps[region]["Mean"] == 0)] = np.nan

# The cleanest way to save these data is as a netcdf file.
probability_map = xr.Dataset()

# Save dimensions (bin centres)
probability_map.coords["X"] = (x_bins[:-1] + x_bins[1:]) / 2
probability_map.coords["CYL"] = (cyl_bins[:-1] + cyl_bins[1:]) / 2

# Add other metadata
probability_map.attrs["Date created"] = dt.datetime.today().__str__()

# Save to NetCDF
output_file = "./data/cache/region_maps.nc"
probability_map.to_netcdf(output_file)
