"""
Create a figure to demonstrate how the data is reduced into samples. We will
show three rows. The first will show data surrounding a bow shock crossing
interval, with a sample in the solar wind and a sample in the magnetosheath
highlighted. The second row will have the distribution of magnetic field data,
as would be drawn from this sample. The third row contains a table describing
numberically the derived features of these two samples.
"""

import datetime as dt
import pathlib
import sys
from typing import Any

import hermpy.mag
import hermpy.plotting
import hermpy.utils
import matplotlib.dates
import matplotlib.pyplot as plt

start_time = dt.datetime(2013, 2, 1, 12)
end_time = dt.datetime(2013, 2, 1, 14)

# Specify paths for hermpy
hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = str(
    pathlib.Path(__file__).parent.parent.parent / "data" / "messenger" / "full_cadence"
)
hermpy.utils.User.METAKERNEL = str(
    pathlib.Path(__file__).parent.parent.parent
    / "SPICE"
    / "messenger"
    / "metakernel_messenger.txt"
)

# Load data
data = hermpy.mag.Load_Between_Dates(
    hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"],
    start_time,
    end_time,
    average=None,
    no_dirs=True,
)

sample_length = dt.timedelta(seconds=10)

# If the inteval is inbound or outbound the labelling of these samples can
# change, so we will just keep it ambiguous for now
data_samples: dict[str, Any] = {
    "Left": {
        # Positioned 1/4 of the way through the data
        "Start": start_time + (end_time - start_time) / 4 - sample_length / 2,
        "End": start_time + (end_time - start_time) / 4 + sample_length / 2,
    },
    "Right": {
        # Positioned 3/4 of the way through the data
        "Start": start_time + (end_time - start_time) * 3 / 4 - sample_length / 2,
        "End": start_time + (end_time - start_time) * 3 / 4 + sample_length / 2,
    },
}

# Add data to dictionaries
for key in data_samples.keys():
    data_samples[key].update(
        {
            "Data": data.loc[
                data["date"].between(
                    data_samples[key]["Start"], data_samples[key]["End"]
                )
            ]
        }
    )

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from reduce_data import get_sample_features

# Determine features for each sample
for key in data_samples.keys():

    data_samples[key].update(
        {"Features": get_sample_features(data_samples[key]["Data"])}
    )


# Plotting
grid_shape = (5, 2)
time_series_ax = plt.subplot2grid(grid_shape, loc=(0, 0), colspan=2)
left_distribution_ax = plt.subplot2grid(grid_shape, loc=(1, 0), rowspan=2)
right_distribution_ax = plt.subplot2grid(
    grid_shape,
    loc=(1, 1),
    sharex=left_distribution_ax,
    sharey=left_distribution_ax,
    rowspan=2,
)
features_table_ax = plt.subplot2grid(grid_shape, loc=(3, 0), rowspan=2, colspan=2)

# Time Series Plot
components = ["|B|", "Bx", "By", "Bz"]
mag_colours = [
    hermpy.plotting.wong_colours[c] for c in ["black", "red", "green", "blue"]
]

for var, colour in zip(components, mag_colours):
    time_series_ax.plot(
        data["date"],
        data[var],
        color=colour,
        label=var,
        alpha=1 if var == "|B|" else 0.75,
    )

time_series_ax.legend()
time_series_ax.set_ylabel("Magnetic Field Stregnth [nT]")


# Distributions
data_samples["Left"]["ax"] = left_distribution_ax
data_samples["Right"]["ax"] = right_distribution_ax

for key in data_samples.keys():

    sample = data_samples[key]

    for var, colour in zip(components, mag_colours):
        sample["ax"].hist(
            sample["Data"][var],
            color=colour,
            histtype="step",
            orientation="horizontal",
            lw=3,
            label=var,
        )

column_labels = list(data_samples["Left"]["Features"].keys())[2:]
row_labels = ["Solar Wind Sample", "Magnetosheath Sample"]

cell_text = [
    # We index these at the 2nd element to remove the time information returned
    [f"{element:.2f}" for element in list(data_samples["Left"]["Features"].values())][
        2:
    ],
    [f"{element:.2f}" for element in list(data_samples["Right"]["Features"].values())][
        2:
    ],
]

features_table_ax.table(
    cellText=cell_text, rowLabels=row_labels, colLabels=column_labels, loc="center"
)

# Disable axis frame
features_table_ax.axis("off")

# Adjust to make room for the elements
plt.tight_layout()

plt.savefig(
    pathlib.Path(__file__).parent.parent.parent
    / "figures/feature_extraction_example.pdf",
    format="pdf",
)
