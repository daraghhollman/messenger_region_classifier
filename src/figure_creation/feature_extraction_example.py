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
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

start_time = dt.datetime(2013, 6, 1, 16)
end_time = dt.datetime(2013, 6, 1, 16, 20)

# Specify paths for hermpy
hermpy.utils.User.DATA_DIRECTORIES["MAG_FULL"] = "./data/messenger/full_cadence"
hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"

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
fig = plt.figure(figsize=(8.3, 5.8))

grid_shape = (4, 6)
time_series_ax = plt.subplot2grid(grid_shape, loc=(0, 0), colspan=4, rowspan=2)
left_distribution_ax = plt.subplot2grid(grid_shape, loc=(2, 0), rowspan=2, colspan=2)
right_distribution_ax = plt.subplot2grid(
    grid_shape,
    loc=(2, 2),
    sharey=left_distribution_ax,
    rowspan=2,
    colspan=2,
)

# Inset some axes to show the time series for each distribution
left_inset_ax = left_distribution_ax.inset_axes((0.05, 0.03, 0.9, 0.1))
right_inset_ax = right_distribution_ax.inset_axes(
    (0.05, 0.03, 0.9, 0.1), sharey=left_inset_ax
)

inset_axes = [left_inset_ax, right_inset_ax]
for ax in inset_axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)


features_table_ax = plt.subplot2grid(grid_shape, loc=(0, 4), rowspan=4, colspan=2)

data_samples["Left"]["ax"] = left_distribution_ax
data_samples["Right"]["ax"] = right_distribution_ax

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

    left_data = data.loc[
        data["date"].between(data_samples["Left"]["Start"], data_samples["Left"]["End"])
    ]
    right_data = data.loc[
        data["date"].between(
            data_samples["Right"]["Start"], data_samples["Right"]["End"]
        )
    ]

    left_inset_ax.plot(
        left_data["date"],
        left_data[var],
        color=colour,
        label=var,
        alpha=1 if var == "|B|" else 0.75,
    )
    right_inset_ax.plot(
        right_data["date"],
        right_data[var],
        color=colour,
        label=var,
        alpha=1 if var == "|B|" else 0.75,
    )

# Set tick formatting
time_series_ax.xaxis.set_major_locator(
    matplotlib.dates.MinuteLocator([0, 5, 10, 15, 20])
)
time_series_ax.xaxis.set_major_formatter(
    matplotlib.dates.DateFormatter("%Y-%m-%d\n%H:%M")
)

time_series_ax.margins(0)
time_series_ax.legend()

time_series_ax.set_ylabel("Magnetic Field Strength [nT]")


# Distributions
for key in data_samples.keys():

    sample = data_samples[key]

    bin_size = 2.5  # nT
    bins = np.arange(-90, 90 + bin_size, bin_size)

    for var, colour in zip(components, mag_colours):
        sample["ax"].hist(
            sample["Data"][var],
            color=colour,
            histtype="step",
            bins=bins,
            orientation="horizontal",
            lw=3,
            label=var,
        )

data_samples["Left"]["ax"].set_ylabel("Magnetic Field Strength [nT]")

data_samples["Left"]["ax"].text(
    0.85, 0.9, "SW", ha="center", transform=data_samples["Left"]["ax"].transAxes
)
data_samples["Right"]["ax"].text(
    0.85, 0.9, "MSh", ha="center", transform=data_samples["Right"]["ax"].transAxes
)

fig.text(0.33, 0.024, "Binned Counts in Sample", ha="center")

# Table
row_labels = list(data_samples["Left"]["Features"].keys())[2:]
string_abbreviations = {
    "Heliocentric Distance (AU)": r"$R_H$ (AU)",
}
shortened_labels = []
for s in row_labels:
    for full, abbr in string_abbreviations.items():
        s = s.replace(full, abbr)

    shortened_labels.append(s)
row_labels = shortened_labels


column_labels = ["SW", "MSh"]

cell_text = np.array(
    [
        # We index these at the 2nd element to remove the time information returned
        [
            f"{element:.2f}"
            for element in list(data_samples["Left"]["Features"].values())
        ][2:],
        [
            f"{element:.2f}"
            for element in list(data_samples["Right"]["Features"].values())
        ][2:],
    ]
).T

table = features_table_ax.table(
    cellText=cell_text.tolist(),
    rowLabels=row_labels,
    colLabels=column_labels,
    colWidths=[0.3] * 2,
    rowLoc="right",
    loc="right",
)

table.auto_set_font_size(False)
table.set_fontsize(10)

# Disable axis frame
features_table_ax.axis("off")

plt.tight_layout()

fig.subplots_adjust(bottom=0.1)

# Shade where the samples are
for sample in [data_samples["Left"], data_samples["Right"]]:
    time_series_ax.axvspan(
        sample["Features"]["Sample Start"],
        sample["Features"]["Sample End"],
        color="grey",
        alpha=0.5,
    )

    # Define vertices in mixed coordinate systems
    start_num = matplotlib.dates.date2num(sample["Features"]["Sample Start"])
    end_num = matplotlib.dates.date2num(sample["Features"]["Sample End"])

    # Create transform that goes: data coords -> figure coords
    ts_to_fig = time_series_ax.transData + fig.transFigure.inverted()
    hist_to_fig = sample["ax"].transAxes + fig.transFigure.inverted()

    vertices = [
        ts_to_fig.transform((start_num, time_series_ax.get_ylim()[0])),
        ts_to_fig.transform((end_num, time_series_ax.get_ylim()[0])),
        hist_to_fig.transform((1, 1)),
        hist_to_fig.transform((0, 1)),
    ]

    poly = matplotlib.patches.Polygon(
        vertices,
        closed=True,
        facecolor="none",
        edgecolor="grey",
        alpha=0.5,
        transform=fig.transFigure,
        zorder=-10,
    )

    # Add to figure instead of ax, so it’s not clipped
    fig.patches.append(poly)


plt.savefig(
    pathlib.Path(__file__).parent.parent.parent
    / "figures/feature_extraction_example.pdf",
    format="pdf",
)
