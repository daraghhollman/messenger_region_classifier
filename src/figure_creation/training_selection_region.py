"""
Show visually the regions in which our training data is selected.
"""

import datetime as dt

import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
from hermpy import boundaries, mag, plotting
from hermpy.plotting import wong_colours

colours = ["black", wong_colours["red"], wong_colours["green"], wong_colours["blue"]]

# import crossings
crossings = boundaries.Load_Crossings("./data/philpott_2020_crossing_list.xlsx")

# Limit to bow shock crossings only
crossings = crossings.loc[crossings["Type"].str.contains("BS")]

# Pick a crossing
crossing = crossings.iloc[1000]

# Determine start and stop times of each sample
if crossing["Type"] == "BS_IN":
    sw_sample_start = crossing["Start Time"] - dt.timedelta(minutes=10)
    sw_sample_end = crossing["Start Time"]

    msh_sample_start = crossing["End Time"]
    msh_sample_end = crossing["End Time"] + dt.timedelta(minutes=10)

else:
    msh_sample_start = crossing["Start Time"] - dt.timedelta(minutes=10)
    msh_sample_end = crossing["Start Time"]

    sw_sample_start = crossing["End Time"]
    sw_sample_end = crossing["End Time"] + dt.timedelta(minutes=10)


time_buffer = dt.timedelta(minutes=1)
all_times = [sw_sample_start, sw_sample_end, msh_sample_start, msh_sample_end]

# Load the data
data = mag.Load_Between_Dates(
    "./data/messenger/one_second_avg/",
    min(all_times) - time_buffer,
    max(all_times) + time_buffer,
    no_dirs=True,
)

sw_sample_data = data.loc[data["date"].between(sw_sample_start, sw_sample_end)]
msh_sample_data = data.loc[data["date"].between(msh_sample_start, msh_sample_end)]


# Plotting
fig = plt.figure(figsize=(8, 8))
mag_axis = plt.subplot2grid((2, 2), (0, 0), colspan=2)
left_sample_axis = plt.subplot2grid((2, 2), (1, 0))
right_sample_axis = plt.subplot2grid(
    (2, 2), (1, 1), sharey=left_sample_axis, sharex=left_sample_axis
)

axes = (mag_axis, left_sample_axis, right_sample_axis)

# Plot time series
mag_axis.plot(
    data["date"],
    data["|B|"],
    color=wong_colours["black"],
    lw=1,
    label="|B|",
)
mag_axis.plot(
    data["date"],
    data["Bx'"],
    color=wong_colours["red"],
    lw=1,
    label="Bx",
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
)
mag_axis.plot(
    data["date"],
    data["By'"],
    color=wong_colours["green"],
    lw=1,
    label="By",
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
)
mag_axis.plot(
    data["date"],
    data["Bz'"],
    color=wong_colours["blue"],
    lw=1,
    label="Bz",
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
)

boundaries.Plot_Crossing_Intervals(
    mag_axis,
    data["date"].iloc[0],
    data["date"].iloc[-1],
    crossings,
    color="black",
    lw=3,
    height=0.95,
)
plotting.Add_Tick_Ephemeris(
    mag_axis,
    include={
        "date",
        "hours",
        "minutes",
    },
)


# Highlight samples
mag_axis.axvspan(
    sw_sample_start,
    sw_sample_end,
    color=wong_colours["yellow"],
    zorder=-1,
    label="Solar Wind Training Selection Region",
)
mag_axis.axvspan(
    msh_sample_start,
    msh_sample_end,
    color=wong_colours["orange"],
    zorder=-1,
    label="Magnetosheath Training Selection Region",
)

mag_legend = mag_axis.legend(
    bbox_to_anchor=(0.5, 1.1), loc="center", ncol=3, borderaxespad=0.5
)

# set the linewidth of each legend object
for legobj in mag_legend.legend_handles:
    legobj.set_linewidth(3.0)

bin_size = 5  # nT
bins = np.arange(-90, 90 + bin_size, bin_size)
components = ["|B|", "Bx'", "By'", "Bz'"]
if crossing["Type"] == "BS_IN":

    left_sample_axis.set_title("Solar Wind Training Region")
    right_sample_axis.set_title("Magnetosheath Training Region")

    for component, colour in zip(components, colours):

        left_sample_hist, bin_edges = np.histogram(sw_sample_data[component], bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        left_sample_axis.stairs(
            left_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

        right_sample_hist, bin_edges = np.histogram(
            msh_sample_data[component], bins=bins
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        right_sample_axis.stairs(
            right_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )
else:

    right_sample_axis.set_title("Solar Wind Training Region")
    left_sample_axis.set_title("Magnetosheath Training Region")

    for component, colour in zip(components, colours):

        left_sample_hist, bin_edges = np.histogram(
            msh_sample_data[component], bins=bins
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        left_sample_axis.stairs(
            left_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

        right_sample_hist, bin_edges = np.histogram(
            sw_sample_data[component], bins=bins
        )
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        right_sample_axis.stairs(
            right_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

panel_labels = ["(a)", "(b)", "(c)"]
for i, ax in enumerate(axes):
    ax.axhline(0, color="black", ls="dotted")

    # Add panel labels
    ax.text(-0.05, 1.05, panel_labels[i], transform=ax.transAxes, fontsize="large")

    # y axes must be symmetric around 0
    ax.set_ylim(-max(np.abs(ax.get_ylim())), max(np.abs(ax.get_ylim())))

mag_axis.margins(x=0)

right_sample_axis.set_xlabel("Number of observations within bin")
left_sample_axis.set_xlabel("Number of observations within bin")

left_sample_axis.set_ylabel("Magnetic Field Strength [nT]")
mag_axis.set_ylabel("Magnetic Field Strength [nT]")

plt.tight_layout()
plt.savefig(
    "./figures/training_selection_region.pdf",
    format="pdf",
)
