import collections
import datetime as dt

import matplotlib.dates
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, plotting, utils
from hermpy.plotting import wong_colours

matplotlib.rcParams["hatch.linewidth"] = 2

# Load Philpott crossing intervals and define crossing groups
# print("Loading crossings intervals")
crossing_intervals = boundaries.Load_Crossings(
    "./data/philpott_2020_crossing_list.xlsx", include_data_gaps=True
)

# print("Grouping crossing intervals")
crossing_groups = []
crossing_index = 0
while crossing_index < len(crossing_intervals) - 1:

    current_crossing = crossing_intervals.loc[crossing_index]
    next_crossing = crossing_intervals.loc[crossing_index + 1]

    assert isinstance(current_crossing, pd.Series)

    if current_crossing["Type"] == "BS_IN":
        # We expect a magnetopause in crossing next
        match next_crossing["Type"]:
            case "MP_IN":
                # This is as normal, we can add to our list of pairs
                crossing_groups.append([current_crossing, next_crossing])

                # We don't want to consider the next crossing as we have already
                # saved it, so we add an extra to the crossing index.
                crossing_index += 1

            case label if label in ["MP_OUT", "BS_IN", "BS_OUT", "DATA_GAP"]:
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

            case label if label in ["MP_IN", "MP_OUT", "BS_IN", "DATA_GAP"]:
                # This is abnormal, we just want to look around the current crossing
                crossing_groups.append([current_crossing])

    else:
        # Otherwise, for some reason the previous part of the crossing pair
        # didn't exist. We save this crossing on its own.
        if current_crossing["Type"] != "DATA_GAP":
            crossing_groups.append([current_crossing])

    crossing_index += 1


# Load probabilities
model_output = pd.read_csv("./data/raw_model_output.csv")
model_output["Time"] = pd.to_datetime(model_output["Time"], format="ISO8601")

# Load the new crossing list
new_crossings = pd.read_csv("./data/postprocessing/crossings_with_unknowns.csv")
new_crossings["Time"] = pd.to_datetime(new_crossings["Time"])

# We want to look at a specific crossing group

# 11369, very messy, maybe some kind of event
# 15916, magnetopause crossing is too late
for i in range(len(crossing_groups)):
    if crossing_groups[i][0].name != 15916:
        continue
    crossing_group = crossing_groups[i]

# Load data around the interval
interval_buffer = dt.timedelta(minutes=10)

if isinstance(crossing_group, pd.Series):
    start = crossing_group["Start Time"] - interval_buffer
    end = crossing_group["End Time"] + interval_buffer

else:
    start = crossing_group[0]["Start Time"] - interval_buffer
    end = crossing_group[1]["End Time"] + interval_buffer

# print("Loading data")
# print(f"Start: {start}")
# print(f"End: {end}")
messenger_data = mag.Load_Between_Dates(
    "./data/messenger/one_minute_avg", start, end, no_dirs=True
)

# Get model_ouput between these times
probabilities = model_output.loc[model_output["Time"].between(start, end)]

# Search the model output for new crossings in this interval
crossings_in_data = new_crossings.loc[
    new_crossings["Time"].between(start, end)
].reset_index(drop=True)

# Create a figure and plot the mag data
# print("Creating plot")
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 11))
(magnitude_axis, components_axis, probability_axis) = axes

# Plot the magnetic field components
for component, component_label, colour in zip(
    ["Bx'", "By'", "Bz'"], ["$B_x$", "$B_y$", "$B_z$"], ["red", "green", "blue"]
):
    components_axis.plot(
        messenger_data["date"],
        messenger_data[component],
        color=wong_colours[colour],
        label=component_label,
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
        zorder=1.5,
    )

magnitude_axis.plot(
    messenger_data["date"],
    messenger_data["|B|"],
    color=wong_colours["black"],
    zorder=1.5,
)

magnitude_axis.set_ylabel("|B| [nT]")

components_axis.set_ylabel("Magnetic Field Strength [nT]")
components_axis.axhline(0, color="black", ls="dotted", lw=2)
leg = components_axis.legend(loc="lower right")
for legobj in leg.legend_handles:
    legobj.set_linewidth(5)

for ax in axes:
    ax.margins(0)

probability_axis.plot(
    probabilities["Time"],
    probabilities["P(Solar Wind)"],
    color=wong_colours["yellow"],
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
    label="P(Solar Wind)",
)
probability_axis.plot(
    probabilities["Time"],
    probabilities["P(Magnetosheath)"],
    color=wong_colours["orange"],
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
    label="P(Magnetosheath)",
)
probability_axis.plot(
    probabilities["Time"],
    probabilities["P(Magnetosphere)"],
    color=wong_colours["light blue"],
    path_effects=[  # Add a black outline to the line
        matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
        matplotlib.patheffects.Normal(),
    ],
    label="P(Magnetosphere)",
)
leg = probability_axis.legend(loc="center right")
for legobj in leg.legend_handles:
    legobj.set_linewidth(5)

probability_axis.set_ylim(0, 1)
probability_axis.set_ylabel("Class Probability")

# Add boundary crossing intervals
# We only need start time within the data as crossing groups never spans
# part of an interval only
intervals_within_data = crossing_intervals.loc[
    crossing_intervals["Start Time"].between(start, end)
]

# Text-box formatting
text_box_formatting = dict(
    boxstyle="square", facecolor="white", edgecolor="black", pad=0.2, alpha=1
)

# LINES BEHIND AXES
for _, crossing_interval in intervals_within_data.iterrows():

    span = magnitude_axis.axvspan(
        crossing_interval["Start Time"],
        crossing_interval["End Time"],
        ymin=-2,
        ymax=0,
        fill=False,
        lw=2,
        ls="dashed",
        color=wong_colours["pink"],
        hatch="/",
        zorder=-1,
    )

    mid_point = (
        crossing_interval["Start Time"]
        + (crossing_interval["End Time"] - crossing_interval["Start Time"]) / 2
    )

    ax_1_transform = matplotlib.transforms.blended_transform_factory(
        axes[0].transData, axes[0].transAxes
    )
    axes[0].text(
        mid_point,
        -fig.subplotpars.hspace / 2,
        crossing_interval["Type"].replace("_", " "),
        ha="center",
        va="center",
        transform=ax_1_transform,
        bbox=text_box_formatting,
    )

    ax_2_transform = matplotlib.transforms.blended_transform_factory(
        axes[1].transData, axes[1].transAxes
    )
    axes[1].text(
        mid_point,
        -fig.subplotpars.hspace / 2,
        crossing_interval["Type"].replace("_", " "),
        ha="center",
        va="center",
        transform=ax_2_transform,
        bbox=text_box_formatting,
    )

    # start_line.set_clip_on(False)
    # end_line.set_clip_on(False)
    span.set_clip_on(False)

    # HATCHING
    for ax in axes:
        ax.axvspan(
            crossing_interval["Start Time"],
            crossing_interval["End Time"],
            fill=False,
            ls="dashed",
            lw=2,
            color=wong_colours["pink"],
            hatch="/",
            zorder=2,
            label="Philpott+ (2020) Crossing Intervals" if ax != axes[-1] else "",
        )

# Plot new crossings

crossing_labels = []
for index, c in crossings_in_data.iterrows():

    for ax in axes:
        ax.axvline(
            c["Time"],
            color="black",
            ls="dashed",
            zorder=5,
            label="Boundary Crossings (this work)",
        )

    assert isinstance(index, int)

    label_y = np.linspace(1.05, 1.3, 4)[index % 4]
    crossing_label = magnitude_axis.text(
        c["Time"],
        label_y,
        (c["Transition"].replace("_", " ") if "UKN" not in c["Transition"] else "UKN"),
        va="bottom",
        ha="center",
        fontweight="bold",
        fontsize="small",
        transform=magnitude_axis.get_xaxis_transform(),
        bbox=text_box_formatting,
    )
    crossing_labels.append(crossing_label)

    # Add a line between label and axis
    line = magnitude_axis.axvline(
        c["Time"],
        1,
        label_y,
        color="black",
        ls="dashed",
    )
    line.set_clip_on(False)

    shading_alpha = 0.7

    if index == 0:
        # Shade the region before the first crossing
        match c["Transition"]:

            case "BS_OUT" | "UKN (MSh -> UKN)" | "MP_IN":
                # Region was magnetosheath
                shade = wong_colours["orange"]

            case "BS_IN" | "UKN (SW -> UKN)":
                # Region was solar wind
                shade = wong_colours["yellow"]

            case "MP_OUT" | "UKN (MSp -> UKN)":
                # Region was magnetosphere
                shade = wong_colours["light blue"]

            case _:
                shade = "white"

        for ax in axes[:-1]:
            ax.axvspan(start, c["Time"], color=shade, alpha=shading_alpha)

        if len(crossings_in_data) == 1:
            # This is the only crossing
            # So we need to shade the next region too
            # Shade between the current crossing and the next
            match c["Transition"]:

                case "BS_OUT" | "UKN (UKN -> SW)":
                    # Region is solar wind
                    shade = wong_colours["yellow"]

                case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                    # Region is magnetosheath
                    shade = wong_colours["orange"]

                case "MP_IN" | "UKN (UKN -> MSp)":
                    # Region is magnetosphere
                    shade = wong_colours["light blue"]

                case _:
                    shade = "white"

            for ax in axes[:-1]:
                ax.axvspan(c["Time"], end, color=shade, alpha=shading_alpha)

    if index < len(crossings_in_data) - 1:

        # Shade between the current crossing and the next
        match c["Transition"]:

            case "BS_OUT" | "UKN (UKN -> SW)":
                # Region is solar wind
                shade = wong_colours["yellow"]

            case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                # Region is magnetosheath
                shade = wong_colours["orange"]

            case "MP_IN" | "UKN (UKN -> MSp)":
                # Region is magnetosphere
                shade = wong_colours["light blue"]

            case _:
                shade = "lightgrey"

        for ax in axes[:-1]:
            ax.axvspan(
                c["Time"],
                crossings_in_data.loc[index + 1]["Time"],
                color=shade,
                alpha=shading_alpha,
            )

    elif index == len(crossings_in_data) - 1:

        # Shade between the current crossing and the next
        match c["Transition"]:

            case "BS_OUT" | "UKN (UKN -> SW)":
                # Region is solar wind
                shade = wong_colours["yellow"]

            case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                # Region is magnetosheath
                shade = wong_colours["orange"]

            case "MP_IN" | "UKN (UKN -> MSp)":
                # Region is magnetosphere
                shade = wong_colours["light blue"]

            case _:
                shade = "lightgrey"

        for ax in axes[:-1]:
            ax.axvspan(c["Time"], end, color=shade, alpha=shading_alpha)

zoom_in = "left"

# Add crossing information to legend
# Some fance code from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
# to avoid duplicate legend labels
handles, labels = axes[0].get_legend_handles_labels()
by_label = collections.OrderedDict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys())

# Ensure components axis is symmetric around 0
max_y_lim_extent = np.max(np.abs(components_axis.get_ylim()))
components_axis.set_ylim(-max_y_lim_extent, max_y_lim_extent)

for ax in axes:

    # Zoom in
    if zoom_in == "left":
        ax.xaxis.set_major_locator(
            matplotlib.dates.MinuteLocator(byminute=np.arange(0, 60, 5))
        )
        ax.set_xlim(dt.datetime(2015, 3, 31, 6, 0), dt.datetime(2015, 3, 31, 6, 15))

    else:
        ax.xaxis.set_major_locator(
            matplotlib.dates.MinuteLocator(byminute=np.arange(0, 60, 20))
        )
    # Ensure ticks are above everything
    ax.set_axisbelow(False)  # Sets tick zorder to 2.5

    # Format ticks
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params("x", which="major", direction="inout", length=20, width=1.5)
    ax.tick_params("x", which="minor", direction="inout", length=10, width=1.5)

    ax.tick_params("y", which="major", direction="out", length=10)
    ax.tick_params("y", which="minor", direction="out", length=5)

# Add panel labels
for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
    panel_label = ax.text(
        -0.05, 1.05, label, fontsize="x-large", transform=ax.transAxes
    )
    panel_label.set_clip_on(False)

"""
# We want to add X MSM' to the third panel
feature_axis = probability_axis.twinx()

feature_axis.plot(
    messenger_data["date"],
    np.sqrt(
        messenger_data["Y MSM' (radii)"] ** 2 + messenger_data["Z MSM' (radii)"] ** 2
    ),
    color="black",
    lw=3,
)

feature_axis.set_ylabel(r"X MSM' [$R_{\rm M}$]")
"""

plotting.Add_Tick_Ephemeris(probability_axis)
plt.savefig("./figures/bad_application_example.pdf", format="pdf")
