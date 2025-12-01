"""
Script to create a plot showing two panels of mag data at different times - both around bow shock crossings.
The first shows a time where the individual crossings are clear, while the second shows a time where this isn't true.
"""

import datetime as dt

import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from hermpy import boundaries, mag, plotting

wong_colours = {
    "black": "black",
    "orange": "#E69F00",
    "light blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
}

# Load crossing intervals
philpott_2020 = boundaries.Load_Crossings("./data/philpott_2020_crossing_list.xlsx")

# Start with quasi-perpendicular case.
# Magnetic field is perpendicular to shock normal,
# we have clean crossings.
start_time = dt.datetime(2012, 7, 2, hour=17, minute=25)
end_time = dt.datetime(2012, 7, 2, hour=17, minute=40)

data = mag.Load_Between_Dates(
    "./data/messenger/one_second_avg/", start_time, end_time, no_dirs=True
)

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# Plot MAG data
axes[0].plot(
    data["date"], data["|B|"], color=wong_colours["black"], label="$|B|$", zorder=5
)
axes[0].plot(data["date"], data["Bx'"], color=wong_colours["red"], label="$B_x$")
axes[0].plot(data["date"], data["By'"], color=wong_colours["green"], label="$B_y$")
axes[0].plot(data["date"], data["Bz'"], color=wong_colours["blue"], label="$B_z$")

# Add boundary crossing interval
boundaries.Plot_Crossing_Intervals(
    axes[0],
    start_time,
    end_time,
    philpott_2020,
    color=wong_colours["black"],
    lw=3,
    height=1.1,
)


# Next we do the quasi-parallel case
start_time = dt.datetime(2013, 4, 28, hour=16, minute=50)
end_time = dt.datetime(2013, 4, 28, hour=17, minute=6)

data = mag.Load_Between_Dates(
    "./data/messenger/one_second_avg/", start_time, end_time, no_dirs=True
)

# Plot MAG data
axes[1].plot(
    data["date"], data["|B|"], color=wong_colours["black"], label="$|B|$", zorder=5
)
axes[1].plot(data["date"], data["Bx'"], color=wong_colours["red"], label="$B_x$")
axes[1].plot(data["date"], data["By'"], color=wong_colours["green"], label="$B_y$")
axes[1].plot(data["date"], data["Bz'"], color=wong_colours["blue"], label="$B_z$")

# Add boundary crossing interval
boundaries.Plot_Crossing_Intervals(
    axes[1],
    start_time,
    end_time,
    philpott_2020,
    color=wong_colours["black"],
    lw=3,
    height=1.1,
)

# Make legend lines larger
legend = axes[1].legend(loc="lower right", ncol=4)
for line in legend.get_lines():
    line.set_linewidth(2)


axis_labels = ["(a)", "(b)"]
for ax, l in zip(axes, axis_labels):

    ax.text(0.96, 0.9, l, fontsize="large", transform=ax.transAxes)

    ax.set_ylabel("Magnetic Field Strength [nT]")

    ax.margins(x=0)

    ax.axhline(0, ls="dashed", color="black")

    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=10))

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # Add ephemeris ticks
    plotting.Add_Tick_Ephemeris(ax)

    # Make ticks larger
    ax.xaxis.set_tick_params("major", length=10, width=1.5)
    ax.xaxis.set_tick_params("minor", length=5, width=1.5)

    ax.yaxis.set_tick_params("major", length=10, width=1.5)
    ax.yaxis.set_tick_params("minor", length=5, width=1.5)

    # y axes must be symmetric around 0
    ax.set_ylim(-max(np.abs(ax.get_ylim())), max(np.abs(ax.get_ylim())))


plt.tight_layout()

plt.savefig("./figures/contrasting_bow_shock_examples.pdf", format="pdf")
