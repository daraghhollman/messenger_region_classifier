"""
A figure applying the model to an entire orbit
"""

import datetime as dt
import pathlib
import sys

import hermpy.mag
import hermpy.plotting
import hermpy.utils
import matplotlib.dates
import matplotlib.pyplot as plt

start_time = dt.datetime(2013, 6, 1, 12)

orbit_length = dt.timedelta(hours=8)
end_time = start_time + orbit_length

# Specify paths for hermpy
# Set up data directories
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

# Load model and apply to data
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from apply_model import get_magnetospheric_region

classification_times, region_probabilities = get_magnetospheric_region(
    start_time,
    end_time,
    model_path=str(
        pathlib.Path(__file__).parent.parent.parent
        / "data/model/messenger_region_classifier.pkl"
    ),
)


# Create figure

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
mag_ax, proba_ax = axes

to_plot = ["|B|", "Bx", "By", "Bz"]
mag_colours = [
    hermpy.plotting.wong_colours[c] for c in ["black", "red", "green", "blue"]
]

for var, colour in zip(to_plot, mag_colours):
    mag_ax.plot(
        data["date"],
        data[var],
        color=colour,
        label=var,
        alpha=1 if var == "|B|" else 0.75,
    )

mag_ax.legend()
mag_ax.set_ylabel("Magnetic Field Stregnth [nT]")

labels = ["P(Magnetosheath)", "P(Magnetosphere)", "P(Solar Wind)"]
region_colours = [
    hermpy.plotting.wong_colours[c] for c in ["orange", "light blue", "yellow"]
]

for i, (colour, label) in enumerate(zip(region_colours, labels)):
    proba_ax.plot(
        classification_times, region_probabilities[:, i], color=colour, label=label
    )

proba_ax.legend()
proba_ax.xaxis.set_major_locator(
    matplotlib.dates.HourLocator(interval=2)
)  # Ticks every second hour
proba_ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d\n%H:%M"))
proba_ax.set_ylabel("Region Probability")

for ax in axes:
    ax.margins(x=0)

fig.savefig(
    pathlib.Path(__file__).parent.parent.parent
    / "figures/full_orbit_classification.pdf",
    format="pdf",
)
