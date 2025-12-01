import datetime as dt

import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
from hermpy import mag, plotting
from hermpy.plotting import wong_colours
from mpl_toolkits.axes_grid1 import make_axes_locatable

wong_colours_list = list(wong_colours.values())

data = mag.Load_Mission("./data/mission_file.pkl")
x_positions = data["X MSM' (radii)"]
y_positions = data["Y MSM' (radii)"]
z_positions = data["Z MSM' (radii)"]

orbits = [
    {
        "Name": "Dayside",
        "Start": dt.datetime(2011, 12, 25, 2),
        "End": dt.datetime(2011, 12, 25, 14),
    },
    {
        "Name": "Nightside",
        "Start": dt.datetime(2013, 6, 5, 1),
        "End": dt.datetime(2013, 6, 5, 9),
    },
]

x_bins = np.linspace(-5, 5, 50).tolist()
z_bins = np.linspace(-8, 2, 50).tolist()
xy_histogram, x_edges, y_edges = np.histogram2d(
    x_positions, y_positions, bins=[x_bins, x_bins]
)
xz_histogram, x_edges, z_edges = np.histogram2d(
    x_positions, z_positions, bins=[x_bins, z_bins]
)

# These positions can then be plotted
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

xy_mesh = axes[0].pcolormesh(
    x_edges, y_edges, xy_histogram.T / 3600, norm="log", cmap="binary_r", zorder=-1
)
xz_mesh = axes[1].pcolormesh(
    x_edges, z_edges, xz_histogram.T / 3600, norm="log", cmap="binary_r", zorder=-1
)

divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="3%", pad=0.05)
fig.colorbar(
    xy_mesh, cax=cax, orientation="vertical", label="MESSENGER Residence [hours]"
)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="3%", pad=0.05)
fig.colorbar(
    xz_mesh, cax=cax, orientation="vertical", label="MESSENGER Residence [hours]"
)

planes = ["xy", "xz"]
hemisphere = ["left", "left"]
for i, ax in enumerate(axes):
    plotting.Plot_Mercury(
        axes[i], shaded_hemisphere="none", alpha=1, plane=planes[i], frame="MSM"
    )
    plotting.Add_Labels(axes[i], planes[i], frame="MSM'")
    plotting.Plot_Magnetospheric_Boundaries(ax, plane=planes[i])

    ax.set_aspect("equal")

    # Set axis background
    ax.axvspan(*ax.get_xlim(), color="#648FFF", alpha=0.3, zorder=-2)

axes[0].set_xlim(x_bins[0], x_bins[-1])
axes[0].set_ylim(x_bins[0], x_bins[-1])

axes[1].set_xlim(x_bins[0], x_bins[-1])
axes[1].set_ylim(z_bins[0], z_bins[-1])

# Add example orbits
for i, orbit in enumerate(orbits):

    # Limit data to within range
    orbit_data = data.loc[data["date"].between(orbit["Start"], orbit["End"])]
    x = orbit_data["X MSM' (radii)"]
    y = orbit_data["Y MSM' (radii)"]
    z = orbit_data["Z MSM' (radii)"]

    axes[0].plot(
        x,
        y,
        color=wong_colours_list[1:][i],
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
    )
    axes[1].plot(
        x,
        z,
        color=wong_colours_list[1:][i],
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
    )


plt.tight_layout()

plt.savefig("./figures/trajectories_example.pdf", format="pdf")
