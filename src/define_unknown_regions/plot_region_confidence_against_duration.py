import kneed
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from hermpy.plotting import wong_colours as colours

fig, ax = plt.subplots()

regions = pd.read_csv("./data/postprocessing/continous_regions.csv").dropna()

# We're not concerned with extremely high duration regions, so we remove
# anything above 3 sigma.
r_before = len(regions)
regions = regions[(np.abs(scipy.stats.zscore(regions["Duration"])) <= 3)]
r_after = len(regions)
print(f"{r_before - r_after} (of {r_before}) regions removed")


def fit_function(x, a, b):
    return (a * x) / (b + x)


# Repeat for all, but also each class on its own
fit = {
    "All": {"Colour": colours["black"]},
    "Solar Wind": {"Colour": colours["yellow"]},
    "Magnetosheath": {"Colour": colours["orange"]},
    "Magnetosphere": {"Colour": colours["blue"]},
}

# fig, axes = plt.subplots(1, 3)
# i = 0

for selection in fit.keys():

    if selection == "All":
        pars, cov = scipy.optimize.curve_fit(
            fit_function, regions["Duration"], regions["Confidence"]
        )
        fit_errors = np.sqrt(np.diag(cov))

    else:
        regions_subset = regions.loc[regions["Label"] == selection]

        print(f"Length of {selection} subset: { len(regions_subset) }")

        # _, _, _, hist = axes[i].hist2d(
        #     regions_subset["Duration"],
        #     regions_subset["Confidence"],
        #     norm="log",
        #     bins=50,
        # )
        # plt.colorbar(hist, ax=axes[i])
        # i += 1

        pars, cov = scipy.optimize.curve_fit(
            fit_function, regions_subset["Duration"], regions_subset["Confidence"]
        )
        fit_errors = np.sqrt(np.diag(cov))

    fit[selection].update({"Params": pars, "Errors": fit_errors})

# plt.show()


# Plot region data
_, _, _, hist = ax.hist2d(
    regions["Duration"],
    regions["Confidence"],
    norm="log",
    bins=50,
)

fig.colorbar(hist, ax=ax, label="# Regions")

# Plot fit
x_range = np.linspace(1, regions["Duration"].max(), 1000)
for key, values in fit.items():

    print(key)
    print(values)

    ax.plot(
        x_range,
        fit_function(x_range, *values["Params"]),
        color=values["Colour"],
        lw=4 if key == "All" else 2,
        label=f"Least squares fit (Region: {key})",
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=3, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
    )

# Find knee point
kneedle = kneed.KneeLocator(
    x_range,
    fit_function(x_range, *fit["All"]["Params"]),
    curve="concave",
    direction="increasing",
)

ax.axvline(
    kneedle.knee,
    color=colours["black"],
    ls="--",
    lw=2,
    label=f"Curve Knee = {kneedle.knee:.0f} seconds",
)
ax.axhline(
    fit_function(kneedle.knee, *fit["All"]["Params"]),
    color=colours["black"],
    ls=":",
    lw=2,
    label=f"Fit @ Curve Knee = {fit_function(kneedle.knee, *fit["All"]["Params"]):.2f}",
)

# Plot settings
ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence [arb.]")

ax.margins(0)

ax.legend()

fig.savefig("./figures/region_confidence_vs_duration.pdf", format="pdf")
