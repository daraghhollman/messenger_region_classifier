import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from hermpy.plotting import wong_colours as colours

fig, ax = plt.subplots()

regions = pd.read_csv("./data/postprocessing/continous_regions.csv").dropna()
r_before = len(regions)

# We're not concerned with extremely high duration regions, so we remove
# anything above 3 sigma.
regions = regions[(np.abs(scipy.stats.zscore(regions["Duration"])) <= 3)]
r_after = len(regions)

print(f"{r_before - r_after} (of {r_before}) regions removed")


def fit_function(x, a, b):
    return (a * x) / (b + x)


# Repeat for all, but also each class on its own
fit = {
    "All": {"Colour": colours["black"]},
    "Solar Wind": {"Colour": colours["orange"]},
    "Magnetosheath": {"Colour": colours["red"]},
    "Magnetosphere": {"Colour": colours["yellow"]},
}
for selection in fit.keys():

    if selection == "All":
        pars, cov = scipy.optimize.curve_fit(
            fit_function, regions["Duration"], regions["Confidence"]
        )
        fit_errors = np.sqrt(np.diag(cov))

    else:
        regions_subset = regions.loc[regions["Label"] == selection]

        # ax.scatter(
        #     regions_subset["Duration"],
        #     regions_subset["Confidence"],
        #     marker=".",
        #     color=fit[selection]["Colour"],
        #     zorder=5,
        # )

        pars, cov = scipy.optimize.curve_fit(
            fit_function, regions_subset["Duration"], regions_subset["Confidence"]
        )
        fit_errors = np.sqrt(np.diag(cov))

    fit[selection].update({"Params": pars, "Errors": fit_errors})


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

    ax.plot(
        x_range,
        fit_function(x_range, *values["Params"]),
        color=values["Colour"],
        lw=5 if key == "All" else 2,
        label=f"Least squares fit (Region: {key})",
    )

# Find knee point
kneedle = kneed.KneeLocator(
    x_range,
    fit_function(x_range, *values["Params"]),
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

# We also need to save the parameters so we can define unknown regions.
pd.DataFrame(
    {
        "Parameter": ["A", "B"],
        "Value": fit["All"]["Params"],
        "Error": fit["All"]["Errors"],
    }
).to_csv("./data/postprocessing/unknown_region_parameters.csv")
