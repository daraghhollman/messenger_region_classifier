import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from hermpy.plotting import wong_colours as colours

regions = pd.read_csv("./data/postprocessing/continous_regions.csv").dropna()
r_before = len(regions)

# We're not concerned with extremely high duration regions, so we remove
# anything above 3 sigma.
regions = regions[(np.abs(scipy.stats.zscore(regions["Duration"])) <= 3)]
r_after = len(regions)

print(f"{r_before - r_after} (of {r_before}) regions removed")


def rational_fit(x, a, b):
    return (a * x) / (b + x)


pars, cov = scipy.optimize.curve_fit(
    rational_fit, regions["Duration"], regions["Confidence"]
)
fit_errors = np.sqrt(np.diag(cov))

fig, ax = plt.subplots()

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
ax.plot(
    x_range,
    rational_fit(x_range, *pars),
    color=colours["orange"],
    lw=3,
    label="Least Squares fit of form:" + r"$\frac{Ax}{B + x}$",
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
        "Value": pars,
        "Error": fit_errors,
    }
).to_csv("./data/postprocessing/unknown_region_parameters.csv")
