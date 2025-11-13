"""
Create a figure showing boxplotted feature importance of the optimised model
(30 runs), and average and std of the confusion matrix for the test set.
"""

import pickle

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hermpy.plotting import wong_colours as colours


def main():

    # First we load the feature importance data, and the confusion matrix data
    with open("./data/model/confusion_matrices.pkl", "rb") as f:
        confusion_matrices: list[np.typing.ArrayLike] = pickle.load(f)

    # This is a csv with 30 rows, corresponding to each model, and 20 columns,
    # corresponding to each feature.
    feature_importances = pd.read_csv("./data/model/feature_importances.csv").to_numpy()

    # Load selected features from ./src/select_features.py
    feature_names = []
    with open("./data/model/selected_features.txt", "r") as f:
        for line in f:
            feature_names.append(line.strip())

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Sorting
    mean_importances = np.mean(feature_importances, axis=0)
    sorted_indices = np.argsort(mean_importances)[::-1]

    feature_importances = feature_importances[:, sorted_indices]  # Reorder importances
    mean_importances = mean_importances[sorted_indices]  # Reorder mean values
    feature_names = [feature_names[i] for i in sorted_indices]  # Reorder feature names

    y_positions = np.arange(len(feature_names))
    # axes[0].barh(y_positions + 1, mean_importances, color="white", edgecolor="black")

    box_plots = axes[0].boxplot(
        feature_importances, widths=0.8, orientation="horizontal", patch_artist=True
    )

    for box, line in zip(box_plots["boxes"], box_plots["medians"]):
        box.set_facecolor(colours["light blue"])
        box.set_edgecolor(colours["black"])
        line.set_color(colours["black"])

    add_boxkey(axes[0], x_center=0.05, xshift=0, ypos=18, width=0.8, y_offset=0.15)

    # Add feature names as labels
    axes[0].set_yticks(ticks=np.arange(len(feature_names)) + 1, labels=feature_names)
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_title(
        f"Feature Importance Distribution for {len(feature_importances)} models"
    )

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    confusion_matrix_std = np.std(confusion_matrices, axis=0)

    # Plot the average confusion matrix
    heatmap = sns.heatmap(
        avg_confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        linecolor="black",
        xticklabels=["Solar Wind", "Magnetosheath", "Magnetosphere"],
        yticklabels=["Solar Wind", "Magnetosheath", "Magnetosphere"],
        norm=matplotlib.colors.LogNorm(),
        square=True,
        ax=axes[1],
    )

    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title(f"Confusion Matrix Average for {len(confusion_matrices)} models")

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("Mean Counts")

    # Annotate with standard deviation
    for i in range(avg_confusion_matrix.shape[0]):
        for j in range(avg_confusion_matrix.shape[1]):
            axes[1].text(
                j + 0.5,
                i + 0.65,
                f"±{confusion_matrix_std[i, j]:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black" if i == j else "white",
            )

    labels = ["(a)", "(b)"]
    for i, ax in enumerate(axes):
        ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, fontsize="large")

    plt.tight_layout()
    plt.savefig("figures/model_results.pdf", format="pdf")


# Simon's beautiful boxkey
def add_boxkey(
    axis,
    size=8,
    x_center=0.175,
    scale=False,
    ypos=9.5,
    xshift=0.0,
    width=0.8,
    boxplotkwargs={},
    y_offset=0.1,
    sample_data=False,
    **text_kwargs,
):
    """
    Adds a box plot key to a given axis and labels key components such as the median, quartiles, and outliers.

    Parameters:
    -----------
    axis : matplotlib axis
        The axis object to draw the box plot on.
    size : int, optional
        Font size for the text labels (default is 8).
    x_center : float, optional
        Controls the centering of the box plot data (default is 0.175).
    scale : bool or float, optional
        Determines the spread of the sample data, either a float or calculated automatically (default is False).
    ypos : float, optional
        Position on the y-axis where the box plot will be placed (default is 9.5).
    xshift : float, optional
        Shifts the sample data along the x-axis (default is 0).
    width : float, optional
        Width of the box plot (default is 0.8).
    boxplotkwargs : dict, optional
        Additional keyword arguments passed to the `axis.boxplot` function (default is {}).
    y_offset : float, optional
        Vertical offset for text labels (default is 0.1).
    sample_data : array or bool, optional
        Custom sample data to be used in the box plot. If False, synthetic data is generated (default is False).
    **text_kwargs : dict
        Additional keyword arguments passed to the `axis.text` function for text styling.

    Returns:
    --------
    sample_boxplot : dict
        A dictionary containing the artists of the box plot components (e.g., medians, boxes, whiskers, etc.).
    text : list
        A list of `matplotlib.text.Text` objects representing the text annotations added to the box plot.
    """

    # If no sample_data is provided, create synthetic data
    if not sample_data:
        np.random.seed(342314)  # Set a random seed for reproducibility
        if not scale:
            # If no scale is provided, calculate it based on x_center
            scale = 0.03 * (x_center / 0.175)
        # Generate synthetic data based on two normal distributions
        sample_data = (
            np.random.normal(0.2, scale=scale, size=(100,))
            + np.random.normal(0.15, scale=scale, size=(100,))
        ) * (x_center / 0.175)
        # Adjust the first data point downward
        sample_data[0] -= 0.15 * (x_center / 0.175)
        # Apply x-axis shift to the data
        sample_data += xshift
    sample_data[sample_data < np.median(sample_data)] -= 0.005
    sample_data[sample_data > np.median(sample_data)] += 0.005

    # Define default boxplot settings with horizontal orientation and mean line
    boxkwargs = dict(
        vert=False, showmeans=False, meanline=False, positions=[ypos], widths=width
    )
    # Update boxplot settings with any additional user-provided keyword arguments
    boxkwargs.update(boxplotkwargs)

    # Default text settings, including font size
    txt_kwargs = {"size": size}
    # Update text settings with any additional user-provided keyword arguments
    txt_kwargs.update(text_kwargs)

    # Create the box plot on the provided axis with the sample data
    sample_boxplot = axis.boxplot(sample_data, **boxkwargs)

    # Add text labels to annotate key components of the box plot
    text = [
        axis.text(
            sample_boxplot["medians"][0].get_xdata()[0],
            sample_boxplot["medians"][0].get_ydata()[0],
            s="Median",
            ha="center",
            va="top",
            color=sample_boxplot["medians"][0].get_color(),
            **txt_kwargs,
        ),
        axis.text(
            sample_boxplot["boxes"][0].get_xdata()[0],
            sample_boxplot["boxes"][0].get_ydata()[0] - y_offset * 3,
            s="Lower\nQuartile",
            ha="center",
            va="top",
            color=sample_boxplot["boxes"][0].get_color(),
            **txt_kwargs,
        ),
        axis.text(
            sample_boxplot["boxes"][0].get_xdata()[-2],
            sample_boxplot["boxes"][0].get_ydata()[-2] - y_offset * 3,
            s="Upper\nQuartile",
            ha="center",
            va="top",
            color=sample_boxplot["boxes"][0].get_color(),
            **txt_kwargs,
        ),
        # axis.text(sample_boxplot['means'][0].get_xdata()[0], sample_boxplot['means'][0].get_ydata()[-1],
        #           s='Mean', ha='center', va='bottom', color=sample_boxplot['means'][0].get_color(), **txt_kwargs),
        axis.text(
            sample_boxplot["caps"][0].get_xdata()[0],
            sample_boxplot["caps"][0].get_ydata()[-1] + y_offset * 2,
            s="Q1 - 1.5 IQR",
            ha="center",
            va="bottom",
            color=sample_boxplot["caps"][0].get_color(),
            **txt_kwargs,
        ),
        axis.text(
            sample_boxplot["caps"][1].get_xdata()[0],
            sample_boxplot["caps"][1].get_ydata()[-1] + y_offset * 2,
            s="Q3 + 1.5 IQR",
            ha="center",
            va="bottom",
            color=sample_boxplot["caps"][1].get_color(),
            **txt_kwargs,
        ),
        axis.text(
            sample_boxplot["fliers"][0].get_xdata()[0],
            sample_boxplot["fliers"][0].get_ydata()[-1] + y_offset,
            s="Outliers",
            ha="center",
            va="bottom",
            color=sample_boxplot["fliers"][0].get_color(),
            **txt_kwargs,
        ),
    ]

    # Return both the box plot object and the text annotations
    return sample_boxplot, text


if __name__ == "__main__":
    main()
