"""
Isolating which features should be included by training multiple random forest
models with the addition of a feature which randomly samples from a normal
distriubtion with mean 0 and std 1. Features with average importance less than
that of the random feature will be marked to be excluded.
"""

import multiprocessing

import numpy as np
import pandas as pd
import sklearn.ensemble

SEED = 1785
n_cores = max(multiprocessing.cpu_count() - 1, 1)


def main():

    ######################################################################################
    #                          LOADING & VALIDATING DATA                                 #
    ######################################################################################

    # Load features datasets
    # These are in 4 different data sets which we need to combine
    inputs = {
        "Solar Wind": "./data/reduced/solar_wind_samples.csv",
        "BS Magnetosheath": "./data/reduced/bs_magnetosheath_samples.csv",
        "MP Magnetosheath": "./data/reduced/mp_magnetosheath_samples.csv",
        "Magnetopause": "./data/reduced/magnetosphere_samples.csv",
    }

    all_samples = []
    for region_type in inputs.keys():
        all_samples.append(pd.read_csv(inputs[region_type]))

    # Combine into one dataframe
    all_samples = pd.concat(all_samples, ignore_index=True).dropna()

    # Combine labelled for bow shock and magnetopause adjacent magnetosheath
    # samples, and balance classes
    all_samples["Label"] = all_samples["Label"].replace(
        "BS Magnetosheath", "Magnetosheath"
    )
    all_samples["Label"] = all_samples["Label"].replace(
        "MP Magnetosheath", "Magnetosheath"
    )

    # Check if there are extreme values in the dataset and remove them as these
    # are either an ICME, or non-physical, neither of which we want to include
    # in our training.
    extreme_rows = all_samples.loc[
        (all_samples["Mean |B|"] >= 5000)
        | (all_samples["Standard Deviation |B|"] >= 5000)
    ]
    all_samples = all_samples.drop(extreme_rows.index)

    # Balance classes through undersampling
    smallest_class_length = all_samples["Label"].value_counts().min()

    all_samples = all_samples.groupby("Label").apply(
        lambda class_data: class_data.sample(n=smallest_class_length, random_state=SEED)
    )

    # Print a summary
    print(
        f"""
        Full dataset lengths:
        Size: {len(all_samples)}
            SW: {len(all_samples.loc[all_samples["Label"] == "Solar Wind"])}
            MSh: {len(all_samples.loc[all_samples["Label"] == "Magnetosheath"])}
            MSp: {len(all_samples.loc[all_samples["Label"] == "Magnetosphere"])}
        """
    )

    ######################################################################################
    #                               FEATURE SELECTION                                    #
    ######################################################################################

    # Lets first add a random feature to the data. This feature will sample
    # from a normal distribution with mean 0, and standard deviation 1.
    all_samples["Normal Distribution"] = np.random.normal(size=len(all_samples))

    # We train on 10 models, and note the feature importances for each. We save
    # these values to be visualised later. We calculate the average feature
    # importance over all models, and sort them for comparison.

    training_data = all_samples[features + ["Normal Distribution", "Label"]]

    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    num_models = 10
    model_feature_importances = []
    for i in range(num_models):

        # Define model with default parameters at this time
        model = sklearn.ensemble.RandomForestClassifier(
            n_jobs=n_cores, random_state=SEED + i
        )

        # Fit model to training data
        model.fit(training_x, training_y)

        model_feature_importances.append(model.feature_importances_)

    # Save these importances for later visualisation
    pd.DataFrame(
        model_feature_importances, columns=np.array(features + ["Normal Distribution"])
    ).to_csv("./data/metrics/feature_importances.csv", index=False)

    # Find mean and sort
    model_feature_importances = np.array(model_feature_importances).T
    mean_importances = np.mean(model_feature_importances, axis=1)
    sorted_feature_indices = np.argsort(mean_importances)[::-1]

    ordered_features = [training_x.columns[i] for i in sorted_feature_indices]

    # We remove non-important features
    random_feature_index = ordered_features.index("Normal Distribution")

    remaining_features = ordered_features[:random_feature_index]
    removed_features = ordered_features[random_feature_index:]

    # We then save these feature names to file so they can be loaded by future models.
    print(f"Removed features: {removed_features}")

    # Save selected features
    with open("./data/metrics/selected_features.txt", "w") as file:
        for feature in remaining_features:
            file.write(feature + "\n")


# A list of all features
features = [
    "Mean |B|",
    "Mean Bx",
    "Mean By",
    "Mean Bz",
    "Median |B|",
    "Median Bx",
    "Median By",
    "Median Bz",
    "Standard Deviation |B|",
    "Standard Deviation Bx",
    "Standard Deviation By",
    "Standard Deviation Bz",
    "Skew |B|",
    "Skew Bx",
    "Skew By",
    "Skew Bz",
    "Kurtosis |B|",
    "Kurtosis Bx",
    "Kurtosis By",
    "Kurtosis Bz",
    "Heliocentric Distance (AU)",
    "Local Time (hrs)",
    "Latitude (deg.)",
    "Magnetic Latitude (deg.)",
    "Mercury Distance (radii)",
    "X MSM' (radii)",
    "Y MSM' (radii)",
    "Z MSM' (radii)",
]

if __name__ == "__main__":
    main()
