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

    # Loop through each feature set, train 10 models and
    feature_set_metrics = []
    for feature_set_id in feature_sets.keys():

        features = feature_sets[feature_set_id]

        # Subset data by features
        training_data = all_samples[features + ["Label"]]
        training_x = training_data.drop(columns="Label")
        training_y = training_data["Label"]

        # At this point there is no need for a training / testing split as we
        # will use training accuracy, and out-of-bag error to quantify
        # performance.

        # For each feature set, we will train 10 models, to capture the
        # variance on the accuracy metrics. If we used the same random state
        # for each model, we would get the exact same result. Hence, we
        # increment the seed in the loop.
        num_models = 10
        training_accuracies = []
        oob_scores = []
        for i in range(num_models):

            # Define model with default parameters at this time
            model = sklearn.ensemble.RandomForestClassifier(
                n_jobs=n_cores, random_state=SEED + i, oob_score=True
            )

            # Fit model to training data
            model.fit(training_x, training_y)

            # Evaluate training performance
            training_accuracy = model.score(training_x, training_y)
            oob = model.oob_score_

            training_accuracies.append(training_accuracy)
            oob_scores.append(oob)

        # Print summary
        print(f"Feature Set: {feature_set_id}")
        print(f"  Accuracy:")
        print(f"    Mean: {np.mean(training_accuracies):.4f}")
        print(f"    StD: {np.std(training_accuracies):.4f}")
        print(f"  OOB Score:")
        print(f"    Mean: {np.mean(oob_scores):.4f}")
        print(f"    StD: {np.std(oob_scores):.4f}")

        feature_set_metrics.append(
            {
                "Feature-set ID": feature_set_id,
                "Accuracy Mean": np.mean(training_accuracies),
                "Accuracy StD": np.std(training_accuracies),
                "OOB Score Mean": np.mean(oob_scores),
                "OOB Score StD": np.mean(oob_scores),
            }
        )

    # Save to csv for later reference
    pd.DataFrame(feature_set_metrics).to_csv(
        "./data/metrics/feature_selection_metrics.csv"
    )

    # Find which feature set had the largest OOB score
    mean_oob_scores = [metrics["OOB Score Mean"] for metrics in feature_set_metrics]
    best_model_index = np.argmax(mean_oob_scores)

    # **Guarenteed** a better way to store/access this information
    best_feature_set = feature_sets[
        feature_set_metrics[best_model_index]["Feature-set ID"]
    ]

    print(
        f"Saving features from set: {feature_set_metrics[best_model_index]["Feature-set ID"]}"
    )

    # Save selected features
    with open("./data/model/selected_features.txt", "w") as file:
        for feature in best_feature_set:
            file.write(feature + "\n")


# Define feature sets to explore
feature_sets = {
    "All Features": [
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
    ],
    "Reduced Features": [
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
        "Heliocentric Distance (AU)",
        "Local Time (hrs)",
        "Latitude (deg.)",
        "Magnetic Latitude (deg.)",
        "Mercury Distance (radii)",
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
    ],
    "No Ephemeris": [
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
    ],
    "Only Mean": [
        "Mean |B|",
        "Mean Bx",
        "Mean By",
        "Mean Bz",
    ],
    "Only Median": [
        "Median |B|",
        "Median Bx",
        "Median By",
        "Median Bz",
    ],
    "Only Standard Deviation": [
        "Standard Deviation |B|",
        "Standard Deviation Bx",
        "Standard Deviation By",
        "Standard Deviation Bz",
    ],
}

if __name__ == "__main__":
    main()
