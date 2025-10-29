"""
Now that we have our feature set selected, we can look at optimising some of the hyperparameters of the model
"""

import multiprocessing

import numpy as np
import optuna
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
    #                                   OPTIMISING MODEL                                 #
    ######################################################################################

    # We optimise some hyperparameters using the optuna library

    training_data = all_samples[features + ["Label"]]
    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    # Define an optuna objective
    def objective(trial: optuna.trial.Trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 250)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "n_jobs": -1,
            "random_state": SEED,
        }

        # Create model
        model = sklearn.ensemble.RandomForestClassifier(**model_params)

        model.fit(training_x, training_y)

        # Evaluate training performance
        oob_error = 1 - model.oob_score_
        training_accurary = model.score(training_x, training_y)

        trial.set_user_attr("training_accuracy", training_accurary)

        # We optimise features for reducing oob
        return oob_error

    # Create optuna study
    study = optuna.create_study()

    # Iterratively search for best parameters so that we can plot this later
    oob_scores = []
    training_accuracies = []
    trials_ellapsed = []
    n_steps = 10
    for i in range(n_steps):
        study.optimize(objective, n_trials=10)

        trials_ellapsed.append(i * 10 + 10)
        oob_scores.append(1 - study.best_value)
        training_accuracies.append(
            # We record the maxmimum training accuracy as the trials continue
            np.max(
                np.array([t.user_attrs.get("training_accuracy") for t in study.trials])
            )
        )

        print(f"{i * 10 + 10} trials completed")
        print(f"Best OOB score: {1 - study.best_value}")

    # Save this score for each iterration
    optimisation_metrics = pd.DataFrame(
        {
            "" "OOB Score": oob_scores,
            "Training Accuracy": training_accuracies,
        }
    ).to_csv("./data/metrics/optimisation_metrics.csv")

    # With these optimised parameters, we want to train many models and report
    # the training accuracy, oob error, and standard deviations.

    # num_models = 30  # This choice is somewhat arbitrary. Reviewers recommended between 30-50 models
    #
    # for model_index in range(num_models):
    #     pass


# Define the chosen feature set
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
