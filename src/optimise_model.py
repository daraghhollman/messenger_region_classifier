"""
Now that we have our feature set selected, we can look at optimising some of the hyperparameters of the model
"""

import multiprocessing
import pickle

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

    # Load selected features from ./src/select_features.py
    features = []
    with open("./data/model/selected_features.txt", "r") as f:
        for line in f:
            features.append(line.strip())

    training_data = all_samples[features + ["Label"]]
    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    # Define an optuna objective
    def objective(trial: optuna.trial.Trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "n_jobs": -1,
            "oob_score": True,
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

    # Iterratively search for best parameters
    # In practice, we are training many models while varying the parameters,
    # optimizing for OOB Score.
    study.optimize(objective, n_trials=100)

    print(f"Best OOB score: {1 - study.best_value}")
    print("Best parameters thus far:")
    print(study.best_params)

    # Save model params to be loaded by create_model.py
    with open("./data/model/best_model_params.pkl", "wb") as f:
        pickle.dump(
            study.best_params.update(
                {
                    "n_jobs": -1,
                    "oob_score": True,
                    "random_state": SEED,
                }
            ),
            f,
        )

    # With these optimised parameters, we want to train many models and report
    # the training accuracy, oob error, and standard deviations.
    # This will be done in another script


if __name__ == "__main__":
    main()
