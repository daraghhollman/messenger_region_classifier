"""
Now that we have our feature set selected, we can look at optimising some of the hyperparameters of the model
"""

import multiprocessing
import pickle
import sys

import optuna
import pandas as pd
import sklearn.ensemble

from load_data import load_reduced_data

SEED = 1785
n_cores = max(multiprocessing.cpu_count() - 1, 1)

if len(sys.argv) > 1:
    num_trials = int(sys.argv[1])

else:
    num_trials = 100


def main():

    ######################################################################################
    #                          LOADING & VALIDATING DATA                                 #
    ######################################################################################

    training_data, _ = load_reduced_data()

    ######################################################################################
    #                                   OPTIMISING MODEL                                 #
    ######################################################################################

    # We optimise some hyperparameters using the optuna library

    # Load selected features from ./src/select_features.py
    features = []
    with open("./data/model/selected_features.txt", "r") as f:
        for line in f:
            features.append(line.strip())

    training_data = training_data[features + ["Label"]]
    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    # Define an optuna objective
    def objective(trial: optuna.trial.Trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 10, 100)
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
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED))

    # Iterratively search for best parameters
    # In practice, we are training many models while varying the parameters,
    # optimizing for OOB Score.
    study.optimize(objective, n_trials=num_trials)

    print(f"Best OOB score: {1 - study.best_value}")
    print("Best parameters thus far:")
    print(study.best_params)

    # Save model params to be loaded by create_model.py
    with open("./data/model/best_model_params.pkl", "wb") as f:
        params = study.best_params
        params.update(
            {
                "n_jobs": -1,
                "oob_score": True,
                "random_state": SEED,
            }
        )
        pickle.dump(
            params,
            f,
        )

    # With these optimised parameters, we want to train many models and report
    # the training accuracy, oob error, and standard deviations.
    # This will be done in another script


if __name__ == "__main__":
    main()
