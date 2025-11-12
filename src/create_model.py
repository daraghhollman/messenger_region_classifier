"""
Takes optimise parameters from ./src/optimise_model.py and trains many random
seed models with these parameters to measure performance and variance.
"""

import multiprocessing
import pickle

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.metrics
from tqdm import tqdm

from load_data import load_reduced_data

SEED = 1785
n_cores = max(multiprocessing.cpu_count() - 1, 1)


def main():

    ######################################################################################
    #                          LOADING & VALIDATING DATA                                 #
    ######################################################################################

    loaded_training_data, loaded_test_data = load_reduced_data()

    ######################################################################################
    #                               TRAIN AND SAVE MODEL                                 #
    ######################################################################################

    # Load selected features from ./src/select_features.py
    features = []
    with open("./data/model/selected_features.txt", "r") as f:
        for line in f:
            features.append(line.strip())

    training_data = loaded_training_data[features + ["Label"]]
    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    testing_data = loaded_test_data[features + ["Label"]]
    testing_x = testing_data.drop(columns="Label")
    testing_y = testing_data["Label"]

    # Load best model parameters from file. Generated with ./src/optimise_model.py
    with open("./data/model/best_model_params.pkl", "rb") as f:
        model_params: dict = pickle.load(f)

    num_models = 30

    model_data = {
        "Model Index": [],
        "Model": [],
        "OOB Score": [],
        "Training Accuracy": [],
        "Testing Accuracy": [],
        "Confusion Matrices": [],
        "Feature Importances": [],
    }
    for i in tqdm(range(num_models)):

        # We need a different (but fixed) random state for each model to
        # measure variance. To do this, we increment the initial fixed seed of
        # 1785 by the model index (+0, +1, +2, etc.).
        updated_model_params = model_params.copy()
        updated_model_params["random_state"] += i

        model = sklearn.ensemble.RandomForestClassifier(**updated_model_params)
        model.fit(training_x, training_y)

        # Evaluate performance
        oob_score = model.oob_score_
        training_accuracy = model.score(training_x, training_y)
        testing_accuracy = model.score(testing_x, testing_y)

        # Evaluate confusion matrix for testing data
        y_predictions = model.predict(testing_x)
        confusion_matrix = sklearn.metrics.confusion_matrix(
            testing_y,
            y_predictions,
            labels=["Solar Wind", "Magnetosheath", "Magnetosphere"],
        )

        model_data["Model Index"].append(i)
        model_data["Model"].append(model)
        model_data["OOB Score"].append(oob_score)
        model_data["Training Accuracy"].append(training_accuracy)
        model_data["Testing Accuracy"].append(testing_accuracy)
        model_data["Feature Importances"].append(model.feature_importances_)
        model_data["Confusion Matrices"].append(confusion_matrix)

    # Save each model's feature importances for later visualisation
    pd.DataFrame(
        model_data.pop("Feature Importances"), columns=np.array(features)
    ).to_csv("./data/model/feature_importances.csv", index=False)

    # Pickle best model for further use
    best_model = model_data["Model"][np.argmax(model_data["OOB Score"])]
    with open("./data/model/messenger_region_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Pickle all model objects
    with open("./data/model/all_models.pkl", "wb") as f:
        pickle.dump(model_data.pop("Model"), f)

    # Pickle all confusion matrix arrays
    with open("./data/model/confusion_matrices.pkl", "wb") as f:
        pickle.dump(model_data.pop("Confusion Matrices"), f)

    pd.DataFrame(model_data).to_csv("./data/model/performance_metrics.csv")

    print(
        f"Average OOB Score: {np.mean(model_data["OOB Score"]):.4f} +/- {np.std(model_data["OOB Score"]):.4f}"
    )
    print(
        f"Average Training Accuracy: {np.mean(model_data["Training Accuracy"]):.4f} +/- {np.std(model_data["Training Accuracy"]):.4f}"
    )
    print(
        f"Average Testing Accuracy: {np.mean(model_data["Testing Accuracy"]):.4f} +/- {np.std(model_data["Testing Accuracy"]):.4f}"
    )


if __name__ == "__main__":
    main()
