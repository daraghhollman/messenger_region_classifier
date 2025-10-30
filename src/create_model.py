"""
Takes optimise parameters from ./src/optimise_model.py and trains many random
seed models with these parameters to measure performance and variance.
"""

import multiprocessing
import pickle

import numpy as np
import pandas as pd
import sklearn.ensemble
from tqdm import tqdm

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
    #                               TRAIN AND SAVE MODEL                                 #
    ######################################################################################

    # Load selected features from ./src/select_features.py
    features = []
    with open("./data/model/selected_features.txt", "r") as f:
        for line in f:
            features.append(line.strip())

    training_data = all_samples[features + ["Label"]]
    training_x = training_data.drop(columns="Label")
    training_y = training_data["Label"]

    # Load best model parameters from file. Generated with ./src/optimise_model.py
    with open("./data/model/best_model_params.pkl", "rb") as f:
        model_params: dict = pickle.load(f)

    num_models = 30

    model_data = {
        "Model Index": [],
        "Model": [],
        "OOB Score": [],
        "Training Accuracy": [],
    }
    model_feature_importances = []
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

        model_data["Model Index"].append(i)
        model_data["Model"].append(model)
        model_data["OOB Score"].append(oob_score)
        model_data["Training Accuracy"].append(training_accuracy)

        model_feature_importances.append(model.feature_importances_)

    # Save each model's feature importances for later visualisation
    pd.DataFrame(model_feature_importances, columns=np.array(features)).to_csv(
        "./data/model/feature_importances.csv", index=False
    )

    # Pickle best model for further use
    best_model = model_data["Model"][np.argmax(model_data["OOB Score"])]
    with open("./data/model/messenger_region_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Pickle all model objects and save the rest as a csv
    with open("./data/model/all_models.pkl", "wb") as f:
        pickle.dump(model_data.pop("Model"), f)

    pd.DataFrame(model_data).to_csv("./data/model/performance_metrics.csv")

    print(
        f"Average OOB Score: {np.mean(model_data["OOB Score"]):.4f} +/- {np.std(model_data["OOB Score"]):.4f}"
    )
    print(
        f"Average Training Accuracy: {np.mean(model_data["Training Accuracy"]):.4f} +/- {np.std(model_data["Training Accuracy"]):.4f}"
    )


if __name__ == "__main__":
    main()
