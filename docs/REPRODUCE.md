# Reproduce this work

## Setup

```shell
git clone https://github.com/daraghhollman/messenger_region_classifier.git
cd messenger_region_classifier
```

The required Philpott+ (2020) boundary crossing intervals list is not available for automatic download. Please manually download [supporting_table_S1.tab](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/1U6FEO) as original format (.xlsx) and save to `./data/philpott_2020_crossing_list.xlsx`.

> [!TIP]
> For portability, we **strongly** recommend and use [uv](https://docs.astral.sh/uv/) to manage dependencies and versions. If you do not wish to use uv, dependencies can be found within [pyproject.toml](./pyproject.toml) and a `requirements.txt` file can be made with `pip compile pyproject.toml -o requirements.txt`.

Download the required MESSENGER data and SPICE kernels, along with other directory setup with:
(requires [wget](https://www.gnu.org/software/wget/))
```shell
chmod +x ./scripts/init
./scripts/init
```
This will take a long time, and is highly dependent on network and read/write speeds.

## Data Reduction

```shell
uv run python src/reduce_data.py
```

Processes the calibrated MESSENGER MAG (full cadence) data near bow shock and magnetopause crossing intervals defined through visual inspection by [Philpott et al. (2020)](https://doi.org/10.1029/2019JA027544). Outputs 4 files, each with labelled samples of solar wind, magnetosheath, or magnetosphere data.

## Feature Selection

```shell
uv run python src/select_features.py
```

Prior to optimisation, we evaluate 5 different feature subsets from those calculated in the data reduction step. 10 independent seed random forest models are trained, and each feature-set's performance is evaluated with Out-Of-Bag (OOB) score. Mean and standard deviation of OOB Score and training accuracy are recorded in `./data/metrics/feature_selection_metrics.csv`. The optimal feature-set is saved automatically for further use.

## Hyperparameter Optimisation

```shell
# Run with default 100 trials
uv run python src/optimise_model.py

# Or set trial count manually
# e.g. uv run src/optimise_model.py 50
uv run python src/optimise_model.py <num_trials>
```

We utilise the Python package [Optuna](https://optuna.org/) hyperparameter selection, and train 100 models while allowing the hyperparameters to vary. Optimised parameters are saved automatically for further use.

## Creating the model

```shell
uv run python src/create_model.py
```

Trains 30 independent random forest models with the optimised parameters and selected features. Training accuracy, OOB score, and impurity-based feature importance are recorded for each. The best performing model (highest mean OOB score) is saved to use in applications.


## Model Applications

```shell
uv run python src/apply_model.py
```

Contains function `get_magnetospheric_region()` along with an example in `main()` of how to set up data directories and run for any time range.

## Applying the model to the whole mission

```shell
uv run python src/apply_to_whole_mission.py
```

This will iterate through the MESSENGER data, applying the model near Philpott boundary crossing intervals. The raw model output is produced at: `data/raw_model_ouptut.csv`

## Determining crossings and post-processing

```shell
uv run python src/find_crossings_from_raw_model_output.py
```

The above will determine crossings from the raw model output and create two files:
* `data/postprocessing/crossings_with_unknowns.csv`, a list of crossings
* `data/postprocessing/regions_with_unknowns`, a list of intervals of continuous region classification

These are then post-processed with the following:

```shell
uv run python src/postprocessing/1_ensure_start_and_end.py
uv run python src/postprocessing/2_include_hidden_crossings.py
```

## Figures

Most scripts to recreate individual figures can be found in `src/figure_creation/`, with additional figure creation scripts in:
* `src/define_unknown_regions/`
* `src/quicklook/` (figures not used in this manuscript)
