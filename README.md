# A MESSENGER Magnetospheric Region Classifier
[![DOI](https://zenodo.org/badge/1082496652.svg)](https://doi.org/10.5281/zenodo.17790264)

Automatically classify whether MESSENGER was in the solar wind, magnetosheath,
or magnetosphere with > 98% accuracy based on magnetometer and ephemeris data
only.

See publication: [Hollman et al. (preprint, in review)](https://essopenarchive.org/users/867402/articles/1320325-identifying-messenger-magnetospheric-boundary-crossings-using-a-random-forest-region-classifier)

## Usage

The model was applied near boundary crossing intervals throughout the MESSENGER mission. Both the raw model output and the post-processed crossing list are available: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15706295.svg)](https://doi.org/10.5281/zenodo.15706295)

**Important:** We would like to highlight some caveats of using this list. As discussed in the publication, we suggest that this output crossing list primarily be used for statistical studies, whereas for individual event studies, we recommend manual evaluation of the raw model output.

Further, due to prior labelling of boundaries, and the automated nature of this work, there are times where you may wish to exclude / inspect manually. When determining a point in time as belonging to a given region based on a crossing, it is paramount that both the **crossing before and after** be referenced. For more detail, please find the guide on handling the crossing list [below](#quicklook)


## Quicklook

You can quickly test out this model with the Python notebook included below:

[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daraghhollman/messenger_region_classifier/blob/main/examples/minimal_application_example.ipynb)

Additionally, please see below for a guide on working with the outputs of this work:
* Working with the raw model output: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daraghhollman/messenger_region_classifier/blob/main/examples/raw_model_output_handelling.ipynb)
* Working with the crossing list: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daraghhollman/messenger_region_classifier/blob/main/examples/crossing_list_handelling.ipynb)


## Reproduce this work

Each step of this method is deterministic and completely reproducible. You can
find out more [here](./docs/REPRODUCE.md).
