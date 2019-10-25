# automl_research
Code repository for AutoML research to support Foreshadow project

---

## Feature Type Inference (Intent Resolution)
When analyzing raw data set feature columns in `Foreshadow`, the type (intent) of the each feature column has to be known a priori to select the appropriate feature transformation downstream.

The goal of this research project is to build an intent resolver that can separate numerical and categorical raw feature columns. More classes can be added in the future.

### Installation
This library was developed on Python 3.6.8 and uses the same package dependencies as `Foreshadow` as of Oct. 17, 2019.

To install additional package dependencies for research-based functionalities, run the following:
```
pip install -r research_requirements.txt
```

### Usage
The functionality of this library is exposed through the `IntentResolver` class API as shown below. The class outputs a prediction of "Numerical", "Categorical" or "Neither" for each raw feature column. Predictions with confidences lower than the `threshold` parameter (default = 0.7) in the `.predict` method are set to "Neither".

```
import pandas as pd
from lib import IntentResolver

# Initialise object
raw = pd.read_csv('path_to_dataset.csv', encoding='latin', low_memory=False)
resolver = IntentResolver(raw)

# Predict intent
# Outputs a pd.Series of predicted intents
resolver.predict()

# OR: Predict intent with confidences at a lower threshold (i.e. less rigorous prediction)
# Outputs a pd.DataFrame of predicted intent and confidences
resolver.predict(threshold=0.6, return_conf=True)
```


### Data Sources
- [Original Meta Data Set (OMDS)](https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/ML%20Schema%20Inference/Data)
- [360 Raw Data Sets (RDSs)](https://drive.google.com/file/d/1HGmDRBSZg-Olym2envycHPkb3uwVWHJX/view) (Sourced from the [GitHub README.md](https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/ML%20Schema%20Inference))


### References
1. V. Shah, P. Kumar, K. Yang, and A. Kumar, “Towards semi-automatic mlfeature type inference."
2. N. Hynes, D. Sculley, and M. Terry, “The data linter: Lightweight, auto-mated sanity checking for ml data sets,” in NIPS MLSys Workshop, 2017.
