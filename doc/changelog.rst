.. _changelog:

.. towncrier release notes start

Foreshadow 0.3.2 (2019-12-03)
=============================

Features
--------

- Feature: Bug fix for intent override
    Add back the missing intent override (bug-fix-for-intent-override)


Foreshadow 0.3.0 (2019-11-21)
=============================

Features
--------

- Feature: Auto Intent Resolving
    Automatically resolve the intent of a column with a machine learning model. (auto-intent-resolving)
- Bug fix of pick_transformer may transform dataframe in place, causing
  inconsistency between the data and intended downstream logic. (bug-fix)
- Feature: Enable logging on Foreshadow
    This feature allows Foreshadow to display the progress of the training. (enable-logging)
- Feature: Adding more default models in Foreshadow
    This allows user to select estimators from the following categories:
    - Linear
    - SVM
    - RandomForest
    - NeuralNetwork (more-default-model)
- Feature: Allow user to override intent resolving decisions
    This feature allows users to override the intent resolving decisions
    through API calls. It can be done both before and after fitting the
    foreshadow object. (user-override)


Foreshadow 0.2.1 (2019-09-26)
=============================

Features
--------

- Bug fix of pick_transformer may transform dataframe in place, causing
  inconsistency between the data and intended downstream logic. (bug-fix)


Foreshadow 0.2.0 (2019-09-24)
=============================

Features
--------

- Add feature_summarizer to produce statistics about the data after
  intent resolving to show the users why such decisions are made. (data-summarization)
- Foreshadow is able to run end-to-end with level 1 optimization with the tpot
  auto-estimator. (level1-optimization)
- Add Feature Reducer as a passthrough transformation step. (pass-through-feature-reducer)
- Multiprocessing:
  1. Enable multiprocessing on the dataset.
  2. Collect changes from each process and update the original columnsharer. (process-safe-columnsharer)
- Serialization and deserialization:
  1. Serialization of the foreshadow object in a non-verbose format.
  2. Deserialization of the foreshadow object. (serialization)
- Adding two major components:
  1. usage of metrics for any statistic computation
  2. changing functionality of wrapping sklearn transformers to give them DataFrame capabilities. This now uses classes and metaclasses, which should be easier to maintain (#74)
- Adding ColumnSharer, a lightweight wrapper for a dictionary that functions
  as a cache system, to be used to pass information in the foreshadow pipeline. (#79)
- Creating DataPreparer to handle data preprocessing. Data Cleaning is the
  first step in this process. (#93)
- Adds skip resolve functionality to SmartTransformer, restructure utils, and add is_wrapped to utils (#95)
- Add serializer mixin and resture package import locations. (#96)
- Add configuration file parser. (#99)
- Add Feature Engineerer as a passthrough transformation step. (#112)
- Add Intent Mapper and Metric wrapper features. (#113)
- Add Preprocessor step to DataPreparer (#118)
- Create V2 architecture shift. (#162)


Foreshadow 0.1.0 (2019-06-28)
=============================

Features
--------

- Initial release. (#71)
