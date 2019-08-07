from foreshadow.preparer import DataPreparer
from foreshadow.steps import CleanerMapper
from foreshadow.steps import IntentMapper
from foreshadow.steps import Preprocessor
from foreshadow.columnsharer import ColumnSharer
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sklearn.datasets as dt

from foreshadow.smart import Scaler

from foreshadow.utils.testing import debug; debug()

from hyperopt import hp
import hyperopt.pyll.stochastic

data = dt.load_iris()

X_data = pd.DataFrame(data.data, columns=data.feature_names).iloc[:, 0]
y_data = pd.DataFrame(data.target, columns=['target'])

# cs = ColumnSharer()
# p = Preprocessor(column_sharer=cs)
s = Scaler()
lr = LogisticRegression()

pipe = Pipeline([('s', s), ('lr', lr)])

pipe.fit(X_data, y_data)

param_distributions = hp.choice(
        's__transformer',
        [
        {
            'class_name': 'StandardScaler',
            'with_mean': hp.choice('wm', [False, True]),
        },
        {
            'class_name': 'MinMaxScaler',
            'feature_range': hp.choice('fr', [(0, 1), (0, 0.5)])
        }
])

import pdb; pdb.set_trace()

# combinations.yaml
"""
combinations:
    X_preparer.cleaner.CHAS:
        Cleaner:
            - date:
                - p1
                - p2
            - financial
        IntentMapper:
            - Something

    X_preparer.cleaner.CHAS.CleanerMapper:
        -Something

    X_preparer.cleaner.CHAS.IntentMapper:
        -Something


    X_preparer:
        cleaner:
            CHAS:
                Cleaner:
                    date:
                        -p1
                        -p2

"""

rscv = RandomizedSearchCV(pipe, param_distributions, iid=True, n_iter=2, scoring='accuracy')

# print("Train Accuracy: {}".format(accuracy_score(y_data, pipe.predict(X_data))))

rscv.fit(X_data, y_data)
results = pd.DataFrame(rscv.cv_results_)
results = results[[c for c in results.columns if all(s not in c for s in ['time', 'params'])]]


import pdb; pdb.set_trace()