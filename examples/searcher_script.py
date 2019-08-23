import hyperopt.pyll.stochastic as stoch
import pandas as pd
import six
import sklearn.datasets as dt
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

import foreshadow
from foreshadow.optimizers import ParamSpec, RandomSearchCV, Tuner
from foreshadow.smart import Scaler
from foreshadow.utils.testing import debug


debug()

data = dt.load_iris()

X_data = pd.DataFrame(data.data, columns=data.feature_names).iloc[:, 0]
y_data = pd.DataFrame(data.target, columns=["target"])["target"]

# cs = ColumnSharer()
# p = Preprocessor(column_sharer=cs)
s = Scaler()
lr = LogisticRegression()

pipe = Pipeline([("s", s), ("lr", lr)])

pipe.fit(X_data, y_data)

param_distributions = hp.choice(
    "s__transformer",
    [
        {
            "s__transformer": "StandardScaler",
            "s__transformer__with_mean": hp.choice("with_mean", [False, True]),
        },
        {
            "s__transformer": "MinMaxScaler",
            "s__transformer__feature_range": hp.choice(
                "feature_range", [(0, 1), (0, 0.5)]
            ),
        },
    ],
)

test = [
    {
        "s__transformer": "StandardScaler",
        "s__transformer__with_mean": [False, True],
    },
    {
        "s__transformer": "MinMaxScaler",
        "s__transformer__feature_range": [(0, 1), (0, 0.5)],
    },
]


class HyperOptSampler(object):
    def __init__(self, param_distributions, n_iter, random_state=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        # check if all distributions are given as lists
        # in this case we want to sample without replacement
        rng = check_random_state(self.random_state)
        for _ in six.moves.range(self.n_iter):
            # import pdb; pdb.set_trace()
            yield stoch.sample(self.param_distributions, rng=rng)

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


class ShadowSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        scoring=None,
        fit_params=None,
        n_jobs=1,
        iid=True,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score="raise",
        return_train_score="warn",
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        out = HyperOptSampler(
            self.param_distributions,
            self.n_iter,
            random_state=self.random_state,
        )
        return out


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

rscv = ShadowSearchCV(
    pipe, param_distributions, iid=True, scoring="accuracy", n_iter=10
)

rscv.fit(X_data, y_data)
results = pd.DataFrame(rscv.cv_results_)
results = results[
    [c for c in results.columns if all(s not in c for s in ["time", "params"])]
]

print(rscv.best_params_)
print(rscv.best_estimator_)
# print(results)

###############

print("simpletest")
ps = ParamSpec()
test = [
    {
        "s__transformer": "StandardScaler",
        "s__transformer__with_mean": [False, True],
    },
    {
        "s__transformer": "MinMaxScaler",
        "s__transformer__feature_range": [(0, 1), (0, 0.5)],
    },
]
ps.set_params(param_distributions=test)
t = Tuner(
    pipe,
    ps,
    RandomSearchCV,
    optimizer_kwargs={
        "iid": True,
        "scoring": "accuracy",
        "n_iter": 2,
        "return_train_score": True,
    },
)
t.fit(X_data, y_data)
print(t.best_pipeline)

###############

print("foreshadow")

t = {
    "iid": True,
    "scoring": "accuracy",
    "n_iter": 2,
    "return_train_score": True,
}

fs = foreshadow.Foreshadow(
    optimizer=RandomSearchCV, optimizer_kwargs=t, estimator=lr
)
fs.fit(X_data, y_data)
