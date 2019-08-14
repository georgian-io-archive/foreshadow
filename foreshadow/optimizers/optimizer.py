"""Classes for optimizing Foreshadow given a param_distribution."""

import six
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state
import hyperopt.pyll.stochastic as stoch
from hyperopt import hp
import foreshadow as fs
import foreshadow.serializers as ser


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
                        
                        
Convention:
    Column name is last. If a .<blank> is present, then applied across all 
    columns.

Things that may be swapped:
    PreparerSteps,
    StepSmartTransformers/ConcreteTransformers.

"""


def _replace_list(key, obj, replace_with=hp.choice):
    """Recursively replace a nested object's lists with a sampling function.

    Replaces lists/tuples with replace_with.

    Args:
        key: Current key. Derived from dict keys in nested calls, but should
            be passed if your top level is a list.
        obj: the object to have list/tuples replaced.
        replace_with: Function that takes a key and list and builds a
            sampling function with it. Built around hp.choice but should be
            extendable.

    Returns:
        obj with lists/tuples replaced with replace_with.

    """
    key = str(key)
    if isinstance(obj, (tuple, list)):
        if not isinstance(obj[0], dict):
            #  we have reached a leaf of parameter specifications.
            return replace_with(key, obj)
        else:  # not a leaf, recurse and replace the output.
            to_replace = []
            for v in obj:
                to_replace.append(_replace_list(key, v, replace_with))
            return replace_with(key, to_replace)
    if isinstance(obj, dict):  # not a leaf for sure, we iterate over dict.
        to_replace = {}
        for key, v in obj.items():
            to_replace[key] = _replace_list(key, v, replace_with)
        obj.update(to_replace)
        return obj
    else:  # no nesting and no need to replace.
        return obj


class HyperOptRandomSampler(object):
    def __init__(self, param_distributions, n_iter, random_state=None):
        self.param_distributions = _replace_list(None,
                                                 param_distributions,
                                                 hp.choice)
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


class RandomSearchCV(BaseSearchCV):
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
        out = HyperOptRandomSampler(
            self.param_distributions,
            self.n_iter,
            random_state=self.random_state,
        )
        return out