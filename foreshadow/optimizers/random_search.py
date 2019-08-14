"""Random optimization of params."""

import six
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state

import hyperopt.pyll.stochastic as stoch
from hyperopt import hp

from .tuner import _replace_list


class HyperOptRandomSampler(object):
    def __init__(self, param_distributions, n_iter, random_state=None):
        self.param_distributions = _replace_list(
            None, param_distributions, hp.choice
        )
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
