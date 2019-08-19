"""Random optimization of params."""

import hyperopt.pyll.stochastic as stoch
import six
from hyperopt import hp
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state

from .tuner import _replace_list


class HyperOptRandomSampler(object):
    def __init__(
        self, param_distributions, n_iter, random_state=None, max_tries=100
    ):
        param_distributions = _replace_list(
            None, param_distributions.param_distributions, hp.choice
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.max_tries = max_tries

    def __iter__(self):
        # check if all distributions are given as lists
        # in this case we want to sample without replacement
        rng = check_random_state(self.random_state)
        prev_samples = []
        for _ in six.moves.range(self.n_iter):
            # import pdb; pdb.set_trace()
            sample = stoch.sample(self.param_distributions, rng=rng)
            n_tries = 0
            while sample not in prev_samples and n_tries < self.max_tries:
                if sample not in prev_samples:
                    prev_samples.append(sample)
                    break
                sample = stoch.sample(self.param_distributions, rng=rng)
                n_tries += 1
        return iter(prev_samples)

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
        max_tries=100,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.max_tries = max_tries
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
            max_tries=self.max_tries,
        )
        return out
