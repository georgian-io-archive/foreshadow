"""Random optimization of params."""

import hyperopt.pyll.stochastic as stoch
from hyperopt import hp
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state


class HyperOptRandomSampler(object):
    """Sampler that is an iterable over param distribution."""

    def __init__(
        self, param_distributions, n_iter, random_state=None, max_tries=100
    ):
        """Constructor.

        Args:
            param_distributions: Parameter distribution as nested list-dict.
            n_iter: length of returned iterator.
            random_state: random state.
            max_tries: max attempts to try to get a new unique value. If
                None, will not attempt to get unique values.
        """
        param_distributions.convert(None, hp.choice)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.max_tries = max_tries

    def __iter__(self):
        """Search parameter distribution for unique states.

        As each state is defined using hp.choice, we don't explicitly know
        each of the unique states that our estimator can be set to. We
        sample the distribution of states up until max_tries times to get
        these unique states and return an iterable of them. if max_tries is
        None (set in constructor), then we sample the search space and add each
        sampled value.

        Returns:
            iterable of unique states.

        """
        # check if all distributions are given as lists
        # in this case we want to sample without replacement
        rng = check_random_state(self.random_state)
        prev_samples = []
        max_tries = self.max_tries if self.max_tries is not None else 1
        for _ in range(self.n_iter):
            sample = stoch.sample(self.param_distributions(), rng=rng)
            n_tries = 0
            while sample not in prev_samples or n_tries < max_tries:
                if sample not in prev_samples or self.max_tries is None:
                    prev_samples.append(sample)
                    break
                sample = stoch.sample(self.param_distributions(), rng=rng)
                n_tries += 1
        return iter(prev_samples)

    def __len__(self):
        """Get number of sampled points for optimization.

        Returns:
            Number of unique states to be returned.

        """
        return self.n_iter


class RandomSearchCV(BaseSearchCV):
    """Optimize Foreshadow.pipeline and/or its sub-objects."""

    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        scoring=None,
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
        """Return ParameterSampler instance for the given distributions.

        Returns:
            iterable of unique states defined by HyperOptRandomSampler.

        """
        out = HyperOptRandomSampler(
            self.param_distributions,
            self.n_iter,
            random_state=self.random_state,
            max_tries=self.max_tries,
        )
        return out
