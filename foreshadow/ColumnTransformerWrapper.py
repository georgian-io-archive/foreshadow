"""Extension of the ColumnTransformer class in Sklearn."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_array


class ColumnTransformerWrapper(ColumnTransformer):
    """See the Docstring in parent class."""

    def _hstack(self, Xs):
        """Stacks Xs horizontally. # noqa DAR201

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Args:
            Xs: List of numpy arrays, sparse arrays, or DataFrames

        Returns:
            the stacked data. It could be a numpy array, sparse array
        or a DataFrame.

        Raises:
            ValueError: # noqa S001
                If not all columns in a sparse out are numeric or
            numeric-convertible.

        """
        # TODO check how adding a text transformer (TFIDF) could affect this
        #  logic.
        if self.sparse_output_:
            try:
                # since all columns should be numeric before stacking them
                # in a sparse matrix, `check_array` is used for the
                # dtype conversion if necessary.
                converted_Xs = [
                    check_array(X, accept_sparse=True, force_all_finite=False)
                    for X in Xs
                ]
            except ValueError:
                raise ValueError(
                    "For a sparse output, all columns should"
                    " be a numeric or convertible to a numeric."
                )

            return sparse.hstack(converted_Xs).tocsr()
        else:
            # TODO In theory, all Xs in Foreshadow are dataframes. However,
            #  this may change once we add Text transformations. The following
            #  code is refactored due to performance concerns of using 2 loops
            #  and easier debugging.
            all_df = True
            for ind, f in enumerate(Xs):
                if not isinstance(f, pd.DataFrame):
                    all_df = False
                if sparse.issparse(f):
                    Xs[ind] = f.toarray()

            if all_df:
                return pd.concat(Xs, axis=1)
            else:
                np.hstack(Xs)
            # Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
            # if all([isinstance(item, pd.DataFrame) for item in Xs]):
            #     return pd.concat(Xs, axis=1)
            # return np.hstack(Xs)
