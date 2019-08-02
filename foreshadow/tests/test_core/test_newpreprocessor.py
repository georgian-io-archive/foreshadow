"""Tests for the foreshadow foreshadow preprocessor step."""

import pytest


@pytest.mark.skip("This is waiting on a patch to the Base class")
def test_preprocessor_none_config(mocker):
    """Tests that a config step can be None.

    Args:
        mocker: A pytest-mocker instance

    """
    import numpy as np
    import pandas as pd
    from foreshadow.core.column_sharer import ColumnSharer
    from foreshadow.core.newpreprocessor import Preprocessor

    from sklearn.base import BaseEstimator, TransformerMixin

    class DummyIntent(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None):
            return X

    dummy_config = {
        "cleaner": [],
        "resolver": [DummyIntent],
        "DummyIntent": {"preprocessor": None},
    }

    mocker.patch(
        "foreshadow.core.newpreprocessor.resolve_config",
        return_value=dummy_config,
        create=True,
    )
    mocker.patch(
        "foreshadow.core.newpreprocessor.Resolver.pick_transformer",
        return_value=DummyIntent(),
        create=True,
    )

    data = pd.DataFrame({"financials": np.arange(10)})
    cs = ColumnSharer()
    p = Preprocessor(cs)

    p.fit(data)
    _ = p.transform(data)

    # import pdb; pdb.set_trace()


def test_preprocessor_numbers(mocker):
    """Test a standard work flow with preprocessor.

    Args:
        mocker: A pytest-mocker instance

    """
    import numpy as np
    import pandas as pd
    from foreshadow.core.column_sharer import ColumnSharer
    from foreshadow.core.newpreprocessor import Preprocessor

    from sklearn.base import BaseEstimator, TransformerMixin

    from foreshadow.transformers.concrete import StandardScaler

    class DummyIntent(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None):
            return X

    dummy_config = {
        "cleaner": [],
        "resolver": [DummyIntent],
        "DummyIntent": {"preprocessor": [StandardScaler]},
    }

    mocker.patch(
        "foreshadow.core.newpreprocessor.resolve_config",
        return_value=dummy_config,
        create=True,
    )
    mocker.patch(
        "foreshadow.core.newpreprocessor.Resolver.pick_transformer",
        return_value=DummyIntent(),
        create=True,
    )

    data = pd.DataFrame({"financials": np.arange(10)})
    cs = ColumnSharer()
    p = Preprocessor(cs)
    p = p.fit(data)
    tf_data = p.transform(data)

    validate = pd.DataFrame(
        {
            "financials": [
                -1.5666989036012806,
                -1.2185435916898848,
                -0.8703882797784892,
                -0.5222329678670935,
                -0.17407765595569785,
                0.17407765595569785,
                0.5222329678670935,
                0.8703882797784892,
                1.2185435916898848,
                1.5666989036012806,
            ]
        }
    )

    assert (tf_data == validate).squeeze().all()
