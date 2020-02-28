import numpy as np
import pandas as pd

from foreshadow.concrete import NaNFiller
from foreshadow.utils import Constant


def test_na_filler():
    fill_value = Constant.NAN_FILL_VALUE
    filler = NaNFiller(fill_value)

    X = pd.DataFrame({"a": [np.nan, np.nan, 1, 2, 3]})
    Xt = filler.fit(X).transform(X)

    check = pd.DataFrame(
        {"a": [Constant.NAN_FILL_VALUE, Constant.NAN_FILL_VALUE, 1, 2, 3]}
    )
    assert Xt.equals(check)
