import pytest


def test_transformer_impute_simple():

    import numpy as np
    import pandas as pd
    from foreshadow.transformers import SimpleImputer

    impute = SimpleImputer()
    df = pd.read_csv("./foreshadow/tests/data/boston_housing.csv")

    data = df[["crim"]]
    impute.fit(data)
