import pytest


@pytest.mark.skip("broken until intents swapped")
def test_subnumeric_is_intent():
    import numpy as np
    import pandas as pd
    from foreshadow.concrete import FinancialIntent

    X = np.array(
        [
            ["0", "1.", "1,000", "-.1", "-0.1", "-0.001", "1,000.10000"],  # US
            ["0", "1,", "1.000", "-,1", "-0,1", "-0,001", "1.000,10000"],  # EU
            [
                "0",
                "1,",
                "1.000",
                "[,1]",
                "(0,1)",
                "[0,001]",
                "(1.000,10000)",
            ],  # ACCT
            [
                "0%",
                "1,%",
                "1.000%",
                "[,1]%",
                "(0,1)%",
                "[0,001]%",
                "(1.000,10000)%",
            ],  # PCT
        ]
    ).T.astype("object")
    nans = np.array([np.nan] * 800).reshape((200, 4)).astype("object")
    X = pd.DataFrame(np.vstack([X, nans]))
    for c in X:
        assert FinancialIntent.is_intent(X[[c]])
