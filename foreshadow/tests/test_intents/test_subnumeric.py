def test_subnumeric_is_intent():
    import pandas as pd
    from foreshadow.intents import FinancialIntent

    X = pd.DataFrame(
        {
            "US": ["0", "1.", "1,000", "-.1", "-0.1", "-0.001", "1,000.10000"],
            "EU": ["0", "1,", "1.000", "-,1", "-0,1", "-0,001", "1.000,10000"],
            "ACCT": ["0", "1,", "1.000", "[,1]", "(0,1)", "[0,001]", "(1.000,10000)"],
            "PCT": [
                "0%",
                "1,%",
                "1.000%",
                "[,1]%",
                "(0,1)%",
                "[0,001]%",
                "(1.000,10000)%",
            ],
        }
    )
    for c in X:
        assert FinancialIntent.is_intent(X[[c]])
