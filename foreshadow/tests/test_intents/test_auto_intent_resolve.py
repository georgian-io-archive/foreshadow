import pytest


@pytest.mark.skip(
    "Not going to work due to hardcoded path structure in the "
    "AutoML core package pickling. Chris is working on a "
    "different approach"
)
def test_auto_intent_resolve():
    from foreshadow.smart.intent_resolving.core import (
        IntentResolver as AutoIntentResolver,
    )

    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    # cancery_df = pd.DataFrame(cancer.target, columns=["target"])

    resolver = AutoIntentResolver(cancerX_df)
    result = resolver.predict()
    print(result)
