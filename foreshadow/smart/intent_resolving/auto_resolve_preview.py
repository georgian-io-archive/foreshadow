import pandas as pd
from sklearn.datasets import load_breast_cancer
from smart.intent_resolving.core import IntentResolver as AutoIntentResolver


cancer = load_breast_cancer()
cancerX_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

resolver = AutoIntentResolver(cancerX_df)
result = resolver.predict()
print(result)
