import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from foreshadow import Foreshadow
from foreshadow.utils import ProblemType


pd.options.display.max_columns = None
pd.options.display.width = 200
np.random.seed(42)

# Load the data
titanic = pd.read_csv("data/titanic_train.csv")
titanic.head(5)

X_train, X_test, y_train, y_test = train_test_split(
    titanic.drop("Survived", axis=1),
    titanic["Survived"],
    train_size=0.8,
    test_size=0.2,
)

shadow = Foreshadow(
    problem_type=ProblemType.CLASSIFICATION,
    random_state=42,
    n_jobs=-1,
    estimator=LogisticRegression(),
)
shadow.fit(X_train, y_train)
print(shadow.score(X_test, y_test))

summary = shadow.get_data_summary()

print(summary)
