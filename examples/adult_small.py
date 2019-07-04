import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import foreshadow as fs


RANDOM_SEED = 42
adult = pd.read_csv("adult_small.csv").iloc[:1000]

print(adult.head())
features = adult.drop(columns="class")
target = adult[["class"]]
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=RANDOM_SEED
)

model = fs.Foreshadow()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy = %f" % accuracy_score(y_test, y_pred))
