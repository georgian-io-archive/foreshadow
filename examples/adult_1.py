import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import foreshadow as fs
import json

RANDOM_SEED = 42
adult = pd.read_csv("adult.csv").iloc[:1000]

print(adult.head())
features = adult.drop(columns="class")
target = adult[["class"]]
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=RANDOM_SEED
)

model = fs.Foreshadow(estimator=LogisticRegression(random_state=RANDOM_SEED))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy = %f" % accuracy_score(y_test, y_pred))

# Serialize the pipeline
x_proc = model.X_preprocessor.serialize()
y_proc = model.y_preprocessor.serialize()

# Write the serialized pipelines to file
json.dump(x_proc, open("adult_x_proc.json", "w"), indent=4)
json.dump(y_proc, open("adult_y_proc.json", "w"), indent=4)

summary = {
    "x_summary": model.X_preprocessor.summarize(X_train),
    "y_summary": model.y_preprocessor.summarize(y_train),
}

json.dump(summary, open("adult_summary.json", "w"), indent=4)
