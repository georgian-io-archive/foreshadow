import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Load in the configs from file
x_proc = json.load(open("adult_x_proc_search.json", "r"))
y_proc = json.load(open("adult_y_proc.json", "r"))

# Create the preprocessors
x_processor = fs.Preprocessor(from_json=x_proc)
y_processor = fs.Preprocessor(from_json=y_proc)
model = fs.Foreshadow(
    X_preprocessor=x_processor,
    y_preprocessor=y_processor,
    estimator=LogisticRegression(random_state=RANDOM_SEED),
    optimizer=GridSearchCV,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy = %f" % accuracy_score(y_test, y_pred))

# Serialize the pipeline
x_proc = model.X_preprocessor.serialize()
y_proc = model.y_preprocessor.serialize()

# Write the serialized pipelines to file
json.dump(x_proc, open("adult_x_proc_searched.json", "w"), indent=4)
json.dump(y_proc, open("adult_y_proc_searched.json", "w"), indent=4)
