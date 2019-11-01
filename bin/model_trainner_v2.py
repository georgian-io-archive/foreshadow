#!/usr/bin/env python

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from foreshadow.config import config
from foreshadow.estimators import AutoEstimator
from foreshadow.foreshadow import Foreshadow


TRAINING_DATA_FOLDER_PATH = "YOUR_TRAIN_DATA_FOLDER_PATH"
MODEL_FOLDER_PATH = "YOUR_TRAINED_MODEL_FOLDER_PATH"
PREDICTION_FOLDER_PATH = "YOUR_FINAL_PREDICTION_FOLDER_PATH"

TARGET = "label"


def load_individual_file(filepath):
    try:
        df = pd.read_csv(filepath)
        del df["SymbolId"]
        del df["SequenceTime"]
        return df
    except Exception:
        raise ValueError(
            "Failed to load file. Please verify it exists and is a valid CSV."
        )


def load_data_per_file(folder):
    files_to_load = get_data_filepath_from_folder(folder)
    # Assuming the test data is formated as {ticker}_test.csv
    return [(load_individual_file(file), file.split("_")[0]) for file in
            files_to_load]


def get_data_filepath_from_folder(folder):

    return [os.path.join(folder, filename) for filename in os.listdir(folder)]


def split_train_test_df(df, test_size=0.2, shuffle=False, stratify=None):
    try:
        X_df = df.drop(columns=TARGET)
        y_df = df[[TARGET]]
    except Exception:
        raise ValueError("Invalid target variable")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=test_size, shuffle=shuffle, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def construct_foreshadow(
    estimator=RandomForestClassifier(n_jobs=10), multiprocess=False, auto=False
):
    if auto:
        estimator = AutoEstimator(problem_type="classification", auto="tpot")
    if multiprocess:
        config.set_multiprocess(True)
        print("multiprocessing enabled.")
    return Foreshadow(estimator=estimator)


def train_model(X_train, y_train, multiprocess=False, auto=False):
    fs = construct_foreshadow(multiprocess=multiprocess, auto=auto)
    fs.fit(X_train, y_train)
    return fs


def evaluate_model(X_test, y_test, model):
    y_scores = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_test, y_scores)
    return auc


def save_models(models, tickers, folder=MODEL_FOLDER_PATH):
    import pickle

    for model, ticker in zip(models, tickers):
        pickle.dump(model, open(os.path.join(folder, ticker) + ".p", "wb"))


def load_model(ticker, folder=MODEL_FOLDER_PATH):
    import pickle

    return pickle.load(open(os.path.join(folder, ticker), "rb"))


def predict(model, X_test):
    return model.predict(X_test)


if __name__ == "__main__":
    filename = "YOUR_TRAINING_DATA"
    df = load_individual_file(
        os.path.join(TRAINING_DATA_FOLDER_PATH, filename)
    )
    # If we are not considering the time series factor, then
    # shuffle should be True
    X_train, X_test, y_train, y_test = split_train_test_df(
        df, test_size=0.2, shuffle=False, stratify=None
    )
    # multiprocess may or may not work. If it is stuck, set it
    # back to False
    fs = train_model(X_train, y_train, multiprocess=True)
    save_models([fs], ["YOUR_MODEL_NAME"])
    auc = evaluate_model(X_test, y_test, fs)
    print("Final AUC: {}".format(str(auc)))

    # Load the prediction data from test folder
    # a list: [(df, ticker1), (df, ticker2), ..., (df, tickerN)]
    to_predict_per_ticker = load_data_per_file(PREDICTION_FOLDER_PATH)
    for df, ticker in to_predict_per_ticker:
        pred = fs.predict(df)
        pred_filename = os.path.join(PREDICTION_FOLDER_PATH,
                                     "foreshadow_predictions",
                                     "_".join([ticker, "pred.csv"]))
        pred.to_csv(pred_filename, index=False, header=False)



