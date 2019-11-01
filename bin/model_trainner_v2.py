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
TEST_DATA_FOLDER_PATH = "YOUR_TEST_DATA_FOLDER_PATH"
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


def get_data_filepath_from_folder(folder):

    return [os.path.join(folder, filename) for filename in os.listdir(folder)]


def train_test_split(df, shuffle=False, stratify=None):
    try:
        X_df = df.drop(columns=TARGET)
        y_df = df[[TARGET]]
    except Exception:
        raise ValueError("Invalid target variable")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2, shuffle=shuffle, stratify=stratify
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
    print("Final AUC: {}".format(str(auc)))
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
    # TODO Version 2, assuming we are handling one file for all, or one for
    #  each exchange.
    filename = "YOUR_TRAINING_DATA"
    df = load_individual_file(
        os.path.join(TRAINING_DATA_FOLDER_PATH, filename)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df, shuffle=False, stratify=None
    )
    fs = train_model(X_train, y_train, multiprocess=False)
    save_models([fs], ["YOUR_MODEL_NAME"])
    evaluate_model(X_test, y_test, fs)
