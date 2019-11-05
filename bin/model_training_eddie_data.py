#!/usr/bin/env python

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from foreshadow.config import config
from foreshadow.estimators import AutoEstimator
from foreshadow.foreshadow import Foreshadow


TRAINING_DATA_FOLDER_PATH = (
    "/home/ec2-user/hackathon/eddie_data/2019.07.02/{}_labeled.csv.v10"
)
MODEL_FOLDER_PATH = "YOUR_TRAINED_MODEL_FOLDER_PATH"
PREDICTION_FOLDER_PATH = (
    "/home/ec2-user/hackathon/eddie_data/2019.07.30/{}_test.csv.v11"
)

TARGET = "label"

tickers = [
    1006,
    10253,
    # 10260,
    # 1031,
    # 1032,
    # 103,
    # 10466,
    # 104,
    # 1059,
    # 10656
]


def load_all_data(folder_path_template):
    processed_dfs = []
    for ticker in tickers:
        print("loading data from ticker {}".format(str(ticker)))
        ticker_df = load_data_per_ticker(
            folder_path_template.format(str(ticker))
        )
        ticker_df = preprocessing_df_before_feeding_to_foreshadow(ticker_df)
        processed_dfs.append(ticker_df)
    total_df = pd.concat(processed_dfs)
    return total_df


def load_individual_file(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception:
        raise ValueError(
            "Failed to load file. Please verify it exists and is a valid CSV."
        )


def load_data_per_ticker(folder):
    files_to_load = get_data_filepath_from_folder(folder)
    # Assuming the test data is formated as {ticker}_test.csv
    return pd.concat([load_individual_file(file) for file in files_to_load])


def preprocessing_df_before_feeding_to_foreshadow(df):
    del df["id"]
    del df["timestamp"]
    for col in [
        "bid_BATS",
        "bid_BATY",
        "bid_EDGA",
        "bid_EDGX",
        "bid_XBOS",
        "bid_XNGS",
    ]:
        df[f"_{col}_div_best_bid"] = df[col] / df["best_bid"]

    for col in [
        "ask_BATS",
        "ask_BATY",
        "ask_EDGA",
        "ask_EDGX",
        "ask_XBOS",
        "ask_XNGS",
    ]:
        df[f"_{col}_div_best_ask"] = df[col] / df["best_ask"]

    for timing in ["1ms", "2ms", "5ms", "10ms"]:
        for col in [
            "AskBidDiff",
            "bid_BATS",
            "bid_BATY",
            "bid_EDGA",
            "bid_EDGX",
            "bid_XBOS",
            "bid_XNGS",
            "ask_BATS",
            "ask_BATY",
            "ask_EDGA",
            "ask_EDGX",
            "ask_XBOS",
            "ask_XNGS",
            "best_bid",
            "best_ask",
            "total_at_best_ask",
            "total_at_best_bid",
        ]:
            # Replace the values with historical ratios.
            df[f"{col}_{timing}"] = df[col] / df[f"{col}_{timing}"]

    df = df.drop(
        [
            "AskBidDiff",
            "bid_BATS",
            "bid_BATY",
            "bid_EDGA",
            "bid_EDGX",
            "bid_XBOS",
            "bid_XNGS",
            "ask_BATS",
            "ask_BATY",
            "ask_EDGA",
            "ask_EDGX",
            "ask_XBOS",
            "ask_XNGS",
            "best_bid",
            "best_ask",
        ],
        axis=1,
    )
    return df


def get_data_filepath_from_folder(folder):
    filepaths = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            filepaths.append(os.path.join(folder, filename))
    return filepaths


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


def save_models(models, names, folder=MODEL_FOLDER_PATH):
    import pickle

    for model, name in zip(models, names):
        pickle.dump(model, open(os.path.join(folder, name), "wb"))


def load_model(ticker, folder=MODEL_FOLDER_PATH):
    import pickle

    return pickle.load(open(os.path.join(folder, ticker), "rb"))


def predict(model, X_test):
    return model.predict(X_test)


if __name__ == "__main__":
    total_df = load_all_data(TRAINING_DATA_FOLDER_PATH)
    X_train, X_test, y_train, y_test = split_train_test_df(
        total_df, test_size=0.2, shuffle=True, stratify=None
    )
    # multiprocess may or may not work. If it is stuck, set it
    # back to False
    print("Start model training...")
    fs = train_model(X_train, y_train, multiprocess=False, auto=True)
    save_models([fs], ["fs_model_eddie.p"])
    auc = evaluate_model(X_test, y_test, fs)
    print("Final AUC: {}".format(str(auc)))

    # Load the prediction data from test folder
    # a list: [(df, ticker1), (df, ticker2), ..., (df, tickerN)]
    print("Start prediction...")
    for ticker in tickers:
        print("Prediction for ticker {}".format(str(ticker)))
        ticker_df = load_data_per_ticker(
            PREDICTION_FOLDER_PATH.format(str(ticker))
        )
        ticker_df = preprocessing_df_before_feeding_to_foreshadow(ticker_df)
        pred = fs.predict(ticker_df)
        pred_filename = os.path.join(
            "foreshadow_predictions", "_".join([str(ticker), "pred.csv"])
        )
        pred.to_csv(pred_filename, index=False, header=False)
