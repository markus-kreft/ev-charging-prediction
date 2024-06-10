import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)

import models
from dataloader import load_features
from metrics import absolute_percentage_error, symmetric_mean_absolute_percentage_error

from config import DIR_MODELS, DIR_RESULTS, DAYS, CONSUMPTION, FEATURE_NAMES


# Ignore noisy warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
np.seterr(divide="ignore", invalid="ignore")

# Initialize random state
RNG = np.random.RandomState(0)


MODELS = [
    {
        "name": "LinearRegression",
        "model": LinearRegression,
        "params": {},
    },
    {
        "name": "QuantileRegressor",
        "model": QuantileRegressor,
        "params": dict(quantile=0.5, alpha=0, solver="highs"),
    },
    {
        "name": "HistGradientBoostingRegressor",
        "model": HistGradientBoostingRegressor,
        "params": {
            "early_stopping": True,
            "learning_rate": 0.1,
            "min_samples_leaf": 100,
            "max_iter": 100,
            "loss": "absolute_error",
            "random_state": RNG,
        },
        # Perform a grid search to find the best hyperparameters
        # "grid_params": [{
        #     'max_iter': [150, 200, 250, 300],
        #     'learning_rate': [0.5, 0.1, 0.02],
        #     'min_samples_leaf': [50, 100, 200, 500]
        # }]
    },
    {
        "name": "MedianValue",
        "model": models.MedianValue,
        "params": {},
    },
    {
        "name": "MeanValue",
        "model": models.MeanValue,
        "params": {},
    },
    {
        "name": "LastValue",
        "model": models.LastConsumedkWh if CONSUMPTION else models.LastDuration,
        "params": {},
    },
]


def train_model(model_config, X_train, y_train, cv, saved):
    """Trains a model with the given configuration and hyperparameters.

    If grid_params are provided, perform a grid search on the training data
    with the given cross-validation.
    """
    path = f"{DIR_MODELS}/{model_config['name']}.joblib"
    if os.path.exists(path) and saved:
        print(" - Loading Model", end="")
        model = joblib.load(path)
        return model

    model = model_config["model"](**model_config["params"])
    if "grid_params" in model_config.keys():
        search_model = GridSearchCV(
            estimator=model,
            param_grid=model_config["grid_params"],
            cv=cv,
            verbose=2,
            n_jobs=-1,
            scoring="neg_median_absolute_error",
        )
        search_model.fit(X_train, y_train)
        model = search_model.best_estimator_

        model = clone(model, safe=False)
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, path)
    return model


def calculate_metrics(true, pred, df, pid, dataset, model):
    """Writes performance metrics to the DataFrame at the specified index."""
    metrics = {
        "medae": median_absolute_error,
        "mae": mean_absolute_error,
        "medape": lambda x, y: np.median(absolute_percentage_error(x, y)),
        "mape": lambda x, y: np.mean(absolute_percentage_error(x, y)),
        "rmse": root_mean_squared_error,
        "smape": symmetric_mean_absolute_percentage_error,
    }
    for metric_name, metric in metrics.items():
        df.loc[pid, (model, dataset, metric_name)] = metric(true, pred)

    ae = np.abs(true - pred)
    ape = np.abs(ae / true)
    for error_name, error in zip(["ae", "ape"], [ae, ape]):
        quantiles = np.quantile(error, [0.25, 0.5, 0.75])
        # Whiskers
        wl = error[error > (quantiles[0] - 1.5 * (quantiles[2] - quantiles[0]))].min()
        wh = error[error < (quantiles[0] + 1.5 * (quantiles[2] - quantiles[0]))].max()
        df.loc[pid, (model, dataset, f"q1-{error_name}")] = quantiles[0]
        df.loc[pid, (model, dataset, f"q2-{error_name}")] = quantiles[1]
        df.loc[pid, (model, dataset, f"q3-{error_name}")] = quantiles[2]
        df.loc[pid, (model, dataset, f"wl-{error_name}")] = wl
        df.loc[pid, (model, dataset, f"wh-{error_name}")] = wh


def experiment(sessions, saved=False):
    """Trains and evaluates models on all sessions in the given DataFrame.

    - Split the data into training and testing sets
    - Train each model on the training dataset
    - Evaluate the model on the testing dataset
    - Calculate performance metrics for each model for all sessions and per participant

    Args:
        df: DataFrame containing the session data
        saved (bool, optional): Use saved models if exist. Defaults to False.

    Returns:
        Tuple of the performance DataFrame and the predictions DataFrame.
    """
    # X still has all columns for later per-participant evaluation
    X = sessions.copy()
    y = sessions["ConsumedkWh"] if CONSUMPTION else sessions["Duration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    # Alternatively: leave out infividual participants for testing
    # test_pids = np.random.choice(X['id'].unique(), size=10, replace=False)
    # X_test = X[X['id'].isin(test_pids)]
    # X_train = X[~X['id'].isin(test_pids)]
    # y_test = y[X['id'].isin(test_pids)]
    # y_train = y[~X['id'].isin(test_pids)]

    # Shuffle before splitting into batches
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Only use actual features for training
    X_train_actual = X_train[FEATURE_NAMES].copy()
    X_test_actual = X_test[FEATURE_NAMES].copy()

    performance = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(
            [
                (model["name"], dataset, score)
                for model in MODELS
                for dataset in ["Train", "Test"]
                for score in ["medape", "medae", "mape", "rmse"]
            ]
        )
    )

    # DataFrame for saving predictions for every session
    predictions = sessions.copy()
    predictions.loc[X_train.index, "dataset"] = "Train"
    predictions.loc[X_test.index, "dataset"] = "Test"

    for i, model_config in enumerate(MODELS):
        print("Starting", model_config["name"], end="")
        model = train_model(model_config, X_train_actual, y_train, cv, saved=saved)

        y_pred_train = model.predict(X_train_actual)
        y_pred_test = model.predict(X_test_actual)
        y_pred_train = pd.Series(index=y_train.index, data=y_pred_train)
        y_pred_test = pd.Series(index=y_test.index, data=y_pred_test)

        predictions.loc[y_pred_train.index, model_config["name"]] = y_pred_train
        predictions.loc[y_pred_test.index, model_config["name"]] = y_pred_test

        for j, (x, true, pred, dataset) in enumerate(
            zip(
                [X_train, X_test],
                [y_train, y_test],
                [y_pred_train, y_pred_test],
                ["Train", "Test"],
            )
        ):
            calculate_metrics(
                true, pred, performance, "ALL", dataset, model_config["name"]
            )
            performance.loc["ALL", ("number_sessions", dataset, "none")] = len(x.index)
            print(
                f' - {dataset} MedAPE {performance.loc["ALL", (model_config["name"], dataset, "medape")]:.3f}',
                end="",
            )
            # Per vehicle evaluation
            for i, (pid, dfp) in enumerate(x.groupby("ParticipantID")):
                calculate_metrics(
                    true.loc[dfp.index],
                    pred.loc[dfp.index],
                    performance,
                    pid,
                    dataset,
                    model_config["name"],
                )
                performance.loc[pid, ("number_sessions", dataset, "none")] = len(
                    dfp.index
                )

        print(" - done")

    return performance, predictions


def main():
    sessions = load_features(reload=False)
    if DAYS == "-nextday":  # Filter for sessions that reach the next day
        sessions = sessions[
            (sessions["StopTime"].dt.date - sessions["StartTime"].dt.date).dt.days > 0
        ]
    elif DAYS == "-sameday":  # Filter for sessions that end on the same day
        sessions = sessions[
            (sessions["StopTime"].dt.date - sessions["StartTime"].dt.date).dt.days == 0
        ]

    print("Number of sessions:", sessions.shape[0])
    print("Unique participants:", sessions["ParticipantID"].nunique())

    performance, predictions = experiment(sessions, saved=True)
    performance.to_csv(f"{DIR_RESULTS}/performance.csv")
    predictions.to_csv(f"{DIR_RESULTS}/predictions.csv")


if __name__ == "__main__":
    main()
