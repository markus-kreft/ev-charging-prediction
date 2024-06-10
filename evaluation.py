"""Script to evaluate the performance of the models.

For details on the individual figures please refer to the paper.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
import joblib
import datetime as dt
import seaborn as sns
import statsmodels.api as sm

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import shap

from plots import PAGE_WIDTH_COLUMN, PAGE_WIDTH_FULL, GOLDEN_RATIO, savefig
from config import DIR_MODELS, DIR_PLOTS, DIR_RESULTS, CONSUMPTION, FEATURE_NAMES
import metrics


COLOR_MARKER = "C3"
UNIT = "kWh" if CONSUMPTION else "h"
SCORE_NAMES = {
    "mae": f"Mean Absolute Error ({UNIT})",
    "medae": f"Median Absolute Error ({UNIT})",
    "mape": "Mean Absolute Percentage Error (%)",
    "medape": "Median Absolute Percentage Error (%)",
    "mse": f"Root Mean Squared Error ({UNIT})",
}
SCORE_NAMES_SHORT = {
    "mae": f"MAE ({UNIT})",
    "medae": f"MdAE ({UNIT})",
    "mape": "MAPE (%)",
    "medape": "MdAPE (%)",
    "mse": f"RMSE ({UNIT})",
}


def heatmap_data_from_pandas(series, interval_minutes):
    """Gets day/hour matrix from DataFrame."""
    data = series.copy()
    timezone = data.index.tz
    df = data.to_frame(name="values")
    df["date"] = df.index.date
    df["time"] = df.index.time
    data = df.pivot_table(
        index="time",
        columns="date",
        values="values",
        aggfunc=lambda x: x.iloc[0],
        dropna=False,
    )
    daterange = data.columns.astype("datetime64[ns]").tz_localize(timezone)
    daterange = pd.date_range(
        start=daterange.min(), end=daterange.max() + dt.timedelta(days=1), tz=timezone
    )
    timerange = pd.date_range(
        start="1970-01-01T00:00:00",
        end="1970-01-02T00:00:00",
        freq=f"{interval_minutes}min",
        tz=timezone,
    )
    data = data.to_numpy()
    return data, daterange, timerange


def plot_pcolormesh(ax, daterange, timerange, data, **kwargs):
    """Plots a pcolormesh plot with datetime x-axis and time y-axis."""
    mesh = ax.pcolormesh(daterange, timerange, data, **kwargs)
    ax.set_xlim(daterange[0], daterange[-1])
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_minor_locator(mdates.HourLocator())

    ax.set_xlabel("Date")
    ax.set_ylabel("Hour")

    for pos in ["left", "bottom", "top", "right"]:
        ax.spines[pos].set_color("#999999")
    ax.tick_params(axis="both", which="both", color="#999999")
    for pos in ["top", "right"]:
        ax.spines[pos].set_visible(False)

    return mesh


def sci(score: float) -> str:
    """Formats `score` in scientific notation."""
    return f"{float(f'{score:.3g}'):g}"


def plot_and_fit(ax, x, y):
    """Filters NaN values from arrays and fits linear regression."""
    fit_func = lambda x, a, b: a * x + b
    x = x[~pd.isna(y)]
    y = y[~pd.isna(y)]
    try:
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        # print(ols.summary())
        # slope, intercept, r, p, se = linregress(x, y)

        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_smooth,
            fit_func(x_smooth, ols.params.iloc[1], ols.params.iloc[0]),
            # color="C1" if ols.f_pvalue < 0.05 else "gray",
            color="C3" if ols.f_pvalue < 0.05 else "gray",
            label=(
                f"a = {ols.params.iloc[1]:.3f}\n"
                f"b = {ols.params.iloc[0]:.3f}\n"
                f"$r^2$ = {ols.rsquared:.3f}\n"
                f"p = {ols.f_pvalue:.3f}"
            ),
        )
    except Exception as e:
        print(repr(e))
    return x, y


def plot_performance_over_features(results, df):
    """Plots dependence of performance metrics on features with linear fit."""
    score = "medape"
    model = "HistGradientBoostingRegressor"
    dataset = "Test"
    features = [
        {
            "name": "Variation of session duration (IQR / Median)",
            "value": results.apply(
                lambda x: df[df["ParticipantID"] == x.name]["Duration"]
                .quantile([0.1, 0.9])
                .diff()
                .loc[0.9]
                / df[df["ParticipantID"] == x.name]["Duration"].median(),
                axis=1,
            ),
        },
        {
            "name": "Battery size (CarKWh)",
            "value": results.apply(
                lambda x: np.nanmean(df[df["ParticipantID"] == x.name]["CarKWh"]),
                axis=1,
            ),
        },
        {
            "name": "Session start hour (StartHour)",
            "value": results.apply(
                lambda x: np.nanmean(
                    df[df["ParticipantID"] == x.name]["StartHourNorm"]
                ),
                axis=1,
            )
            * 24,
        },
    ]
    fig, ax = plt.subplots(
        1,
        len(features),
        figsize=(PAGE_WIDTH_FULL, PAGE_WIDTH_FULL / GOLDEN_RATIO / 2),
        sharey=True,
        tight_layout=True,
    )
    ax[0].set_ylabel(
        f"{'Charged energy' if CONSUMPTION else 'Session duration'}\n"
        + SCORE_NAMES_SHORT[score]
    )

    titles = ["(a)", "(b)", "(c)"] if not CONSUMPTION else ["(d)", "(e)", "(f)"]

    for i, feature in enumerate(features):
        if i == 2:
            ax[i].set_xlim(10, 22)
        ax[i].set_title(titles[i], weight="bold")
        name = feature["name"]
        x = feature["value"]
        y = results.loc[:, (model, dataset, score)]
        plot_and_fit(ax[i], x, y)
        ax[i].scatter(x, y, s=1)
        ax[i].set_xlabel(name)
        ax[i].set_ylim(0, 150)
        sns.kdeplot(x=x, y=y, ax=ax[i], color="grey", linewidths=0.5, levels=4)
        ax[i].legend(loc="upper left")

    savefig(fig, f"{DIR_PLOTS}/performance_over_features")


def plot_heatmaps(results, predictions):
    """Plots heatmaps of charging timeseries for best and worst vehicles."""
    score = "mae"
    model = "HistGradientBoostingRegressor"
    n = 3
    resolution = "15min"
    dataset = "Test"

    predictions["end_predicted"] = predictions["StartTime"] + pd.to_timedelta(
        predictions[model], unit="h"
    )

    performance = results.loc[:, (model, dataset, score)].dropna().sort_values()
    # Prediction plugged in:
    # 0: True Negative
    # 1: False Negative
    # 2: False Positive
    # 3: True Positive
    colormap = ListedColormap(["#ffffff", "C1", "C3", "C0"])
    quantile_names = [
        ["Best", "Second best", "Third best", "Fourth best", "Fifth best"],
        ["Worst", "Second worst", "Third worst", "Fourth worst", "Fifth worst"],
    ]

    for j, (sample, data) in enumerate(
        zip(["best", "worst"], [performance[:n], performance[-n:][::-1]])
    ):
        for i, (pid, score) in enumerate(data.items()):
            sessions = predictions[predictions["ParticipantID"] == pid]
            # make time series from sessions with 4-part encoding of real and predicted state
            start = sessions["StartTime"].min().round(resolution)
            end = sessions[["StopTime", "end_predicted"]].max().max().round(resolution)
            data = pd.Series(data=0, index=pd.date_range(start, end, freq=resolution))
            for _, session in sessions.iterrows():
                # see above
                start_index = session["StartTime"].round(resolution)
                end_index = session["StopTime"].round(resolution)
                end_predicted_index = session["end_predicted"].round(resolution)
                data.loc[start_index:end_index] += 1
                data.loc[start_index:end_predicted_index] += 2
            data.clip(0, 3, inplace=True)

            title = f"{quantile_names[j][i]} prediction performance (MAE={score:.2f}h)"
            data, daterange, timerange = heatmap_data_from_pandas(
                data, interval_minutes=15
            )
            fig, ax = plt.subplots(
                1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
            )
            plot_pcolormesh(
                ax, daterange, timerange, data, cmap=colormap, rasterized=True
            )
            ax.set_title(title)
            ax.tick_params(direction="out", which="both")
            for pos in ["top", "right"]:
                ax.spines[pos].set_visible(True)
            savefig(fig, f"{DIR_PLOTS}/heatmap-{sample}-{i}")


def print_performance_table(results, results_all, dataset="Test"):
    """Prints a table with the performance of all models.

    Table format:
                      all data         |     per vehicle mean      |
           | MAE | MdAE | ... | SMAPE  | MAE | MdAE | ... | SMAPE  |
    model1 |
    model2 |
    """
    scores = ["mae", "medae", "mape", "medape", "smape"]
    scorenames = ["MAE", "MdAE", "MAPE", "MdAPE", "SMAPE"]

    text = ""
    text += "| " + 30 * " " + f" | {'All data':<37} | {'Per vehicle mean':<37} |\n"
    text += f"| {'Model':<30} | "
    text += " | ".join([f"{scorename:>5}" for scorename in scorenames])
    text += " | "
    text += " | ".join([f"{scorename:>5}" for scorename in scorenames])
    text += " |\n"

    results.drop(columns="number_sessions", inplace=True)
    models = results.columns.get_level_values(0).unique()

    for i, model in enumerate(models):
        text += f"| {model:<30} | "
        text += " | ".join(
            [f"{sci(results_all[(model, dataset, score)]):>5}" for score in scores]
        )
        text += " | "
        text += " | ".join(
            [
                f"{sci(results.loc[:, (model, dataset, score)].mean()):>5}"
                for score in scores
            ]
        )
        text += " |\n"
    print(text, end="")


def plot_performance_all(results, results_all):
    """Plot boxplots of performance metrics comparing all models."""
    WIDTH = 0.25
    STEP = WIDTH + 0.1
    scores = ["mae", "mape"]
    dataset = "Test"
    labels = [
        f"Absolute Error ({UNIT})",
        "Absolute Percentage Error (%)",
        f"RMSE ({UNIT})",
    ]
    results.drop(columns="number_sessions", inplace=True)
    models = results.columns.get_level_values(0).unique()

    fig, ax = plt.subplots(
        len(scores),
        1,
        sharex="col",
        figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN * 1.2),
    )

    for i, model in enumerate(models):
        position = i - 0.5 * STEP
        for k, error in enumerate(["ae", "ape"]):
            ax[k].bxp(
                [
                    {
                        "label": model,
                        "whislo": results_all[(model, dataset)][f"wl-{error}"],
                        "q1": results_all[(model, dataset)][f"q1-{error}"],
                        "med": results_all[(model, dataset)][f"q2-{error}"],
                        "q3": results_all[(model, dataset)][f"q3-{error}"],
                        "whishi": results_all[(model, dataset)][f"wh-{error}"],
                        "mean": results_all[(model, dataset)][f"m{error}"],
                        "fliers": [],
                    }
                ],
                positions=[position],
                widths=WIDTH,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
                boxprops={"facecolor": "C0", "edgecolor": "C0"},
                medianprops={"color": COLOR_MARKER},
                meanprops={
                    "markerfacecolor": COLOR_MARKER,
                    "markeredgecolor": "none",
                },
                whiskerprops={"color": "C0"},
                capprops={"color": "C0"},
                zorder=3,
            )
        score = results_all[(model, dataset, "smape")]
        ax[k].plot(
            [position],
            [score],
            marker="x",
            color=COLOR_MARKER,
            zorder=9,
            linewidth=0,
            markersize=5,
            markeredgewidth=1,
        )
        # distribution of means on each vehicle
        for k, (score_name, label) in enumerate(zip(scores, labels)):
            # Note that mean of mean of subsets != mean of all data
            ax[k].boxplot(
                results[(model, dataset, score_name)].dropna(),
                positions=[position + STEP],
                widths=WIDTH,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
                boxprops={"facecolor": "C1", "edgecolor": "C1"},
                medianprops={"color": COLOR_MARKER},
                meanprops={
                    "markerfacecolor": COLOR_MARKER,
                    "markeredgecolor": "none",
                },
                whiskerprops={"color": "C1"},
                capprops={"color": "C1"},
                zorder=3,
            )
    colors = ["C0"]
    legend_labels = ["Error on all sessions"]
    colors.append("C1")
    legend_labels.append("Mean on each vehicle")

    ax[0].legend(
        handles=_make_legend(colors, legend_labels),
        frameon=False,
        labelspacing=0.5,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.02),
    )

    for i, a in enumerate(ax):
        a.set_ylabel(labels[i])
        a.set_ylim(-0.02 * a.get_ylim()[1], None)

    ax[-1].tick_params(axis="x", which="major", pad=2)

    tick_labels = [
        m.removesuffix("Regressor")
        .removesuffix("Regression")
        .replace("HistGradientBoosting", "HGBR")
        for m in models
    ]
    ax[-1].set_xticks(range(len(models)), tick_labels, rotation=15, fontsize=7)
    ax[-1].set_xlim(-0.5, len(models) - 0.5)

    fig.tight_layout()
    savefig(fig, f"{DIR_PLOTS}/performance")


def _make_legend(colors, legend_labels):
    """Creates legend elements for metrics."""
    legend_elements = []
    for color, legend_label in zip(colors, legend_labels):
        legend_elements.append(
            Line2D(
                [],
                [],
                marker="s",
                label=legend_label,
                color=color,
                markerfacecolor=color,
                markeredgewidth=0,
                linewidth=0,
                markersize=8,
            )
        )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="^",
            label="Mean",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=1,
            linewidth=0,
            markersize=5,
        )
    )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="",
            label="Median",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=0,
            linewidth=1,
            markersize=5,
        )
    )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="x",
            label="SMAPE",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=1,
            linewidth=0,
            markersize=5,
        )
    )
    return legend_elements


def best_model(results, predictions):
    """Plots histograms of performance of best model per session and per vehicle."""
    model_name = "HistGradientBoostingRegressor"
    score = "mae"
    score_name = "MAE"
    value = "Duration" if not CONSUMPTION else "ConsumedkWh"
    alpha = 0.8

    preds = predictions[predictions["dataset"] == "Test"]
    results = results.loc[:, (model_name, "Test", score)]
    # Best and worst performing participants
    # print(results.min(), results.max())

    bins = np.arange(0, 15, 0.66)
    separate_evaluations = [True]
    # separate_evaluations = [False, True]:
    for separate_evaluation in separate_evaluations:
        fig, ax = plt.subplots(
            figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
        )
        ae = np.abs((preds[model_name] - preds[value]))
        hist_ae, _ = np.histogram(ae, bins=bins)
        hist_ae = hist_ae / hist_ae.sum()
        ax.bar(
            bins,
            np.append(hist_ae, hist_ae[-1]),
            color="C0",
            label="AE on all sessions",
            alpha=alpha,
            width=bins[1] - bins[0],
            align="edge",
            linewidth=0.5,
            edgecolor="white",
            zorder=0,
        )
        ax.axvline(np.mean(ae), color="C0", linestyle="--", zorder=2)
        ax.axvline(np.median(ae), color="C0", linestyle=":", zorder=2)

        if separate_evaluation:
            dist, _ = np.histogram(results.dropna(), bins=bins)
            dist = dist / dist.sum()
            ax.bar(
                bins,
                np.append(dist, dist[-1]),
                color="C1",
                label=f"{score_name} for each vehicle",
                alpha=alpha,
                width=bins[1] - bins[0],
                align="edge",
                linewidth=0.5,
                edgecolor="white",
                zorder=1,
            )
            ax.axvline(results.mean(), color="C1", linestyle="--", zorder=2)
            ax.axvline(results.median(), color="C1", linestyle=":", zorder=2)

        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], linestyle="--", label="Mean", color="gray"),
            Line2D([], [], linestyle=":", label="Median", color="gray"),
        ]
        labels += ["Mean", "Median"]
        legend = ax.legend(
            handles,
            labels,
            frameon=False,
            labelspacing=0.5,
            handletextpad=0.2,
            loc="upper right",
            bbox_to_anchor=(1.01, 1.02),
            markerfirst=False,
        )
        for t in legend.get_texts():
            t.set_ha("right")
        ax.set_xlabel("Absolute Error (h)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, bins[-1])
        ax.set_ylim(0, None)

        savefig(fig, f"{DIR_PLOTS}/best_model")


def model_agreement(results):
    """Plots differences in per-vehicle MAE between models."""
    dataset = "Test"
    score = "mae"
    models = ["HistGradientBoostingRegressor", "LinearRegression", "QuantileRegressor"]

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )
    axins = ax.inset_axes(
        [0.4, 0.08, 0.57, 0.57],
        xlim=(0, 20),
        ylim=(0, 20),
        xticklabels=[],
        yticklabels=[],
        xticks=[0, 10, 20],
        yticks=[0, 10, 20],
    )
    axins.patch.set_alpha(0.5)
    color = [[], ["C0"], ["C3", "C1"]]
    for i, model_1 in enumerate(models):
        for j, model_2 in enumerate(models):
            if i <= j:
                continue
            mask_nna = (
                results[(model_1, dataset, score)].notna()
                & results[(model_2, dataset, score)].notna()
            )
            x = results.loc[mask_nna, (model_1, dataset, score)]
            y = results.loc[mask_nna, (model_2, dataset, score)]

            m1 = model_1.removesuffix("Regressor").removesuffix("Regression")
            m2 = model_2.removesuffix("Regressor").removesuffix("Regression")
            ax.scatter(
                x,
                y,
                s=2,
                c=color[i][j],
                label=f"{m1} vs {m2}",
                alpha=1,
                linewidth=0,
            )
            axins.scatter(x, y, s=2, c=color[i][j], alpha=1, linewidth=0)
    axins.plot(
        [0, 1],
        [0, 1],
        transform=axins.transAxes,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1)

    ax.plot(
        [0, 1],
        [0, 1],
        transform=ax.transAxes,
        color="gray",
        linestyle="--",
        zorder=0,
        alpha=0.5,
    )
    ax.set_xlabel(f"Model 1 {SCORE_NAMES_SHORT[score]}")
    ax.set_ylabel(f"Model 2 {SCORE_NAMES_SHORT[score]}")
    ax.legend(
        markerscale=3,
        frameon=False,
        labelspacing=0.5,
        handletextpad=0.2,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
    )

    fig.tight_layout()
    savefig(fig, f"{DIR_PLOTS}/model_agreement")


def evaluate_over_time(predictions):
    """Plots relative change in MAE from first to second half of test set."""
    model_name = "HistGradientBoostingRegressor"
    value = "Duration" if not CONSUMPTION else "ConsumedkWh"
    dataset = "Test"

    predictions = predictions[predictions["dataset"] == dataset]
    predictions.sort_values(by="StartTime", inplace=True)
    predictions["error"] = np.abs(
        np.array((predictions[model_name] - predictions[value]))
    )

    data = []
    number_of_sessions = []
    timespan = []
    for pid, df in predictions.groupby("ParticipantID"):
        number_of_sessions.append(len(df))
        timespan.append(
            (df["StartTime"].max() - df["StartTime"].min()).total_seconds() / 3600 / 24
        )
        df = df["error"]
        split = len(df) // 2
        first, last = df.iloc[:split], df.iloc[split:]
        data.append((first.mean() - last.mean()) / df.mean())

    # Mean and std of shift in MAE from first to second half of test set
    # print(np.nanmean(np.array(data)), np.nanstd(np.array(data)))

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )
    bins = np.linspace(-2, 2, 21)
    hist = np.histogram(data, bins=bins)[0]
    ax.bar(
        bins,
        np.append(hist, hist[-1]),
        color="C0",
        label="AE on all sessions",
        width=bins[1] - bins[0],
        align="edge",
        linewidth=0.5,
        edgecolor="white",
        zorder=0,
    )

    ax.set_xlabel("Relative change in MAE from first to second half of test set")
    ax.set_ylabel("Vehicles")

    savefig(fig, f"{DIR_PLOTS}/ae_over_time_hist")


def seasonality(predictions):
    """Plots monthly average MAPE."""

    def get_monthly(ae):
        groups, dates, means = [], [], []
        for date, group in ae.groupby([ae.index.month, ae.index.year], sort=False):
            groups.append(group.values)
            dates.append(dt.datetime(year=date[1], month=date[0], day=1))
            means.append(np.nanmean(group))
        return groups, dates, means

    model_name = "HistGradientBoostingRegressor"
    value = "Duration" if not CONSUMPTION else "ConsumedkWh"

    predictions.set_index("StartTime", inplace=True)
    predictions.sort_index(inplace=True)
    predictions["tmp"] = (
        metrics.absolute_percentage_error(predictions[model_name], predictions[value])
        * 100
    )

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )
    for dataset in ["Train", "Test"]:
        preds = predictions[predictions["dataset"] == dataset]
        ae = preds["tmp"]
        groups, dates, means = get_monthly(ae)
        # from scipy.stats import kruskal
        # print(kruskal(*groups))
        ax.plot(dates, means, zorder=4, label=f"{dataset} data")

    ax.set_ylim(0, 200)
    ax.legend()
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator((6, 12)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))
    ax.grid(which="both", axis="x", linestyle="--", zorder=0)
    savefig(fig, f"{DIR_PLOTS}/seasonality")


def feature_importance(predictions):
    """Plots feature importance using SHAP values."""

    def get_xy(predictions):
        feature_names = FEATURE_NAMES
        X = predictions[feature_names]
        y = predictions["ConsumedkWh"] if CONSUMPTION else predictions["Duration"]
        return X, y, feature_names

    dataset = "Test"
    model_name = "HistGradientBoostingRegressor"
    model = joblib.load(f"{DIR_MODELS}/{model_name}.joblib")

    predictions = predictions[predictions["dataset"] == dataset]
    X, y, feature_names = get_xy(predictions)
    y_range = range(1, len(feature_names) + 1)

    # shap values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    values = shap_values.abs.mean(0).values
    # shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
    # shap.plots.violin(shap_values, features=X, feature_names=X.columns, show=False)
    # shap.plots.bar(shap_values, max_display=len(feature_names), show=False)

    feature_name_map = {
        "weekday": "weekday",
        "weekend": "weekend",
        "holiday": "holiday",
        "month": "month",
        "StartHourNorm": "startHourNorm",
        "StartCos": "startCos",
        "StartSin": "startSin",
        "CarKWh": "carKWh",
        "CarKW": "carKW",
        "TimeSinceLastStop": "timeSinceLastStop",
        "sessionsToday": "sessionsToday",
        "LastDuration": "lastDuration",
        "LastConsumedkWh": "lastConsumedKWh",
    }

    sorted_importances_idx = values.argsort()
    names = X.columns
    sorted_importances_names = names[sorted_importances_idx]

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    importances = values[sorted_importances_idx]
    ax.plot(
        importances,
        y_range,
        marker="x",
        color="C0",
        zorder=9,
        linewidth=0,
        markersize=4,
        markeredgewidth=1,
    )

    # distribution of individual vehicles
    groups = predictions.groupby("ParticipantID")
    importances = np.zeros((len(groups), len(feature_names)))
    print("Getting SHAP for each vehicle")
    for i, (pid, dfp) in enumerate(groups):
        X, y, feature_names = get_xy(predictions[predictions["ParticipantID"] == pid])
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        values = shap_values.abs.mean(0).values
        importances[i] = values

    importances = pd.DataFrame(importances, columns=X.columns)
    importances = importances[sorted_importances_names].rename(columns=feature_name_map)
    importances.plot.box(
        ax=ax,
        vert=False,
        showfliers=False,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "C1", "edgecolor": "C1"},
        medianprops={"color": COLOR_MARKER},
        meanprops={
            "markerfacecolor": COLOR_MARKER,
            "markeredgecolor": "none",
            "markersize": 4,
        },
        whiskerprops={"color": "C1"},
        capprops={"color": "C1"},
    )

    legend_elements = []
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="x",
            label="Mean of all sessions",
            color="C0",
            markerfacecolor="C0",
            markeredgewidth=1,
            linewidth=0,
            markersize=4,
        )
    )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="s",
            label="Mean on each vehicle",
            color="C1",
            markerfacecolor="C1",
            markeredgewidth=0,
            linewidth=0,
            markersize=8,
        )
    )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="^",
            label="Mean",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=1,
            linewidth=0,
            markersize=4,
        )
    )
    legend_elements.append(
        Line2D(
            [],
            [],
            marker="",
            label="Median",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=0,
            linewidth=1,
            markersize=3,
        )
    )
    ax.legend(
        handles=legend_elements,
        frameon=False,
        labelspacing=0.5,
        loc="lower right",
        bbox_to_anchor=(1.01, 0),
    )

    ax.set_xlabel("mean(|SHAP value|)")
    savefig(fig, f"{DIR_PLOTS}/shap_values")


def average_data_in_split(predictions):
    """Prints average amount of data in test and train sets."""
    print("Average data in split")
    for dataset in ["Train", "Test"]:
        pred = predictions[predictions["dataset"] == dataset]
        number_of_sessions = []
        timespan = []
        for pid, df in pred.groupby("ParticipantID"):
            number_of_sessions.append(len(df))
            timespan.append(
                (df["StartTime"].max() - df["StartTime"].min()).total_seconds()
                / 3600
                / 24
            )
        print(f"    {dataset}")
        print(
            "    Session count: ",
            f"{np.mean(np.array(number_of_sessions)):.1f} ± {np.std(np.array(number_of_sessions)):.1f}",
        )
        print(
            "    Timespan: ",
            f"{np.mean(np.array(timespan)):.1f} ± {np.std(np.array(timespan)):.1f}",
        )


if __name__ == "__main__":
    results = pd.read_csv(
        f"{DIR_RESULTS}/performance.csv", index_col=0, header=[0, 1, 2]
    )
    predictions = pd.read_csv(
        f"{DIR_RESULTS}/predictions.csv",
        index_col=0,
        parse_dates=["StartTime", "StopTime"],
    )
    # Display as percentage
    for score in [
        "mape",
        "medape",
        "smape",
        "q1-ape",
        "q2-ape",
        "q3-ape",
        "wl-ape",
        "wh-ape",
    ]:
        results.loc[:, (slice(None), slice(None), score)] *= 100
    # Split out data for predictions for all vehicles
    results_all = results.loc["ALL", :]
    results.drop("ALL", inplace=True)

    plot_performance_all(results.copy(), results_all.copy())
    best_model(results.copy(), predictions.copy())
    plot_performance_over_features(results.copy(), predictions.copy())
    model_agreement(results.copy())
    print_performance_table(results.copy(), results_all.copy())
    feature_importance(predictions.copy())
    average_data_in_split(predictions.copy())
    evaluate_over_time(predictions.copy())
    seasonality(predictions.copy())
    plot_heatmaps(results.copy(), predictions.copy())
