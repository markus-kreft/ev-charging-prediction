import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.colors as colors

from dataloader import time_of_day_in_hours


PAGE_WIDTH_FULL = 6.84386  # inches
PAGE_WIDTH_COLUMN = 3.29736  # inches
GOLDEN_RATIO = 1.618


def savefig(fig: mpl.Figure, name: str):
    # fig.savefig(f'{name}.pdf')
    fig.savefig(f"{name}.png")
    plt.close(fig)


def _durations(sessions: pd.DataFrame):
    return (sessions.end - sessions.start).dt.total_seconds() / 3600


def data_availability(sessions: pd.DataFrame, name: str = "") -> None:
    """Plots data availability statistics.

    Args:
        sessions: DataFrame with columns start, end, and id
        name: Prefix for the saved plot
    """
    fig, ax = plt.subplots(
        1,
        4,
        figsize=(PAGE_WIDTH_FULL, PAGE_WIDTH_FULL / GOLDEN_RATIO / 2 / 1.2),
        tight_layout=True,
    )
    titles = ["({})".format(c) for c in "abcd"]
    for i, title in enumerate(titles):
        ax[i].set_title(title, weight="bold")

    # statistics
    duration_mean, number, span, sessions_per_week = [], [], [], []
    for i, (pid, dfp) in enumerate(sessions.groupby("id")):
        span.append((dfp["end"].max() - dfp["start"].min()).days)
        number.append(len(dfp.index))
        durations = (dfp.end - dfp.start).dt.total_seconds() / 3600
        duration_mean.append(durations.mean())
        sessions_per_week.append(len(dfp.index) / span[-1] * 7)
    print("Data availability statistics:")
    print("Average days per vehicle:", np.mean(np.array(span)))
    print("Average sessions per vehicle:", np.mean(np.array(number)))
    print(
        "Average sessions per week per vehicle:", np.mean(np.array(sessions_per_week))
    )
    print("Std sessions per week per vehicle:", np.std(np.array(sessions_per_week)))
    print("Average session duration per vehicle:", np.mean(np.array(duration_mean)))
    print("90% quantile session duration:", np.quantile(np.array(duration_mean), 0.9))

    color = "C0"
    ax[0].set_ylabel("Vehicles")
    ax[0].hist(span, color=color, bins=26, zorder=1, edgecolor="white", linewidth=0.5)
    ax[0].set_xlabel("Data availability span (days)")
    ax[0].set_xlim(0, None)

    ax[1].hist(number, color=color, bins=26, zorder=1, edgecolor="white", linewidth=0.5)
    ax[1].set_xlabel("Sessions")
    ax[1].set_xlim(0, None)

    ax[2].hist(
        sessions_per_week,
        color=color,
        bins=26,
        zorder=1,
        edgecolor="white",
        linewidth=0.5,
    )
    ax[2].set_xlabel("Mean weekly sessions")
    ax[2].set_xlim(0, None)

    ax[3].hist(
        duration_mean,
        color=color,
        bins=np.linspace(0, 72, 26),
        zorder=1,
        edgecolor="white",
        linewidth=0.5,
    )
    ax[3].set_xlabel("Mean session duration (h)")
    ax[3].set_xlim(0, None)

    savefig(fig, f"{name}data_availability")


def target_over_hour(sessions: pd.DataFrame, which: str, name: str = "") -> None:
    """Plots target variable over time of day of session start.

    Plots a 2d heatmap of target variable over time of day of session start with
    marginal histograms.

    Args:
        sessions: DataFrame with columns 'start', 'end', and target variable
        which: Which target variable to plot. Must be 'duration' or 'energy'.
        name: Prefix for the saved plot
    """
    fig = plt.figure(figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(5, 1),
        height_ratios=(1, 5),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.02,
        hspace=0.02,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_cbar_tmp = fig.add_subplot(gs[0, 1])
    ax_cbar_tmp.axis("off")
    ax_cbar = ax_cbar_tmp.inset_axes((0, 0, 0.2, 1))
    ax_cbar.yaxis.set_tick_params(direction="out", which="both")
    ax.tick_params(direction="out", which="both")

    if which == "duration":
        y = _durations(sessions)
        cutoff = 72
        ax.set_ylabel("Duration (hours)")
    elif which == "energy":
        y = sessions.ConsumedkWh
        cutoff = 75
        ax.set_ylabel("Energy (kWh)")
    else:
        raise ValueError(f"Unknown target variable: {which}")

    x = time_of_day_in_hours(sessions.start.dt)
    bins_x = np.arange(0, 24.5, 0.5)
    bins_y = np.arange(0, cutoff + 0.5, 0.5)
    H, xedges, yedges = np.histogram2d(y, x, bins=(bins_y, bins_x))
    X, Y = np.meshgrid(yedges, xedges)
    mesh = ax.pcolormesh(
        X,
        Y,
        H,
        rasterized=True,
        norm=colors.LogNorm(clip=True),
    )

    ax.set_xlabel("Start hour")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, cutoff)

    ax_histy.fill_betweenx(
        bins_y[:-1] + 0.5 * (bins_y[1] - bins_y[0]),
        np.histogram(y, bins=bins_y)[0],
        facecolor="#aaaaaa",
        alpha=0.5,
        zorder=10,
    )
    ax_histy.set_ylim(0, cutoff)
    ax_histy.set_xlabel("Sessions")
    ax_histy.set_xscale("log")
    ax_histy.tick_params(axis="y", which="both", labelleft=False, left=False)
    ax_histy.set_xticks([1e1, 1e3])
    ax_histy.minorticks_off()

    ax_histx.fill_between(
        bins_x[:-1] + 0.5 * (bins_x[1] - bins_x[0]),
        H.sum(axis=0),
        facecolor="#aaaaaa",
        alpha=0.5,
        zorder=10,
    )
    ax_histx.set_xlim(0, 24)
    ax_histx.set_ylim(0, None)
    ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_histx.set_ylabel("Sessions")
    ax_histx.tick_params(axis="x", which="both", labelbottom=False, bottom=False)

    ax_histx.spines[["right", "top"]].set_visible(False)
    ax_histy.spines[["right", "top"]].set_visible(False)

    fig.colorbar(mesh, cax=ax_cbar, label="Sessions")

    # Create offset transform and apply to all ticklabels.
    offset = mpl.transforms.ScaledTranslation(
        xt=0 / 72, yt=-2 / 72, scale_trans=fig.dpi_scale_trans
    )
    for label in ax_cbar.yaxis.get_ticklabels():
        label.set_verticalalignment("bottom")
        label.set_transform(label.get_transform() + offset)

    savefig(fig, f"{name}{which}_over_hour")


def ratio_charging_pluggedin(sessions: pd.DataFrame, name: str = "") -> None:
    """Plots distribution of charging to pluggedin duration ratio.

    Separate histograms for overnight and same day sessions in one plot.

    Args:
        sessions: DataFrame with columns start, end, ChargingDuration
        name: Prefix for the saved plot
    """

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO),
        tight_layout=True,
    )

    mask_sameday = sessions["start"].dt.date == sessions["end"].dt.date
    sameday = sessions[mask_sameday]
    overnight = sessions[~mask_sameday]

    bins = np.linspace(0, 1, 26)
    for i, sessions in enumerate([sameday, overnight]):
        duration_pluggedin = _durations(sessions)
        duration_charging = sessions["ChargingDuration"] / 60
        na = duration_charging.isna()
        ratio = duration_charging[~na] / duration_pluggedin[~na]

        hist, _ = np.histogram(ratio, bins=bins)
        hist = hist / hist.sum() * 100
        ax.bar(
            bins * 100,
            np.append(hist, hist[-1]),
            label="Same day" if i == 0 else "Overnight",
            alpha=0.8,
            width=(bins[1] - bins[0]) * 100,
            align="edge",
            linewidth=0.5,
            edgecolor="white",
        )
    ax.legend(
        frameon=False,
        labelspacing=0.5,
        handletextpad=0.2,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
    )

    ax.set_xlim(0, 100)
    ax.set_ylabel("Probability mass function (%)")
    ax.set_xlabel(r"$T_\mathrm{charging}$ / $T_\mathrm{pluggedin}$ (%)")

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))

    savefig(fig, f"{name}ratio_charging_pluggedin")


if __name__ == "__main__":
    from dataloader import load_features
    from config import DIR_PLOTS_GENERAL

    sessions = load_features(reload=False)
    # Convert times to local timezone for plotting
    sessions["start"] = sessions.StartTime.dt.tz_convert("Europe/London")
    sessions["end"] = sessions.StopTime.dt.tz_convert("Europe/London")

    data_availability(sessions, name=DIR_PLOTS_GENERAL + "/")
    ratio_charging_pluggedin(sessions, name=DIR_PLOTS_GENERAL + "/")
    target_over_hour(sessions, which="duration", name=DIR_PLOTS_GENERAL + "/")
    target_over_hour(sessions, which="energy", name=DIR_PLOTS_GENERAL + "/")
