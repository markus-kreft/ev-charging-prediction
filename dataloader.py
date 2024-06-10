import os
import logging
import pandas as pd
import numpy as np

from config import DIR_DATA, DIR_DATA_RAW

COLUMNS_UTC = ["StartTime", "StopTime"]


def get_holidays() -> pd.Series:
    """Makes a DataFrame with holidays in the UK for the years 2017 and 2018.

    Returns:
        DataFrame with the holidays
    """
    holidays = pd.Series([x[0] for x in [
        ("01-Jan-2017", "Sunday", "New Year's Day"),
        ("02-Jan-2017", "Monday", "New Year's Day observed"),
        ("14-Apr-2017", "Friday", "Good Friday"),
        ("01-May-2017", "Monday", "Early May Bank Holiday"),
        ("29-May-2017", "Monday", "Spring Bank Holiday"),
        ("25-Dec-2017", "Monday", "Christmas Day"),
        ("26-Dec-2017", "Tuesday", "Boxing Day"),
        ("01-Jan-2018", "Monday", "New Year's Day"),
        ("30-Mar-2018", "Friday", "Good Friday"),
        ("07-May-2018", "Monday", "Early May Bank Holiday"),
        ("28-May-2018", "Monday", "Spring Bank Holiday"),
        ("25-Dec-2018", "Tuesday", "Christmas Day"),
        ("26-Dec-2018", "Wednesday", "Boxing Day"),
    ]])
    holidays = (
        pd.to_datetime(holidays, format="%d-%b-%Y")
        .astype("datetime64[s]")
        .dt.tz_localize("UTC")
        .dt.date
    )
    return holidays


def load_data() -> pd.DataFrame:
    """Loads the `CrowdCharge Transactions.xlsx` file.

    Timestamps are localized to UTC

    Returns:
        DataFrame with the raw data
    """
    df = pd.read_excel(
        os.path.join(DIR_DATA_RAW, "CrowdCharge Transactions.xlsx"),
        sheet_name="Transaction Data",
    )
    for column in COLUMNS_UTC + ["ActiveCharging_Start", "EndCharge"]:
        df[column] = (
            pd.to_datetime(df[column], format="%m%d%y %H:%M")
            .astype("datetime64[s]")
            .dt.tz_localize("UTC")
        )
    return df


def load_metadata() -> pd.DataFrame:
    """Loads the `ChargerInstall.xlsx` file.

    Returns:
        DataFrame with the metadata
    """
    df = pd.read_excel(
        os.path.join(DIR_DATA_RAW, "ChargerInstall.xlsx"),
        sheet_name="20181126_ChargerInstall_DriveEl",
        index_col="ParticipantID",
    )
    return df


def time_of_day_in_hours(
    dt: pd.core.indexes.accessors.DatetimeProperties,
) -> "pd.Series[float]":
    """Calculates the time of day in hours from the datetime properties.

    Args:
        dt: Datetime object

    Returns:
        Series with time of day in hours
    """
    return dt.hour + dt.minute / 60 + dt.second / 3600


def derive_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derives features from the raw data.

    Args:
        data: Raw data

    Returns:
        New DataFrame with features
    """
    # Keep track of excluded data
    sessions_dropped_inconsistent_data = 0
    vehicles_dropped_low_data = 0
    vehicles_dropped_low_data_density = 0

    logging.info(f"Number of sessions in raw dataset: {len(data.index)}")
    logging.info(
        f"Number of participants in raw dataset: {data['ParticipantID'].nunique()}"
    )

    data = data[
        [
            "StartTime",
            "StopTime",
            "ParticipantID",
            "ConsumedkWh",
            "CarKW",
            "CarKWh",
            "PluggedInTime",
            "ChargingDuration",
            "Hot_Unplug",
        ]
    ]

    # Calculate duration of session in hours
    data.loc[:, "Duration"] = (
        data["StopTime"] - data["StartTime"]
    ).dt.total_seconds() / 3600

    # Filter out short sessions
    lower_threshold = 5 / 60
    mask = data["Duration"] < lower_threshold
    logging.warning(
        f"Filtering out {mask.sum()} sessions with duration <{lower_threshold}h"
    )
    data = data[~mask].reset_index(drop=True)

    # Threshold long sessions. NOTE: do not change StopTime so the time since last charge stays true
    upper_threshold = 2 * 7 * 24
    mask = data["Duration"] > upper_threshold
    logging.warning(
        f"Thresholding {mask.sum()} sessions with duration >{upper_threshold}h"
    )
    data.loc[mask, "Duration"] = upper_threshold

    # Calculate categorical labels of the day based on *plug-in time
    holidays = get_holidays()
    data.loc[:, "weekday"] = 0
    data.loc[:, "weekend"] = 0
    data.loc[:, "holiday"] = 0
    for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        data.loc[:, d] = 0
        data.loc[data["StartTime"].dt.weekday == i, d] = 1
    data.loc[data["StartTime"].dt.weekday.isin([0, 1, 2, 3, 4]), "weekday"] = 1
    data.loc[data["StartTime"].dt.weekday.isin([5, 6]), "weekend"] = 1
    data.loc[data["StartTime"].dt.date.isin(holidays), "holiday"] = 1
    # Remove weekday flag from holidays
    data.loc[data["holiday"] == 1, "weekday"] = 0
    data.loc[data["holiday"] == 1, "weekend"] = 0

    data.loc[:, "month"] = data["StartTime"].dt.month

    # Cyclically encode plugin time
    data.loc[:, "StartHour"] = time_of_day_in_hours(data.StartTime.dt)
    data.loc[:, "StartHourNorm"] = data["StartHour"] / data["StartHour"].max()
    x_norm = 2 * np.pi * data["StartHourNorm"]
    data["StartCos"] = np.cos(x_norm)
    data["StartSin"] = np.sin(x_norm)

    # Features dependent on single vehicle information
    dfps = []
    for i, (name, dfp) in enumerate(data.groupby("ParticipantID")):
        dfp.sort_values(by="StartTime", inplace=True)
        dfp = dfp.copy().reset_index(drop=True)

        # Time since last plugin of this vehicle
        TimeSinceLastStop = (
            dfp["StartTime"] - dfp["StopTime"].shift(1)
        ).dt.total_seconds() / 3600
        dfp["TimeSinceLastStop"] = TimeSinceLastStop
        if (dfp["TimeSinceLastStop"] < 0).sum() != 0:
            logging.warning(
                f"Excluding {(TimeSinceLastStop < 0).sum()} sessions "
                f"for {name} due to inconsistent data"
            )
            sessions_dropped_inconsistent_data += (TimeSinceLastStop < 0).sum()
            # Use not <0 (instead of >=0) to avoid loosing the nan entry in the first row
            dfp = dfp[~(dfp["TimeSinceLastStop"] < 0)]

        mydf = dfp.iloc[1:].copy()

        number_sessions = len(mydf.index)
        if number_sessions < 10:
            logging.warning(
                f"Excluding {name} due to too low data: {number_sessions} sessions"
            )
            vehicles_dropped_low_data += 1
            continue
        observation_duration = (mydf["StopTime"].max() - mydf["StartTime"].min()).days
        if number_sessions < observation_duration / 7:
            vehicles_dropped_low_data_density += 1
            logging.warning(
                f"Excluding {name} due to too low data density: "
                f"{number_sessions} sessions / {observation_duration} days"
            )
            continue

        assert mydf.index[0] == 1, (
            "First row should have index 1 since we drop "
            "the first row that has no previous session"
        )

        # Unique id for each participant
        mydf["id"] = i

        # Number of previous sessions today: https://stackoverflow.com/a/27626699
        tmp = mydf["StartTime"].dt.date
        mydf["sessionsToday"] = tmp.groupby((tmp != tmp.shift()).cumsum()).cumcount()
        mydf["LastDuration"] = dfp["Duration"].shift(1).to_numpy()[1:]

        mydf["LastConsumedkWh"] = dfp["ConsumedkWh"].shift(1).to_numpy()[1:]

        dfps.append(mydf)
    data = pd.concat(dfps)
    data.sort_values(by="StartTime", inplace=True)
    data.reset_index(drop=True, inplace=True)
    logging.warning(
        f"Excluded {sessions_dropped_inconsistent_data} sessions due to inconsistent data"
    )
    logging.warning(
        f"Excluded {vehicles_dropped_low_data} vehicles due to too low data"
    )
    logging.warning(
        f"Excluded {vehicles_dropped_low_data_density} vehicles due to too low data density"
    )
    return data


def load_features(reload: bool = True) -> pd.DataFrame:
    """Loads the processed data.

    Args:
        reload: Reload data and generate features or use cached features. Defaults to True.

    Returns:
        DataFrame with features
    """
    path_features = os.path.join(DIR_DATA, "features.csv")
    if reload or not os.path.exists(path_features):
        data = load_data()
        # Trial 3 has ToU tariff incentive
        data = data[data["Trial"] != 3]
        data = derive_features(data)
        data.to_csv(path_features, index=False)
    else:
        data = pd.read_csv(path_features, parse_dates=COLUMNS_UTC)
    logging.info(f"Number of sessions in processed dataset: {len(data.index)}")
    logging.info(f"Number of participants in processed dataset: {data['id'].nunique()}")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_features(reload=True)

    meta = load_metadata()
    meta = meta[meta["DCSProvider"] == "Crowd Charge"]
    print("Unique car models:", meta["CarModel"].nunique())
    print("Unique car brands:", meta["CarMake"].nunique())
