import os

DIR_DATA = "data"
DIR_DATA_RAW = f"{DIR_DATA}/electric_nation_data"
DIR_PLOTS_GENERAL = f"{DIR_DATA}/plots"

CONSUMPTION = False
# CONSUMPTION = True

DAYS = ''
# DAYS = '-nextday'
# DAYS = '-sameday'

name = f"{'energy' if CONSUMPTION else 'duration'}{DAYS}"
DIR_MODELS = f"{DIR_DATA}/{name}/models"
DIR_PLOTS = f"{DIR_DATA}/{name}/plots"
DIR_RESULTS = f"{DIR_DATA}/{name}"

for directory in [DIR_MODELS, DIR_PLOTS, DIR_RESULTS, DIR_DATA_RAW, DIR_PLOTS_GENERAL]:
    if not os.path.exists(directory):
        os.makedirs(directory)

FEATURE_NAMES = [
    "weekday",
    "weekend",
    "holiday",
    "month",
    "StartCos",
    "StartSin",
    "StartHourNorm",
    "TimeSinceLastStop",
    "sessionsToday",
    "LastDuration",
    "LastConsumedkWh",
    "CarKW",
    "CarKWh",
]
