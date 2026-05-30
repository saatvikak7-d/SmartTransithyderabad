import os
import pandas as pd
import joblib

from utils.timetable import merged
from train_model import train_model

MODEL_PATH = "models/delay_predictor.pkl"

# TRAIN MODEL IF IT DOESN'T EXIST
if not os.path.exists(MODEL_PATH):
    train_model()

# LOAD SAVED MODEL
delay_model = joblib.load(MODEL_PATH)

# STATION TRAFFIC STATISTICS
station_frequency = (
    merged["stop_id"]
    .value_counts()
)

# GENERATE DELAY FEATURES
def generate_delay_features(
    row
):

    hour = row["hour"]

    station_load = station_frequency.get(
        row["stop_id"],
        20
    )

    route_speed = row["IN_speed"]

    num_stops = row["IN_numstops"]

    if hour in [7, 8, 9, 17, 18, 19]:

        pct_right_time = 60
        pct_slight_delay = 25
        pct_significant_delay = 10
        pct_cancelled_unknown = 5

    else:

        pct_right_time = 85
        pct_slight_delay = 10
        pct_significant_delay = 3
        pct_cancelled_unknown = 2
    # BUSY STATION
    if station_load > 50:

        pct_right_time -= 10
        pct_slight_delay += 5
    # SLOW ROUTE EFFECT
    if route_speed < 30:

        pct_significant_delay += 3
    # LONG ROUTE EFFECT
    if num_stops > 15:

        pct_slight_delay += 3

    # PREVENT INVALID PERCENTAGES
    pct_right_time = max(
        pct_right_time,
        5
    )

    pct_slight_delay = min(
        pct_slight_delay,
        90
    )


    # RETURN MODEL FEATURES

    return pd.DataFrame({

        "pct_right_time": [
            pct_right_time
        ],

        "pct_slight_delay": [
            pct_slight_delay
        ],

        "pct_significant_delay": [
            pct_significant_delay
        ],

        "pct_cancelled_unknown": [
            pct_cancelled_unknown
        ]

    })

# =========================================================
# PREDICT DELAY
# =========================================================

def predict_delay(
    row
):


    features = generate_delay_features(
        row
    )

    predicted_delay = delay_model.predict(
        features
    )[0]

    # prevent negative delay
    predicted_delay = max(
        predicted_delay,
        0
    )

    return predicted_delay
