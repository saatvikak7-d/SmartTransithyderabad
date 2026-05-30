
import pandas as pd

from utils.timetable import (
    get_next_train,
    get_remaining_stops
)

from utils.delay_model import (
    predict_delay
)

# ADJUST ARRIVAL TIMES

def adjust_arrival_times(
    remaining,
    predicted_delay
):


    remaining = remaining.copy()

    # DELAY GROWS SLIGHTLY ALONG ROUTE

    remaining["delay_multiplier"] = (
        remaining["stop_sequence"]
        - remaining["stop_sequence"].min()
    )

    remaining["adjusted_delay"] = (
        predicted_delay
        + remaining["delay_multiplier"] * 0.5
    )

    # GENERATE PREDICTED ARRIVAL
    remaining["predicted_arrival"] = (
        remaining["arrival_time"]
        + pd.to_timedelta(
            remaining["adjusted_delay"],
            unit="m"
        )
    )

    remaining["predicted_departure"] = (
        remaining["departure_time"]
        + pd.to_timedelta(
            remaining["adjusted_delay"],
            unit="m"
        )
    )

    return remaining

# GENERATE ETA TABLE
def generate_eta_table(
    station,
    current_time
):

    # FIND NEXT TRAIN

    next_train = get_next_train(
        station,
        current_time
    )

    if next_train is None:
        return None

    # GET DOWNSTREAM STOPS

    remaining = get_remaining_stops(
        next_train["trip_id"],
        next_train["stop_sequence"]
    )

    # PREDICT DELAY

    predicted_delay = predict_delay(
        next_train
    )

    # GENERATE ETA TABLE

    eta_table = adjust_arrival_times(
        remaining,
        predicted_delay
    )

    # FORMAT OUTPUT

    eta_table = eta_table[
        [
            "stop_id",
            "arrival_time",
            "predicted_arrival",
            "departure_time",
            "predicted_departure"
        ]
    ]

    return {
        "train_id": next_train["trip_id"],
        "predicted_delay": round(
            predicted_delay,
            2
        ),
        "eta_table": eta_table
    }