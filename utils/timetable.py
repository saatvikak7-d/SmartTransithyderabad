
import pandas as pd

# =========================================================
# LOAD DATA
# =========================================================

stop_times = pd.read_csv(
    "data/stop_times.txt"
)

trips = pd.read_csv(
    "data/trips.txt"
)

# =========================================================
# FIX INVALID TIMES
# =========================================================

def fix_time(t):

    if pd.isna(t):
        return t

    h, m, s = map(
        int,
        str(t).split(":")
    )

    h = h % 24

    return f"{h:02d}:{m:02d}:{s:02d}"


stop_times["arrival_time"] = (
    stop_times["arrival_time"]
    .apply(fix_time)
)

stop_times["departure_time"] = (
    stop_times["departure_time"]
    .apply(fix_time)
)

# =========================================================
# CONVERT TO DATETIME
# =========================================================

stop_times["arrival_time"] = pd.to_datetime(
    stop_times["arrival_time"],
    format="%H:%M:%S"
)

stop_times["departure_time"] = pd.to_datetime(
    stop_times["departure_time"],
    format="%H:%M:%S"
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

stop_times["hour"] = (
    stop_times["departure_time"]
    .dt.hour
)

stop_times["travel_segment"] = (
    stop_times
    .groupby("trip_id")["arrival_time"]
    .diff()
    .dt.total_seconds() / 60
)

stop_times["travel_segment"] = (
    stop_times["travel_segment"]
    .fillna(0)
)

# =========================================================
# MERGE TRIP DATA
# =========================================================

merged = pd.merge(
    stop_times,
    trips,
    on="trip_id",
    how="left"
)

# =========================================================
# GET NEXT TRAIN
# =========================================================

def get_next_train(
    station,
    current_time
):

    """
    Finds the next available train
    from a given station after
    the specified time.
    """

    current_time = pd.to_datetime(
        current_time,
        format="%H:%M:%S"
    )

    station_data = merged[
        merged["stop_id"] == station
    ].copy()

    upcoming = station_data[
        station_data["arrival_time"]
        >= current_time
    ]

    if upcoming.empty:
        return None

    next_train = (
        upcoming
        .sort_values("arrival_time")
        .iloc[0]
    )

    return next_train

# =========================================================
# GET REMAINING STOPS
# =========================================================

def get_remaining_stops(
    trip_id,
    current_stop_sequence
):

    """
    Returns all downstream stations
    for a given train trip.
    """

    route = merged[
        merged["trip_id"] == trip_id
    ].copy()

    remaining = route[
        route["stop_sequence"]
        >= current_stop_sequence
    ]

    remaining = remaining.sort_values(
        "stop_sequence"
    )

    return remaining[
        [
            "trip_id",
            "stop_id",
            "stop_sequence",
            "arrival_time",
            "departure_time",
            "hour",
            "travel_segment",
            "IN_numstops",
            "IN_distance",
            "IN_speed"
        ]
    ]

# =========================================================
# GET STATION LIST
# =========================================================

def get_all_stations():

    """
    Returns sorted station list
    for dropdowns/UI.
    """

    stations = sorted(
        merged["stop_id"]
        .unique()
    )

    return stations

# =========================================================
# GET TRAIN SCHEDULE
# =========================================================

def get_train_schedule(
    trip_id
):

    """
    Returns complete schedule
    for a train trip.
    """

    schedule = merged[
        merged["trip_id"] == trip_id
    ].copy()

    schedule = schedule.sort_values(
        "stop_sequence"
    )

    return schedule[
        [
            "stop_id",
            "arrival_time",
            "departure_time"
        ]
    ]
