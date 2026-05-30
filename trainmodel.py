import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model():

    os.makedirs("models", exist_ok=True)

    delay_data = pd.read_csv(
        "data/etrain_delays.csv"
    )

    delay_data["average_delay_minutes"] = (
        delay_data["average_delay_minutes"]
        .fillna(delay_data["average_delay_minutes"].median())
    )

    numeric_cols = [
        "average_delay_minutes",
        "pct_right_time",
        "pct_slight_delay",
        "pct_significant_delay",
        "pct_cancelled_unknown"
    ]

    delay_data[numeric_cols] = (
        delay_data[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    delay_data = delay_data.dropna()

    features = [
        "pct_right_time",
        "pct_slight_delay",
        "pct_significant_delay",
        "pct_cancelled_unknown"
    ]

    X = delay_data[features]

    y = delay_data["average_delay_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    rf_regressor = RandomForestRegressor(
        n_estimators=150,
        max_depth=70,
        random_state=42
    )

    rf_regressor.fit(X_train, y_train)

    joblib.dump(
        rf_regressor,
        "models/delay_predictor.pkl"
    )

    return rf_regressor


if __name__ == "__main__":
    train_model()
