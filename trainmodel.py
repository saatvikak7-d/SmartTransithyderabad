import numpy as np
import seaborn as sns
from statsmodels import imputation as imp
import matplotlib.pyplot as plt
import pandas as pd
import os

import os
import joblib

def train_model():

    os.makedirs("models", exist_ok=True)
data = pd.read_csv('data/stop_times.txt')

print(data.columns)

def fix_time(t):
    if pd.isna(t):
        return t

    h, m, s = map(int, str(t).split(":"))
    h = h % 24

    return f"{h:02d}:{m:02d}:{s:02d}"


# fix invalid hours like 24:xx:xx or 25:xx:xx
data["arrival_time"] = data["arrival_time"].apply(fix_time)
data["departure_time"] = data["departure_time"].apply(fix_time)

# convert to datetime
data["arrival_time"] = pd.to_datetime(
    data["arrival_time"],
    format="%H:%M:%S"
)

data["departure_time"] = pd.to_datetime(
    data["departure_time"],
    format="%H:%M:%S"
)

data.describe()
data.apply(np.max)
data["hour"] = data["departure_time"].dt.hour

data["hour"].value_counts().sort_index().plot(kind="bar")

counts = data["stop_id"].head(20).value_counts()

plt.figure(figsize=(12, 6))
plt.bar(counts.index, counts.values)

plt.xlabel("Station ID")
plt.ylabel("Number of Stops")
plt.title("Frequency of Stops by Station")

plt.xticks(rotation=90)

plt.show()

data["travel_segment"] = (
        data.groupby("trip_id")["arrival_time"]
        .diff()
        .dt.total_seconds() / 60
)
hour_counts = data["hour"].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(hour_counts.index, hour_counts.values, marker='o')

plt.xlabel("Hour of Day")
plt.ylabel("Number of Train Stops")
plt.title("MMTS Activity Throughout the Day")

plt.grid(True)

plt.show()

top_stations = data["stop_id"].value_counts().head(15)

plt.figure(figsize=(12, 6))
plt.bar(top_stations.index, top_stations.values)

plt.xticks(rotation=45)

plt.xlabel("Station")
plt.ylabel("Number of Stops")
plt.title("Top MMTS Stations by Traffic")

plt.show()
station_data = data[data["stop_id"] == "HYB"].copy()

station_data = station_data.sort_values("arrival_time")

station_data["gap_minutes"] = (
        station_data["arrival_time"]
        .diff()
        .dt.total_seconds() / 60
)

print(station_data["gap_minutes"].describe())

trips = pd.read_csv("data/trips.txt")

merged = pd.merge(
    data,
    trips,
    on="trip_id",
    how="left"
)
pivot = pd.crosstab(
    data["stop_id"],
    data["hour"]
)

plt.figure(figsize=(14, 10))

plt.imshow(pivot, aspect='auto')

plt.colorbar(label="Train Frequency")

plt.yticks(range(len(pivot.index)), pivot.index)

plt.xlabel("Hour")
plt.ylabel("Station")
plt.title("MMTS Station Activity Heatmap")

plt.show()



delay_data = pd.read_csv(
    "data/etrain_delays.csv"
)

print(delay_data.head())

print(delay_data.info())




print(
    delay_data.isnull().sum()
)



delay_data["average_delay_minutes"] = ( delay_data["average_delay_minutes"] .fillna( delay_data["average_delay_minutes"].median() ) )

numeric_cols = [ "average_delay_minutes",
                 "pct_right_time",
                 "pct_slight_delay",
                 "pct_significant_delay",
                 "pct_cancelled_unknown" ]
delay_data[numeric_cols] = ( delay_data[numeric_cols] .apply(pd.to_numeric, errors="coerce") )

delay_data = delay_data.dropna()


plt.figure(figsize=(10,6))

plt.hist(
    delay_data["average_delay_minutes"],
    bins=90
)

plt.xlabel("Average Delay (Minutes)")
plt.ylabel("Frequency")

plt.title("Distribution of Railway Delays")

plt.show()




from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
#after the eda, we notice the important parts of our datasetgi
features = [
    "pct_right_time",
    "pct_slight_delay",
    "pct_significant_delay",
    "pct_cancelled_unknown"
]

X = delay_data[features]

y = delay_data["average_delay_minutes"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# We choose the parameters of the random forest to suit the size of the dataset.
rf_regressor = RandomForestRegressor(
    n_estimators=150,
    max_depth=70,
    random_state=42
)

# Training the random forest model

rf_regressor.fit(X_train, y_train)

# Prediction
predictions = rf_regressor.predict(X_test)

# EVALUATION

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Performance:\n")

print(f"MAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")
print(f"R²   : {r2:.4f}")

#Feature importance

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf_regressor.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(importance_df)

#plotting of the features
plt.figure(figsize=(10, 6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.title("Feature Importance for Delay Prediction")

plt.xticks(rotation=20)

plt.show()

#plotting the predictions
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.xlabel("Actual Delay")
plt.ylabel("Predicted Delay")

plt.title("Actual vs Predicted Delay")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()


    # after training:
    joblib.dump(model, "models/delay_predictor.pkl")

    return model

#saving the model
