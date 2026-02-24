import pandas as pd
import joblib
import os
from prophet import Prophet

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("dataset/model_df.csv")

# Ensure folder exists
os.makedirs("models/forecast", exist_ok=True)

# ==========================
# GET COUNTRY LIST
# ==========================
countries = df["Country"].unique()

for country in countries:

    print(f"Training Prophet model for {country}...")

    country_df = df[df["Country"] == country].copy()

    # Prepare time series
    ts = country_df[["Year", "Food Insecurity Rate"]].copy()
    ts.columns = ["ds", "y"]
    ts["ds"] = pd.to_datetime(ts["ds"], format="%Y")

    # Sort by date (IMPORTANT)
    ts = ts.sort_values("ds")

    # ==========================
    # TRAIN PROPHET
    # ==========================
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(ts)

    # ==========================
    # FORECAST 5 YEARS
    # ==========================
    future = model.make_future_dataframe(periods=5, freq="YS")
    forecast = model.predict(future)

    # Save forecast dataframe (NOT model)
    joblib.dump(forecast, f"models/forecast/{country}_prophet.pkl")

    # ==========================
    # SAVE FORECAST
    # ==========================
    joblib.dump(forecast, f"models/forecast/{country}_prophet.pkl")

print("All forecast models saved successfully.")
