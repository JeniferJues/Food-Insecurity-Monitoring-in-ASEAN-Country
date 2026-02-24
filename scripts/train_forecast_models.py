(Multi-country Prophet models)
import pandas as pd
import joblib
from prophet import Prophet

df = pd.read_csv("dataset/model_df.csv")

countries = df["Area"].unique()

for country in countries:

    country_df = df[df["Country"] == country]

    ts = country_df[["Year","Food Insecurity Rate"]].copy()
    ts.columns = ["ds","y"]
    ts["ds"] = pd.to_datetime(ts["ds"], format="%Y")

    model = Prophet()
    model.fit(ts)

    future = model.make_future_dataframe(periods=5, freq="Y")
    forecast = model.predict(future)

    joblib.dump(forecast, f"models/forecast/{country}_prophet.pkl")

print("Forecast models saved.")
