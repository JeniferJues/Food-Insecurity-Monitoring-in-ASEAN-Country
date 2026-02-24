import pandas as pd
import joblib

def preprocess_input(user_input_dict):

    # Load correct feature order
    feature_columns = joblib.load("models/prediction/feature_columns.pkl")

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input_dict])

    # Ensure correct order
    input_df = input_df[feature_columns]

    return input_df
