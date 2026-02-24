import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from utils.preprocess import preprocess_input

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ASEAN Food Security Monitoring",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/eda_df.csv")

@st.cache_resource
def load_prediction_model():
    return joblib.load("models/prediction/food_model.pkl")

df = load_data()
model = load_prediction_model()

# -----------------------------
# HEADER
# -----------------------------
st.image("assets/Header.png", use_column_width=True)
st.title("ASEAN Food Security Monitoring Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview Dashboard",
     "Driver Analysis",
     "Forecasting",
     "Insights"]
)

# =====================================================
# PAGE 1 — OVERVIEW
# =====================================================
if page == "Overview Dashboard":

    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Food Insecurity", round(df["Food Insecurity Rate"].mean(),2))
    col2.metric("Highest Country", df.groupby("Area")["Food Insecurity Rate"].mean().idxmax())
    col3.metric("Lowest Country", df.groupby("Area")["Food Insecurity Rate"].mean().idxmin())

    # Trend line
    st.subheader("Trend by Country")
    fig = px.line(df, x="Year", y="Food Insecurity Rate", color="Area")
    st.plotly_chart(fig, use_container_width=True)

    # Country comparison
    st.subheader("Country Comparison")
    country_avg = df.groupby("Area")["Food Insecurity Rate"].mean().reset_index()
    fig2 = px.bar(country_avg, x="Area", y="Food Insecurity Rate")
    st.plotly_chart(fig2, use_container_width=True)

    # Tableau embed
    st.subheader("Interactive Tableau Dashboard")
    tableau_url = "YOUR_TABLEAU_EMBED_LINK"
    st.components.v1.iframe(tableau_url, height=700)

# =====================================================
# PAGE 2 — DRIVER ANALYSIS
# =====================================================
elif page == "Driver Analysis":

    st.header("Driver Analysis")

    # Feature importance
    st.subheader("Feature Importance")

    try:
        fi = pd.read_csv("dataset/feature_importance.csv")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Upload feature importance data.")

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    st.dataframe(corr)

    # Prediction tool
    st.subheader("Food Insecurity Prediction Tool")

    irrigation = st.slider("Irrigation %",0.0,100.0,50.0)
    water = st.slider("Water Access %",0.0,100.0,60.0)

    if st.button("Predict Risk"):
        input_df = preprocess_input(irrigation, water)
        pred = model.predict(input_df)
        st.success(f"Predicted Food Insecurity: {pred[0]:.2f}")

# =====================================================
# PAGE 3 — FORECASTING
# =====================================================
elif page == "Forecasting":

    st.header("Country Forecast")

    country = st.selectbox("Select Country", df["Area"].unique())

    try:
        forecast = joblib.load(f"models/forecast/{country}_forecast.pkl")

        st.subheader("Forecast Trend")
        fig = px.line(forecast, x="ds", y="yhat")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Interval")
        fig2 = px.line(forecast, x="ds", y=["yhat_lower","yhat_upper"])
        st.plotly_chart(fig2, use_container_width=True)

        forecast["growth"] = forecast["yhat"].pct_change()*100
        st.subheader("Growth Rate")
        fig3 = px.line(forecast, x="ds", y="growth")
        st.plotly_chart(fig3, use_container_width=True)

    except:
        st.warning("Forecast model not available.")

# =====================================================
# PAGE 4 — INSIGHTS
# =====================================================
elif page == "Insights":

    st.header("Key Insights")

    st.markdown("""
### Key Findings
- Infrastructure access strongly influences food insecurity.
- Water and sanitation are major drivers.
- Some ASEAN countries show stable improvement.
- Forecasting helps policy planning.

### Policy Implications
- Improve agricultural production stability.
- Enhance food distribution systems.
- Invest in infrastructure.
""")
