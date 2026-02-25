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
def load_eda_data():
    return pd.read_csv("dataset/eda_df.csv")

@st.cache_data
def load_model_df_data():
    return pd.read_csv("dataset/model_df.csv")

@st.cache_resource
def load_prediction_model():
    return joblib.load("models/prediction/food_model.pkl")

df = load_eda_data()
model_df = load_model_df_data()
model = load_prediction_model()

# -----------------------------
# TABLEAU LINKS (REPLACE WITH YOUR LINKS)
# -----------------------------
TABLEAU_OVERVIEW = "https://public.tableau.com/views/FoodInsecurityRateDashboard/OverviewDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
TABLEAU_DRIVER = "https://public.tableau.com/views/FoodInsecurityRateDashboard/DriverAnalysis?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
TABLEAU_FORECAST = "https://public.tableau.com/views/FoodInsecurityRateDashboard/Forecasting?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
TABLEAU_INSIGHTS = "https://public.tableau.com/shared/5842KMTCW?:display_count=n&:origin=viz_share_link"

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

    st.header("Overview Dashboard")

    # 🔹 Embedded Tableau
    st.subheader("Interactive Tableau Dashboard")
    st.components.v1.iframe(TABLEAU_OVERVIEW, height=800)

    st.divider()

    # 🔹 KPI Cards (Python Version)
    col1, col2, col3 = st.columns(3)

    col1.metric("Average Food Insecurity",
                round(df["Food Insecurity Rate"].mean(),2))
    col2.metric("Highest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmax())
    col3.metric("Lowest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmin())

    # 🔹 Trend line
    st.subheader("Trend by Country (Python View)")
    fig = px.line(df, x="Year", y="Food Insecurity Rate", color="Area")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 2 — DRIVER ANALYSIS
# =====================================================
elif page == "Driver Analysis":

    st.header("Driver Analysis")

    # 🔹 Embedded Tableau
    st.subheader("Interactive Tableau Driver Dashboard")
    st.components.v1.iframe(TABLEAU_DRIVER, height=800)

    st.divider()

    # 🔹 Feature importance (Python)
    st.subheader("Feature Importance (Model Output)")

    try:
        fi = pd.read_csv("dataset/feature_importance.csv")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Upload feature importance data.")

    # 🔹 Correlation matrix
    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    st.dataframe(corr)

    # 🔹 Prediction tool
    st.subheader("Food Insecurity Prediction Tool")

    col1, col2 = st.columns(2)

    with col1:
        avg_food_value = st.number_input("Average Food Production Value", 0.0, 5000.0, 1000.0)
        cereal_import = st.number_input("Cereal Import Dependency (%)", 0.0, 100.0, 30.0)
        caloric_losses = st.number_input("Caloric Losses (%)", 0.0, 100.0, 10.0)
        food_prod_var = st.number_input("Food Production Variability", 0.0, 500.0, 50.0)
        food_supply_var = st.number_input("Food Supply Variability", 0.0, 500.0, 50.0)
        irrigation_land = st.number_input("Arable Land Irrigation (%)", 0.0, 100.0, 40.0)
        child_overweight = st.number_input("Children Overweight (%)", 0.0, 100.0, 5.0)

    with col2:
        water_access = st.number_input("Water Access (%)", 0.0, 100.0, 70.0)
        sanitation = st.number_input("Sanitation Access (%)", 0.0, 100.0, 70.0)
        irrigation = st.number_input("Irrigation Index", 0.0, 100.0, 50.0)
        political_stability = st.number_input("Political Stability Index", -3.0, 3.0, 0.0)
        anemia = st.number_input("Anemia Prevalence (%)", 0.0, 100.0, 20.0)
        cereal_energy = st.number_input("Dietary Energy from Cereals", 0.0, 1000.0, 400.0)
        food_imports = st.number_input("Food Imports (%)", 0.0, 100.0, 20.0)
        cpi = st.number_input("Consumer Price Index", 0.0, 300.0, 120.0)

    if st.button("Predict Food Insecurity"):

        user_input = {
            'Average value of food production (constant 2004-2006 I$/cap) (3-year average)': avg_food_value,
            'Cereal import dependency ratio (percent) (3-year average)': cereal_import,
            'Incidence of caloric losses at retail distribution level (percent)': caloric_losses,
            'Per capita food production variability (constant 2004-2006 thousand int$ per capita)': food_prod_var,
            'Per capita food supply variability (kcal/cap/day)': food_supply_var,
            'Percent of arable land equipped for irrigation (percent) (3-year average)': irrigation_land,
            'Percentage of children under 5 years of age who are overweight (percent)': child_overweight,
            'water access': water_access,
            'Percentage of population using at least basic sanitation services (percent)': sanitation,
            'irrigation': irrigation,
            'Political stability and absence of violence/terrorism (index)': political_stability,
            'Prevalence of anemia among women of reproductive age (15-49 years)': anemia,
            'Share of dietary energy supply derived from cereals (kcal/cap/day)': cereal_energy,
            'Value of food imports (percent)': food_imports,
            'Consumer Prices Index': cpi
        }

        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)

        st.success(f"Predicted Food Insecurity Rate: {prediction[0]:.2f}")

# =====================================================
# PAGE 3 — FORECASTING
# =====================================================
elif page == "Forecasting":

    st.header("Forecasting Dashboard")

    # 🔹 Embedded Tableau Forecast
    st.subheader("Interactive Tableau Forecast")
    st.components.v1.iframe(TABLEAU_FORECAST, height=800)

    st.divider()

    country = st.selectbox(
        "Select Country",
        model_df["Country"].unique()
    )

    try:
        model = joblib.load(
            f"models/forecast/{country}_prophet.pkl"
        )

        future = model.make_future_dataframe(periods=5, freq="YS")
        forecast = model.predict(future)

        st.subheader("Forecast Trend (Python)")
        fig = px.line(forecast, x="ds", y="yhat")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Forecast model not available.")

# =====================================================
# PAGE 4 — INSIGHTS
# =====================================================
elif page == "Insights":

    st.header("Strategic Insights")

    st.subheader("Interactive Tableau Insights Dashboard")
    st.components.v1.iframe(TABLEAU_INSIGHTS, height=800)

    st.divider()

    st.markdown("""
### Key Findings
- Infrastructure access strongly influences food insecurity.
- Water and sanitation are major structural drivers.
- Some ASEAN countries show long-term improvement.
- Forecasting supports policy planning and risk management.

### Policy Implications
- Improve rural water access
- Stabilize food production systems
- Strengthen agricultural resilience
- Monitor price volatility
""")
