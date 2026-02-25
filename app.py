import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils.preprocess import preprocess_input

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ASEAN Food Security Monitoring",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
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

# ---------------------------------------------------
# 🔥 TABLEAU DASHBOARD PATHS
# Replace with YOUR workbook + sheet names
# Format: views/WorkbookName/DashboardName
# ---------------------------------------------------
TABLEAU_PATHS = {
    "Overview Dashboard": "views/FoodInsecurityRateDashboard/OverviewDashboard",
    "Driver Analysis": "views/FoodInsecurityRateDashboard/DriverAnalysis",
    "Forecasting": "views/FoodInsecurityRateDashboard/Forecasting",
    "Insights": "views/FoodInsecurityRateDashboard/Insights"
}

# ---------------------------------------------------
# FUNCTION TO EMBED TABLEAU (STREAMLIT SAFE)
# ---------------------------------------------------
def embed_tableau(path, height=800):
    html_code = f"""
    <script type='module' src='https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js'></script>
    <tableau-viz
        src="https://public.tableau.com/{path}"
        width="100%"
        height="{height}"
        toolbar="hidden"
        hide-tabs>
    </tableau-viz>
    """
    st.components.v1.html(html_code, height=height)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.image("assets/Header.png", use_container_width=True)
st.title("ASEAN Food Security Monitoring Dashboard")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview Dashboard",
     "Driver Analysis",
     "Forecasting",
     "Insights"]
)

# ===================================================
# PAGE 1 — OVERVIEW
# ===================================================
if page == "Overview Dashboard":

    st.header("Overview Dashboard")

    # 🔹 Tableau Interactive Dashboard
    embed_tableau(TABLEAU_PATHS["Overview Dashboard"])

    st.divider()

    # 🔹 KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric("Average Food Insecurity",
                round(df["Food Insecurity Rate"].mean(), 2))

    col2.metric("Highest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmax())

    col3.metric("Lowest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmin())

    # 🔹 Trend (Python)
    st.subheader("Trend by Country")
    fig = px.line(df, x="Year", y="Food Insecurity Rate", color="Area")
    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# PAGE 2 — DRIVER ANALYSIS
# ===================================================
elif page == "Driver Analysis":

    st.header("Driver Analysis")

    # 🔹 Tableau Dashboard
    embed_tableau(TABLEAU_PATHS["Driver Analysis"])

    st.divider()

    # 🔹 Feature Importance
    st.subheader("Feature Importance")

    try:
        fi = pd.read_csv("dataset/feature_importance.csv")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance file not found.")

    # 🔹 Correlation
    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    st.dataframe(corr)

    # 🔹 Prediction Tool
    st.subheader("Prediction Tool")

    col1, col2 = st.columns(2)

    with col1:
        avg_food_value = st.number_input("Food Production Value", 0.0, 5000.0, 1000.0)
        cereal_import = st.number_input("Cereal Import (%)", 0.0, 100.0, 30.0)
        caloric_losses = st.number_input("Caloric Losses (%)", 0.0, 100.0, 10.0)
        food_prod_var = st.number_input("Food Production Variability", 0.0, 500.0, 50.0)
        food_supply_var = st.number_input("Food Supply Variability", 0.0, 500.0, 50.0)

    with col2:
        irrigation_land = st.number_input("Irrigation (%)", 0.0, 100.0, 40.0)
        water_access = st.number_input("Water Access (%)", 0.0, 100.0, 70.0)
        sanitation = st.number_input("Sanitation (%)", 0.0, 100.0, 70.0)
        political_stability = st.number_input("Political Stability", -3.0, 3.0, 0.0)
        cpi = st.number_input("Consumer Price Index", 0.0, 300.0, 120.0)

    if st.button("Predict Food Insecurity"):

        user_input = {
            "Food Production Value": avg_food_value,
            "Cereal Import (%)": cereal_import,
            "Caloric Losses (%)": caloric_losses,
            "Food Production Variability": food_prod_var,
            "Food Supply Variability": food_supply_var,
            "Irrigation (%)": irrigation_land,
            "Water Access (%)": water_access,
            "Sanitation (%)": sanitation,
            "Political Stability": political_stability,
            "CPI": cpi
        }

        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)

        st.success(f"Predicted Food Insecurity Rate: {prediction[0]:.2f}")

# ===================================================
# PAGE 3 — FORECASTING
# ===================================================
elif page == "Forecasting":

    st.header("Forecasting Dashboard")

    # 🔹 Tableau Forecast
    embed_tableau(TABLEAU_PATHS["Forecasting"])

    st.divider()

    country = st.selectbox(
        "Select Country",
        model_df["Country"].unique()
    )

    try:
        prophet_model = joblib.load(
            f"models/forecast/{country}_prophet.pkl"
        )

        future = prophet_model.make_future_dataframe(periods=5, freq="YS")
        forecast = prophet_model.predict(future)

        st.subheader("Forecast Trend")
        fig = px.line(forecast, x="ds", y="yhat")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.warning("Forecast model not available for this country.")

# ===================================================
# PAGE 4 — INSIGHTS
# ===================================================
elif page == "Insights":

    st.header("Strategic Insights")

    embed_tableau(TABLEAU_PATHS["Insights"])

    st.divider()

    st.markdown("""
### Key Findings

- Infrastructure access strongly reduces food insecurity  
- Water and sanitation are major structural drivers  
- Import dependency increases vulnerability  
- Forecasting enables proactive policy planning  

### Policy Recommendations

- Invest in irrigation and water systems  
- Strengthen local agricultural production  
- Reduce food import exposure  
- Monitor price volatility
""")
