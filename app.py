import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils.preprocess import preprocess_input

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ASEAN Food Security Monitoring",
    layout="wide"
)

st.components.v1.html("""
<script>
window.scrollTo({top: 0, behavior: 'smooth'});
</script>
""", height=0, width=0)

# =====================================================
# LOAD DATA
# =====================================================
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
prediction_model = load_prediction_model()

# =====================================================
# TABLEAU PATHS
# Replace with your real workbook + dashboard names
# Example: views/ASEANFoodSecurity/Overview
# =====================================================
TABLEAU_PATHS = {
    "Overview Dashboard": "views/FoodInsecurityRateDashboard/OverviewDashboard",
    "Driver Analysis": "views/FoodInsecurityRateDashboard/DriverAnalysis",
    "Forecasting": "views/FoodInsecurityRateDashboard/Forecasting",
    "Insights": "views/FoodInsecurityRateDashboard/Insights"
}

# =====================================================
# SAFE TABLEAU EMBED FUNCTION
# =====================================================
def embed_tableau(path, height=700):
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

# =====================================================
# HEADER
# =====================================================
st.image("assets/Header.png", use_container_width=True)
st.title("ASEAN Food Security Monitoring Dashboard")

# =====================================================
# SIDEBAR
# =====================================================
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

    st.markdown("""
    This dashboard provides a high-level overview of food insecurity trends
    across ASEAN countries including country comparison and historical patterns.
    """)

    st.divider()

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric("Average Food Insecurity",
                round(df["Food Insecurity Rate"].mean(), 2))

    col2.metric("Highest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmax())

    col3.metric("Lowest Country",
                df.groupby("Area")["Food Insecurity Rate"].mean().idxmin())

    st.divider()

    # Python Trend
    st.subheader("Trend by Country")
    fig = px.line(df, x="Year", y="Food Insecurity Rate", color="Area")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Tableau
    st.subheader("Interactive Tableau Dashboard")
    embed_tableau(TABLEAU_PATHS["Overview Dashboard"])


# =====================================================
# PAGE 2 — DRIVER ANALYSIS
# =====================================================
elif page == "Driver Analysis":

    st.header("Driver Analysis")

    # -----------------------------
    # DESCRIPTION (ALWAYS FIRST)
    # -----------------------------
    st.markdown("""
    ## 🔍 Prediction Model Overview

    The prediction model was built using **Random Forest Regressor** 
    after comparing multiple algorithms.

    ### Why Random Forest?
    - Achieved highest model performance
    - Average Cross-Validation Score: **0.95**
    - Captures nonlinear relationships
    - Reduces overfitting via ensemble learning

    ### Feature Selection
    Selected features were determined through:
    - Importance ranking
    - Performance validation
    - Multicollinearity reduction

    The model predicts the **Food Insecurity Rate**
    using economic, agricultural, and infrastructure indicators.
    """)

    st.divider()

    # Feature Importance
    st.subheader("Feature Importance")

    try:
        fi = pd.read_csv("dataset/feature_importance.csv")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance file not found.")

    st.divider()

    # Prediction Tool
    st.subheader("Food Insecurity Prediction Tool")

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
        prediction = prediction_model.predict(input_df)

        st.success(f"Predicted Food Insecurity Rate: {prediction[0]:.2f}")

    st.divider()

    # Tableau
    st.subheader("Interactive Driver Dashboard")
    embed_tableau(TABLEAU_PATHS["Driver Analysis"])


# =====================================================
# PAGE 3 — FORECASTING
# =====================================================
elif page == "Forecasting":

    st.header("Forecasting")

    # -----------------------------
    # DESCRIPTION FIRST
    # -----------------------------
    st.markdown("""
    ## 📈 Forecasting Model Overview

    Forecasting was conducted using **Facebook Prophet**.

    ### Why Prophet?
    - Designed for business and policy forecasting
    - Captures trend changes effectively
    - Works well with yearly time-series data
    - Robust to missing values and outliers
    - Provides confidence intervals

    Prophet was selected due to its interpretability and 
    stable forecasting performance across ASEAN countries.
    """)

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

        st.subheader("Confidence Interval")
        fig2 = px.line(forecast, x="ds",
                       y=["yhat_lower", "yhat_upper"])
        st.plotly_chart(fig2, use_container_width=True)

        forecast["growth"] = forecast["yhat"].pct_change() * 100
        st.subheader("Projected Growth Rate (%)")
        fig3 = px.line(forecast, x="ds", y="growth")
        st.plotly_chart(fig3, use_container_width=True)

    except:
        st.warning("Forecast model not available.")

    st.divider()

    # Tableau
    st.subheader("Interactive Forecast Dashboard")
    embed_tableau(TABLEAU_PATHS["Forecasting"])


# =====================================================
# PAGE 4 — INSIGHTS
# =====================================================
elif page == "Insights":

    st.header("Strategic Insights")

    st.markdown("""
    ## Key Findings

    - Infrastructure access significantly reduces food insecurity  
    - Water and sanitation are major structural drivers  
    - Import dependency increases vulnerability  
    - Forecasting enables proactive planning  

    ## Policy Recommendations

    - Strengthen irrigation systems  
    - Improve rural water access  
    - Reduce food import dependency  
    - Monitor food price volatility  
    """)

    st.divider()

    embed_tableau(TABLEAU_PATHS["Insights"])
