import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="ASEAN Food Security Monitoring",
    layout="wide"
)

# -------------------------------------------------
# BACKGROUND FUNCTION
# -------------------------------------------------
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}

        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .block-container {{
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }}

        div[role="radiogroup"] > label > div:first-child {{
            display:none !important;
        }}

        div[role="radiogroup"] {{
            gap:30px;
            justify-content:center;
        }}

        div[role="radiogroup"] label {{
            background:none !important;
            border:none !important;
            padding:0 !important;
            cursor:pointer;
        }}

        div[role="radiogroup"] label div {{
            color:rgba(255,255,255,0.7)!important;
            font-size:18px!important;
        }}

        div[role="radiogroup"] label:hover div,
        div[role="radiogroup"] label:has(input:checked) div {{
            color:white!important;
            font-weight:bold!important;
            text-decoration:underline;
            text-underline-offset:8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_forecast_df():
    return pd.read_csv("dataset/forecast_dataset.csv")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/prediction/pred_pipeline.pkl")

@st.cache_resource
def load_forecast_model():
    return joblib.load("models/forecast/rf_forecast_model.pkl")

@st.cache_resource
def load_forecast_features():
    return joblib.load("models/forecast/forecast_feature_columns.pkl")

@st.cache_resource
def load_feature_columns():
    return joblib.load("models/prediction/feature_columns.pkl")

@st.cache_resource
def load_prediction_metrics():
    return joblib.load("models/prediction/prediction_metrics.pkl")

@st.cache_resource
def load_forecast_metrics():
    return joblib.load("models/forecast/forecast_metrics.pkl")

# -------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------
forecast_df = load_forecast_df()

prediction_model = load_pipeline()
forecast_model = load_forecast_model()

feature_columns = load_feature_columns()
forecast_features = load_forecast_features()

prediction_metrics = load_prediction_metrics()
forecast_metrics = load_forecast_metrics()

# -------------------------------------------------
# TABLEAU
# -------------------------------------------------
TABLEAU_PATHS = {
    "Overview": "views/FoodInsecurityRateDashboard/Overview"
}

def embed_tableau(path, height=650):
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

# -------------------------------------------------
# NAVIGATION
# -------------------------------------------------
nav = st.radio(
    "",
    ["Home","Dashboard","ML Prediction","ML Forecasting","Methodology"],
    horizontal=True,
    label_visibility="collapsed"
)

# =================================================
# HOME
# =================================================
if nav == "Home":

    set_background("https://i.pinimg.com/736x/cb/38/21/cb3821211ff00fb8456a24467e709004.jpg")

    col1,col2 = st.columns([1,2])

    with col1:
        st.image(
            "https://simplyadtype.wordpress.com/wp-content/uploads/2020/09/gif-1.gif",
            use_container_width=True
        )

    with col2:
        st.markdown(
        """
        <h1 style='font-size:60px;color:white'>ASEAN Food Security Monitoring</h1>
        <h3 style='color:white'>Machine Learning Forecast & Prediction</h3>
        """,
        unsafe_allow_html=True
        )

# =================================================
# DASHBOARD
# =================================================
elif nav == "Dashboard":

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Food Security Dashboard")
    embed_tableau(TABLEAU_PATHS["Overview"])

# =================================================
# ML PREDICTION
# =================================================
elif nav == "ML Prediction":

    set_background("https://i.pinimg.com/originals/7d/f1/43/7df143904208d2295c8428caaf86cb3c.gif")

    st.title("Food Insecurity Prediction")

    col1,col2 = st.columns(2)

    with col1:
        styled_predict = prediction_metrics.style.set_properties(**{
        'background-color': '#262730',  # Dark grey background
        'color': 'white',               # White text
        'border-color': 'white'         # Optional: visible borders
        })

        st.subheader("ℹ️ Prediction Model Info")
        st.dataframe(styled_predict)

        st.image(
            "https://i.pinimg.com/originals/bb/88/64/bb88641bbc1dc8e9583ee7029c546eff.gif",
            width=500
        )

    with col2:

        st.subheader("✨ Prediction Tool")

        user_inputs = {}
        errors = False

        for feature in feature_columns:

            value = st.text_input(feature)

            if value != "":
                try:
                    user_inputs[feature] = float(value)
                except ValueError:
                    st.error(f"⚠️ '{feature}' must be numeric.")
                    errors = True
            else:
                user_inputs[feature] = None

        if st.button("Predict"):

            if errors:
                st.warning("Please correct invalid inputs.")

            elif None in user_inputs.values():
                st.warning("Please fill all fields.")

            else:

                input_df = pd.DataFrame([user_inputs])
                input_df = input_df[feature_columns]

                prediction = prediction_model.predict(input_df)[0]

                st.success(
                    f"Predicted Food Insecurity Rate: {prediction:.2f}"
                )

# =================================================
# ML FORECASTING
# =================================================
elif nav == "ML Forecasting":

    set_background("https://i.pinimg.com/originals/7d/f1/43/7df143904208d2295c8428caaf86cb3c.gif")

    st.title("Food Insecurity Forecast")

    col1,col2 = st.columns(2)

    with col1:
        styled_forecast = forecast_metrics.style.set_properties(**{'background-color': '#262730',
        'color': 'white','border-color': 'white'         # Optional: visible borders
        })

        st.subheader("ℹ️ Forecasting Model Info")
        st.dataframe(styled_forecast)

        st.image(
            "https://i.pinimg.com/originals/bb/88/64/bb88641bbc1dc8e9583ee7029c546eff.gif",
            width=500
        )

    with col2:

        st.subheader("✨ Forecast Tool")

        countries = forecast_df["Country_orig"].unique()
        st.markdown("""
          <style>
          /* Target the selectbox input area */
          div[data-baseweb="select"] > div {
          background-color: #c9ae85 !important;
          color: white !important;
          }
        </style>
        """, unsafe_allow_html=True)
        
        country = st.selectbox("Select Country", countries)
        future_year = st.slider("Forecast Year", 2024, 2035)

        country_data = forecast_df[forecast_df["Country_orig"] == country]

        last_row = country_data.iloc[-1]
        prev_row = country_data.iloc[-2]

        lag1 = last_row["Food Insecurity Rate"]
        lag2 = prev_row["Food Insecurity Rate"]

        water_lag1 = last_row["water access"]
        water_lag2 = prev_row["water access"]

        rolling_mean = country_data["Food Insecurity Rate"].tail(3).mean()

        time_index = future_year - forecast_df["Year"].min()

        input_data = pd.DataFrame(0, index=[0], columns=forecast_features)

        input_data["Food Insecurity Rate_lag1"] = lag1
        input_data["Food Insecurity Rate_lag2"] = lag2
        input_data["water access_lag1"] = water_lag1
        input_data["water access_lag2"] = water_lag2
        input_data["food_insecurity_roll3"] = rolling_mean
        input_data["time_index"] = time_index
        input_data["water access"] = last_row["water access"]

        country_col = f"Country_orig_{country}"

        if country_col in input_data.columns:
            input_data[country_col] = 1

        if st.button("Generate Forecast"):

            prediction = forecast_model.predict(input_data)[0]

            st.success(
                f"Forecast Food Insecurity Rate: {prediction:.2f}"
            )

            chart_df = country_data[["Year","Food Insecurity Rate"]].copy()

            forecast_point = pd.DataFrame({
                "Year":[future_year],
                "Food Insecurity Rate":[prediction]
            })

            chart_df = pd.concat([chart_df,forecast_point])

            fig = px.line(
                chart_df,
                x="Year",
                y="Food Insecurity Rate",
                title=f"{country} Forecast"
            )

            st.plotly_chart(fig)

# =================================================
# METHODOLOGY
# =================================================
elif nav == "Methodology":

    set_background("https://i.pinimg.com/1200x/78/1d/4c/781d4c6becbd05f20e26057f6cbaf9bc.jpg")

    st.title("Project Methodology")
    st.image("assets/methodology.png", width=800)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
footer_html = """
<style>

.footer {
position: relative;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: white;
text-align: center;
padding: 20px 0;
font-size: 14px;
z-index: 9999;
}

.footer a {
margin: 0 10px;
text-decoration: none;
}

.footer img {
width: 28px;
margin-left: 8px;
margin-right: 8px;
vertical-align: middle;
transition: transform 0.2s;
}

.footer img:hover {
transform: scale(1.2);
}

</style>

<div class="footer">

<p>
Built with Data & Passion | © 2026 Jenifer M Jues
</p>

<a href="https://github.com/JMJ-ai/Fraud-Analytic-and-ML-Detection" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
</a>

<a href="https://www.linkedin.com/in/jenifermayangjues/" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
</a>

<a href="https://icons8.com/" target="_blank">
<img src="https://img.icons8.com/?size=100&id=ayJDJ6xQKgM6&format=png&color=000000">
</a>

<a href="mailto:jeniferjues@gmail.com">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png">
</a>

</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
