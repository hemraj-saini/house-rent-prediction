import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Trained Model

model = joblib.load("house_prediction_model.joblib")
area_freq = joblib.load("area_locality_freq.pkl")

# Title
st.title("🏠 House Rent Prediction System")

st.markdown(
    """
    Predict house rent using Machine Learning.
    """
    
)

st.image(
    "house.jpg",
    use_container_width=True
)


# Sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
Model : GradientBoostingRegressor

CV Score : 0.8286

Project : House Rent Prediction
""")

st.markdown("---")
st.subheader("📋 Enter House Details")

col1, col2 = st.columns(2)

with col1:

    bhk = st.number_input(
        "🏠 BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    size = st.number_input(
        "📐 Size (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1000
    )

    bathroom = st.number_input(
        "🚿 Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    current_floor = st.number_input(
        "🏢 Current Floor",
        min_value=-2,
        max_value=100,
        value=1
    )

with col2:

    total_floor = st.number_input(
        "🏙️ Total Floors",
        min_value=1,
        max_value=100,
        value=5
    )

    month = st.selectbox(
        "📅 Month",
        list(range(1, 13))
    )

    day = st.selectbox(
        "📆 Day",
        list(range(1, 32))
    )

    dayofweek = st.selectbox(
        "🗓️ Day Of Week",
        list(range(0, 7))
    )

st.markdown("---")
st.subheader("🌍 Location & Property Details")

col3, col4 = st.columns(2)

with col3:

    city = st.selectbox(
        "🏙️ City",
        [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Kolkata"
        ]
    )

    area_type = st.selectbox(
        "📍 Area Type",
        [
            "Super Area",
            "Carpet Area",
            "Built Area"
        ]
    )

    furnishing = st.selectbox(
        "🛋️ Furnishing Status",
        [
            "Unfurnished",
            "Semi-Furnished",
            "Furnished"
        ]
    )

with col4:

    tenant = st.selectbox(
        "👨‍👩‍👧 Tenant Preferred",
        [
            "Bachelors",
            "Family",
            "Bachelors/Family"
        ]
    )

    contact = st.selectbox(
        "📞 Point of Contact",
        [
            "Contact Owner",
            "Contact Agent",
            "Contact Builder"
        ]
    )

    area_locality = st.selectbox(
    "📍 Area Locality",
    sorted(area_freq.keys())
    
    )

st.markdown("---")

# Create Input DataFrame

input_df = pd.DataFrame({
    'BHK': [bhk],
    'Size': [size],
    'Area Type': [area_type],
    'City': [city],
    'Furnishing Status': [furnishing],
    'Tenant Preferred': [tenant],
    'Bathroom': [bathroom],
    'Point of Contact': [contact],
    'day': [day],
    'Month': [month],
    'dayofweek': [dayofweek],
    'Current Floor': [current_floor],
    'Total Floor': [total_floor],
    'Area_Locality_Freq': [area_freq[area_locality]]
})

# Predict Button

predict_btn = st.button(
    "🔮 Predict House Rent",
    use_container_width=True
)

if predict_btn:

    try:

        # Prediction on log scale
        pred_log = model.predict(input_df)

        # Convert back to original rent scale
        rent = np.expm1(pred_log)

        st.markdown("---")

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg,#1f2937,#111827);
                padding:30px;
                border-radius:20px;
                text-align:center;
                color:white;
                margin-top:20px;
            ">
                <h3>🏠 Predicted Monthly Rent</h3>
                <h1>₹ {rent[0]:,.0f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()

               # Metrics
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🤖 Model",
                "GradientBoosting"
            )

        with col2:
            st.metric(
                "📈 CV Score",
                "0.8286"
            )

        with col3:
            st.metric(
                "🎯 Features",
                "14"
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

st.markdown("---")

st.caption(
    "🏠 House Rent Prediction System | Developed by Hemraj Saini"
)

st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("Model")
st.sidebar.info("GradientBoostingRegressor")

st.sidebar.write("CV Score")
st.sidebar.info("0.8286")

st.sidebar.write("Target")
st.sidebar.info("House Rent")

st.sidebar.write("Features")
st.sidebar.info("14 Features")

st.sidebar.write("Developer")
st.sidebar.info("Hemraj Saini")

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    text-align:center;
}

.stButton>button {
    width:100%;
    height:3rem;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


