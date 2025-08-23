import streamlit as st
import numpy as np
import pandas as pd
import joblib 

# --- Page Configuration ---
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="auto"
)
# Adding custom CSS for better styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    
    
    
    
    
    
    
    # Load The Model Pipeline ---
try:
    # Load the entire Pipeline object.
    model_pipeline = joblib.load('rent_model.pkl')
except FileNotFoundError:
    st.error("Error: 'rent_model.pkl' not found. Please make sure the model file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model pipeline: {e}")
    st.stop()

# --- Application Header ---
st.title("🏠 Advanced House Rent Prediction")
st.markdown("Enter the details of the property to get an estimated rent price.")
st.markdown("---")

# --- User Input Section ---
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"])
    bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2, step=1)
    size = st.number_input("Size (in Square Feet)", min_value=100, max_value=10000, value=1000, step=50)
    bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built-up Area"])

with col2:
    # Furnishing status as 0,1,2 for Unfurnished, Semi-Furnished, Furnished
    furnishing_status = st.number_input(
        "Furnishing Status (0: Unfurnished, 1: Semi-Furnished, 2: Furnished)",
        min_value=0, max_value=2, value=1, step=1
    )
    tenant_type = st.selectbox("Preferred Tenant", ["Bachelors", "Family", "Bachelors/Family"])
    cur_floor = st.number_input("Current Floor", min_value=0, max_value=100, value=1, step=1)
    tot_floor = st.number_input("Total Floors", min_value=cur_floor, max_value=200, value=5, step=1)

# --- Prediction Logic ---
if st.button('Predict Rent', key='predict_button'):
    if tot_floor < cur_floor:
        st.error("Error: Total floors cannot be less than the current floor.")
    else:
        try:
            # 1. Create a DataFrame from user inputs.
            # The Pipeline will handle feature engineering and encoding internally.
            # BHK	Size	Area Type	City	Furnishing Status	Tenant Preferred
            # 	Bathroom	current_floor	total_floors	rent_per_sqft	bath_bhk_ratio
            input_data = {
                "BHK": [bhk],
                "Size": [size],
                "Area Type": [area_type],
                "City": [city],
                "Furnishing Status": [furnishing_status],
                "Tenant Preferred": [tenant_type],
                "Bathroom": [bathroom],
                "current_floor": [cur_floor],
                #"total_floors": [tot_floor], 
                #"rent_per_sqft": [0],  # Placeholder, will be calculated in the model
                "bath_bhk_ratio": [bathroom / bhk if bhk > 0 else 0]
            }
            df_in = pd.DataFrame(input_data)
            st.write(df_in)

            # --- Make Prediction ---
            # Pass the raw DataFrame to the pipeline.
            pred_log = model_pipeline.predict(df_in)[0]
            predicted_rent = float(np.expm1(pred_log))

            # --- Display Result ---
            st.success(f"**Predicted Monthly Rent: ₹ {predicted_rent:,.0f}**")
            st.balloons()

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.warning("Please ensure your input features and preprocessing steps match the model's training data.")


#--- Disclimber
st.warning("Disclaimer")
st.markdown("""
            This application is for educational purposes only. 
            The predictions are based on a machine learning model and may not reflect actual market conditions.
            Always verify with real estate professionals before making any decisions.
            """)
# ---- about model
st.sidebar.header("About the Model")
st.sidebar.markdown("""
This model is built using a machine learning pipeline that includes preprocessing steps such as encoding categorical variables,
scaling numerical features, and applying a regression model to predict house rents based on various property features.  """)
st.sidebar.markdown("The model was trained on a dataset of house rents and is designed to provide accurate predictions based on user inputs.")

