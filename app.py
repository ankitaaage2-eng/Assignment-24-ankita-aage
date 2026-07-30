
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Ford Car Prediction",
    layout="centered"
)

st.title("Ford Car Prediction")
st.write("Session 24 AIML Assignment")

# Load saved files
classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")

classification_scaler = joblib.load("classification_scaler.pkl")
regression_scaler = joblib.load("regression_scaler.pkl")

label_encoders = joblib.load("label_encoders.pkl")
columns = joblib.load("columns.pkl")

# Select problem type
problem_type = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

st.subheader("Enter Car Details")

model_name = st.selectbox(
    "Car Model",
    label_encoders["model"].classes_.tolist()
)

year = st.number_input("Year", 2000, 2026, 2019)

transmission = st.selectbox(
    "Transmission",
    label_encoders["transmission"].classes_.tolist()
)

mileage = st.number_input("Mileage", 0, 200000, 10000)

fuel_type = st.selectbox(
    "Fuel Type",
    label_encoders["fuelType"].classes_.tolist()
)

tax = st.number_input("Tax", 0, 1000, 150)

mpg = st.number_input("MPG", 0.0, 200.0, 50.0)

engine_size = st.number_input(
    "Engine Size",
    0.0,
    10.0,
    1.0
)

# Encode inputs
model_encoded = label_encoders["model"].transform([model_name])[0]
transmission_encoded = label_encoders["transmission"].transform([transmission])[0]
fuel_encoded = label_encoders["fuelType"].transform([fuel_type])[0]

# Create input data
input_data = pd.DataFrame([[
    model_encoded,
    year,
    transmission_encoded,
    mileage,
    fuel_encoded,
    tax,
    mpg,
    engine_size
]], columns=columns)

# Prediction
if st.button("Predict"):

    if problem_type == "Classification":

        input_scaled = classification_scaler.transform(input_data)

        prediction = classification_model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Prediction: Higher Price Category")
        else:
            st.success("Prediction: Lower Price Category")

    else:

        input_scaled = regression_scaler.transform(input_data)

        prediction = regression_model.predict(input_scaled)[0]

        st.success(
            f"Predicted Car Price: £{prediction:,.2f}"
        )
