import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load data
cars = pd.read_csv("cleaned_car_dataset.csv")

# Load model
model = pickle.load(
    open("car_price_model.pkl", "rb")
)

st.title("🚗 Car Price Predictor")

st.markdown(
    """
Predict the resale value of a used car using Machine Learning.

Model Performance:
- R² Score: 0.73
- Algorithm: Gradient Boosting Regressor
"""
)

# Company selection
company = st.selectbox(
    "Select Company",
    sorted(cars["company"].unique())
)

# Filter cars by company
filtered_cars = cars[
    cars["company"] == company
]["name"].unique()

name = st.selectbox(
    "Select Car Model",
    sorted(filtered_cars)
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(cars["fuel_type"].unique())
)

year = st.slider(
    "Manufacturing Year",
    int(cars["year"].min()),
    int(cars["year"].max()),
    2018
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)

if st.button("Predict Price"):

    data = pd.DataFrame(
        {
            "name": [name],
            "company": [company],
            "year": [year],
            "kms_driven": [kms_driven],
            "fuel_type": [fuel_type]
        }
    )

    prediction = model.predict(data)[0]

    st.success(
        f"Estimated Price: ₹{int(prediction):,}"
    )

st.markdown("---")
st.caption(
    "Built with Streamlit and Scikit-Learn"
)