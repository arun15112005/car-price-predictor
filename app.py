import streamlit as st
import pandas as pd
import pickle

cars = pd.read_csv(
    "cleaned_car_dataset.csv"
)

model = pickle.load(
    open("car_price_model.pkl", "rb")
)

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗"
)

st.title("🚗 Car Price Predictor")

st.write(
    "Predict the resale value of a used car."
)

name = st.selectbox(
    "Car Name",
    sorted(cars["name"].unique())
)

company = st.selectbox(
    "Company",
    sorted(cars["company"].unique())
)

year = st.slider(
    "Year",
    1995,
    2026,
    2018
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(
        cars["fuel_type"].unique()
    )
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

    prediction = model.predict(data)

    st.success(
        f"Estimated Price: ₹{int(prediction[0]):,}"
    )