import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Air Passenger Forecast",
    page_icon="✈️"
)

# Load files
model = tf.keras.models.load_model(
    "models/model.keras",
    compile=False
)

scaler = joblib.load(
    "models/scaler.pkl"
)

st.title(
    "✈️ Air Passenger Forecasting"
)

st.write(
    "Enter previous 5 months passenger values"
)

p1 = st.number_input(
    "Month 1",
    value=100
)

p2 = st.number_input(
    "Month 2",
    value=110
)

p3 = st.number_input(
    "Month 3",
    value=120
)

p4 = st.number_input(
    "Month 4",
    value=130
)

p5 = st.number_input(
    "Month 5",
    value=140
)

if st.button(
    "Predict"
):

    values = np.array(
        [[p1], [p2], [p3], [p4], [p5]]
    )

    values_scaled = scaler.transform(
        values
    )

    X = values_scaled.reshape(
        1,
        5
    )

    pred = model.predict(
        X,
        verbose=0
    )

    forecast = scaler.inverse_transform(
        pred.reshape(-1, 1)
    )

    st.success(
        f"Predicted Passengers: {forecast[0][0]:.2f}"
    )