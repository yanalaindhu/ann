import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

st.set_page_config(
    page_title="Air Passenger Forecast",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Air Passenger Forecasting")

# ==========================
# LOAD MODEL
# ==========================

try:
    model = tf.keras.models.load_model(
        "models/model.keras",
        compile=False
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    st.success("Model Loaded Successfully")

except Exception as e:

    st.error("Model Loading Error")

    st.code(str(e))

    st.stop()

# ==========================
# INPUTS
# ==========================

st.subheader(
    "Enter Previous 5 Months Passenger Counts"
)

p1 = st.number_input(
    "Month 1",
    value=112
)

p2 = st.number_input(
    "Month 2",
    value=118
)

p3 = st.number_input(
    "Month 3",
    value=132
)

p4 = st.number_input(
    "Month 4",
    value=129
)

p5 = st.number_input(
    "Month 5",
    value=121
)

# ==========================
# PREDICT
# ==========================

if st.button("Predict"):

    values = np.array([
        [p1],
        [p2],
        [p3],
        [p4],
        [p5]
    ])

    scaled = scaler.transform(
        values
    )

    X = scaled.reshape(
        1,
        5
    )

    prediction = model.predict(
        X,
        verbose=0
    )

    forecast = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )

    st.success(
        f"Predicted Passengers: {forecast[0][0]:.2f}"
    )
