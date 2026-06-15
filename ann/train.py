import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("data/AirPassengers.csv")

# Use passenger column
data = df["#Passengers"].values.reshape(-1, 1)

# Scaling
scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Create sequences
X = []
y = []

window_size = 5

for i in range(window_size, len(data_scaled)):
    X.append(
        data_scaled[i-window_size:i, 0]
    )
    y.append(
        data_scaled[i, 0]
    )

X = np.array(X)
y = np.array(y)

# ANN Model
model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(window_size,)
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        1
    )
)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    y,
    epochs=100,
    batch_size=8,
    verbose=1
)

model.save(
    "models/model.keras"
)

print("Training Completed")