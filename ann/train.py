import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv(
    "data/AirPassengers.csv"
)

data = df["#Passengers"].values.reshape(
    -1,
    1
)

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(
    data
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

window = 5

X = []
y = []

for i in range(
    window,
    len(data_scaled)
):

    X.append(
        data_scaled[
            i-window:i,
            0
        ]
    )

    y.append(
        data_scaled[
            i,
            0
        ]
    )

X = np.array(X)
y = np.array(y)

model = Sequential()

model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(5,)
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X,
    y,
    epochs=50,
    batch_size=8
)

model.save(
    "models/model.keras"
)

print("Training Completed")
