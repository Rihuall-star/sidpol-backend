# ml_keras.py
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from ml_utils import preparar_mensual  # ya lo tienes

def entrenar_keras_total(col):
    df = preparar_mensual(col, modalidad=None)
    if df.empty:
        return None, None, None

    X = df[["t"]].values.astype("float32")
    y = df["total"].values.astype("float32")

    # Normalizar t
    X = (X - X.mean()) / X.std()

    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, verbose=0)

    # Preparamos futuro (t_future)
    t_last = df["t"].max()
    t_future = np.arange(t_last + 1, t_last + 7).reshape(-1, 1).astype("float32")
    X_future = (t_future - X.mean()) / X.std()
    y_future = model.predict(X_future).flatten()

    total_pred_keras = float(y_future.sum())
    return model, t_future.flatten().tolist(), y_future.tolist(), total_pred_keras
