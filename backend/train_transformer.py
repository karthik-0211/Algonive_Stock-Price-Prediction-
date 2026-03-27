# =========================
# ENV SETTINGS
# =========================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"   # 🔥 VERY IMPORTANT

# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
from sklearn.preprocessing import MinMaxScaler

print("TensorFlow version:", tf.__version__)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/RELIANCE.csv")

data = df[['Close']].values

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# =========================
# CREATE SEQUENCES
# =========================
X, y = [], []
window = 60

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# =========================
# TRANSFORMER MODEL
# =========================
inputs = Input(shape=(60, 1))

# Attention block
x = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
x = LayerNormalization()(x)

# Feed-forward
x = Dense(64, activation="relu")(x)
x = LayerNormalization()(x)

# Pooling instead of slicing (IMPORTANT FIX)
x = GlobalAveragePooling1D()(x)

# Output layer
outputs = Dense(1)(x)

model = Model(inputs, outputs)

# Compile
model.compile(optimizer='adam', loss='mse')

# =========================
# TRAIN
# =========================
model.fit(X, y, epochs=5, batch_size=32)

# =========================
# SAVE MODEL
# =========================
os.makedirs("model", exist_ok=True)
os.makedirs("scaler", exist_ok=True)

# 🔥 SAVE IN KERAS FORMAT (COMPATIBLE)
model.save("model/transformer_model.keras")

# Save scaler
joblib.dump(scaler, "scaler/scaler.pkl")

print("✅ Transformer model trained and saved successfully!")
