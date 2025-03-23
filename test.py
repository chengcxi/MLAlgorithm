import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
df = pd.read_csv("tqqq.csv")  # Replace with your actual file
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select relevant feature(s) (e.g., 'Close' for price prediction)
data = df[['Close']].values  # Use other features if needed

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

SEQ_LENGTH = 30  # Number of past days to consider
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split data into train and test sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Rescale back to original values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_actual, label="Actual Prices")
plt.plot(df.index[-len(y_test):], predictions, label="Predicted Prices")
plt.legend()
plt.title("Stock Price Prediction with LSTM")
plt.show()
