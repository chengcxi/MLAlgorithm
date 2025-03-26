from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import polars as pl
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load Data using Polars
tqqq = pl.read_csv("tqqq.csv")
tqqq = tqqq.with_columns(pl.col("Date").str.to_datetime()) # Convert Date column to datetime

#Graph0 (Graphical Representation of the data)
print(tqqq.head())
print(tqqq.shape)
print(tqqq.describe())

# Sort Data by Date & extract features (Opening Price) and labels (Low, High, Close)
tqqq = tqqq.sort("Date")
features = tqqq.select(["Open"]).to_numpy()  # Input feature: Open price
labels = tqqq.select(["Low", "High", "Close"]).to_numpy()  # Output labels: Low, High, Close

# Convert to NumPy arrays and reshape
features = np.array(features).reshape(-1, 1)
labels = np.array(labels).reshape(-1, 3)

# Split into training and testing & normalize data using MinMaxScaler
training_size = int(np.ceil(len(features) * 0.95))  # 95% for training

feature_scaler = MinMaxScaler(feature_range=(0, 1))
label_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(features)
scaled_labels = label_scaler.fit_transform(labels)

# Create training sequences
SEQ_LENGTH = 120  # Lookback period

x_train, y_train = [], []
for i in range(SEQ_LENGTH, training_size):
    x_train.append(scaled_features[i - SEQ_LENGTH:i, 0])  # Last 120 days of Open prices
    y_train.append(scaled_labels[i, :])  # Predict Low, High, Close

# Convert to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # Reshape for LSTM

# Build LSTM Model
model = keras.models.Sequential([
    keras.layers.LSTM(units=64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    keras.layers.LSTM(units=64),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3)  # Output 3 values: Low, High, Close
])

print(model.summary())

# Compile & train the model
optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)
model.compile(optimizer=optimizer, loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

history = model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.1)

# Prepare Test Data
test_data = scaled_features[training_size - SEQ_LENGTH:]  # Use last 120 points

x_test = []
for i in range(SEQ_LENGTH, len(test_data)):
    x_test.append(test_data[i - SEQ_LENGTH:i, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Predict Future Prices
future_predictions = []
current_batch = test_data[-SEQ_LENGTH:].reshape((1, SEQ_LENGTH, 1))

for i in range(30):  # Predict next 30 days
    pred = model.predict(current_batch)[0]  # Predict Low, High, Close
    future_predictions.append(pred)

    # Shift input sequence and add the prediction
    new_sequence = np.append(current_batch[:, 1:, :], [[[pred[2]]]], axis=1)  # Use predicted Close
    current_batch = new_sequence

# Inverse transform predictions
future_predictions = label_scaler.inverse_transform(np.array(future_predictions))

# Write future dates
last_date = tqqq["Date"].to_list()[-1]
future_dates = [last_date + BDay(i) for i in range(1, 31)]

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(tqqq["Date"], tqqq["Open"], color="blue", label="Open Price")
plt.plot(tqqq["Date"], tqqq["Close"], color="black", label="Close Price")
plt.plot(tqqq["Date"], tqqq["High"], color="green", label="High Price")
plt.plot(tqqq["Date"], tqqq["Low"], color="red", label="Low Price")

# Plot predicted values
plt.plot(future_dates, future_predictions[:, 0], color="purple", label="Predicted Low")
plt.plot(future_dates, future_predictions[:, 1], color="orange", label="Predicted High")
plt.plot(future_dates, future_predictions[:, 2], color="brown", label="Predicted Close")

plt.title("TQQQ Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()