from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import polars as pl
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

# Load Data using Polars
tqqq = pl.read_csv('tqqq.csv')

# Convert Date column to datetime
tqqq = tqqq.with_columns(pl.col("Date").str.to_datetime())

# Print dataset info
print(tqqq.head())
print(tqqq.shape)
print(tqqq.describe())

# Plot Open/Close price chart
plt.figure(figsize=(12, 6))
plt.plot(tqqq["Date"], tqqq["Open"], color="blue", label="Open")
plt.plot(tqqq["Date"], tqqq["High"], color="green", label="High")
plt.plot(tqqq["Date"], tqqq["Low"], color="red", label="Low")
plt.plot(tqqq["Date"], tqqq["Close"], color="black", label="Close")
plt.title("TQQQ Open-Close Stock Prices")
plt.legend()
plt.show()

# Prepare dataset for training
tqqqClose = tqqq.select("Close").to_numpy()  # Convert to NumPy array
dataset = np.array(tqqqClose).reshape(-1, 1)  # Ensure it's 2D
training = int(np.ceil(len(dataset) * 0.95))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create training data sequences
train_data = scaled_data[:training]

SEQ_LENGTH = 120  # Time step size
x0_train, y_train = [], []

for i in range(SEQ_LENGTH, len(train_data)):
    x0_train.append(train_data[i-SEQ_LENGTH:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x0_train), np.array(y_train)
x1_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model
model = keras.models.Sequential([
    keras.layers.LSTM(units=64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    keras.layers.LSTM(units=64),
    keras.layers.Dense(128),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

print(model.summary())

# Compile and train model
optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(x1_train, y_train, epochs=5, batch_size=16, validation_split=0.1)

# Prepare testing data
testing = scaled_data[training-SEQ_LENGTH:]
x0_test = []
y_test = dataset[training:]

for i in range(SEQ_LENGTH, len(testing)):
    x0_test.append(testing[i-SEQ_LENGTH:i, 0])

x0_test = np.array(x0_test)
x1_test = np.reshape(x0_test, (x0_test.shape[0], x0_test.shape[1], 1))

# Make predictions
# First, sort the data chronologically
tqqq = tqqq.sort("Date")

# Now check the date range
print(f"Data ranges from {tqqq['Date'][0]} to {tqqq['Date'][-1]}")

# Prepare dataset for training
tqqqClose = tqqq.select("Close").to_numpy()
dataset = np.array(tqqqClose).reshape(-1, 1)
training = int(np.ceil(len(dataset) * 0.95))  # Use 95% for training

# Print the training cutoff date to verify
print(f"Training cutoff date: {tqqq['Date'][training-1]}")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# For predictions, use the most recent data (at the end of the sorted dataframe)
last_sequence = scaled_data[-SEQ_LENGTH:]  # Take the most recent SEQ_LENGTH days
current_batch = last_sequence.reshape((1, SEQ_LENGTH, 1))

# Generate future dates starting from the last date in your dataset
last_date = tqqq["Date"].to_list()[-1]
print(f"Last date in dataset: {last_date}")
print(f"Predicting prices from: {last_date + BDay(1)}")

# Generate business days for the next 30 trading days
predicted_dates = []
next_date = last_date
for i in range(30):
    next_date = next_date + BDay(1)
    predicted_dates.append(next_date)

# Then continue with your predictions as before

future_predictions = []
for i in range(30):  # Predict 30 days
    # Predict the next value.
    current_pred = model.predict(current_batch)[0, 0]

    # Append the prediction to the list.
    future_predictions.append(current_pred)

    # Update the input sequence:  Remove the oldest value, add the new prediction.
    new_sequence = np.append(current_batch[:, 1:, :], [[[current_pred]]], axis=1)
    current_batch = new_sequence

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = tqqq["Date"].to_list()[training - 1]
predicted_dates = [last_date + BDay(i) for i in range(1, 31)]

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(tqqq["Date"], tqqq["Open"], color="blue", label="Open")
plt.plot(tqqq["Date"], tqqq["Close"], color="black", label="Close")
plt.plot(tqqq["Date"], tqqq["High"], color="green", label="High")
plt.plot(tqqq["Date"], tqqq["Low"], color="red", label="Low")

# Plot test and predictions properly
plt.plot(predicted_dates, future_predictions, color="orange", label="Predicted")

plt.title("TQQQ Stock Price with Test and Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()