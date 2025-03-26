import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model from file
model = load_model('tqq_model.h5')  # You might need to retrain your model!

# Load csv file using Polars
test_df = pl.read_csv('tqqq.csv')

# Convert the 'Date' column to datetime
test_df = test_df.with_columns(pl.col("Date").str.to_datetime("%Y-%m-%d").alias("Date"))

# 1. Include BOTH Open and Close
data = test_df.select(["Open", "Close"]).to_numpy()

# 2. Scale Open and Close (separately is usually better)
scaler_open = MinMaxScaler(feature_range=(0, 1))
scaler_close = MinMaxScaler(feature_range=(0, 1))  # Separate scaler for Close

data[:, 0] = scaler_open.fit_transform(data[:, 0].reshape(-1, 1)).flatten()  # Scale Open
data[:, 1] = scaler_close.fit_transform(data[:, 1].reshape(-1, 1)).flatten()  # Scale Close

SEQ_LENGTH = 30

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length, 0])  # Only Open prices for the sequence
        labels.append(data[i + seq_length -1, 1])    # Close price of the *last* day in sequence
    return np.array(sequences), np.array(labels)

X_test, y_test = create_sequences(data, SEQ_LENGTH)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions with the trained model (you'll likely need to retrain)
predictions = model.predict(X_test)

# Rescale predictions back to the original *Close* values
predictions_rescaled = scaler_close.inverse_transform(predictions)

# Get the actual *close* prices for comparison
actual_data = scaler_close.inverse_transform(data[SEQ_LENGTH-1:-1, [1]]) # Correct indexing for actual close

# Get the dates, adjusted for the sequence length
dates = test_df.get_column("Date").to_numpy()[SEQ_LENGTH-1:-1]

plt.figure(figsize=(12, 6))
plt.plot(dates, actual_data, label="Actual Close Prices")
plt.plot(dates, predictions_rescaled, label="Predicted Close Prices")
plt.legend()
plt.title("Stock Price Prediction (Open to Close)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()