import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model (assuming it's already trained and saved)
model = load_model('tqq_model.h5')

# Load your new test CSV (with Date, Open, High, Low, Close, AdjClose, Volume columns)
test_df = pd.read_csv('tqqq.csv')  # Replace with your actual file path

# Convert the 'Date' column to datetime and set it as the index (optional)
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df.set_index('Date', inplace=True)

# Extract the 'Close' prices (this is the feature used for prediction)
close_data = test_df[['Close']].values  # Use only the 'Close' column

# Normalize the 'Close' prices using the same scaler from training
scaler = MinMaxScaler(feature_range=(0, 1))
close_data_scaled = scaler.fit_transform(close_data)  # Fit and transform the 'Close' data

# Create sequences from the normalized 'Close' prices (same as training data)
SEQ_LENGTH = 30  # This should match the SEQ_LENGTH used during training

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])  # Predict the next 'Close' value
    return np.array(sequences), np.array(labels)

# Create sequences from the scaled 'Close' data
X_test, y_test = create_sequences(close_data_scaled, SEQ_LENGTH)

# Make predictions with the trained model
predictions = model.predict(X_test)

# Rescale predictions back to the original 'Close' values
predictions_rescaled = scaler.inverse_transform(predictions)  # Inverse transform the predictions

# Visualize the results (predictions vs actual 'Close' prices)
# Here we use the last 'SEQ_LENGTH' data points for actual 'Close' values
actual_data = close_data[SEQ_LENGTH:]  # Actual 'Close' values from the test data (matching the sequence length)

plt.figure(figsize=(12, 6))
plt.plot(test_df.index[SEQ_LENGTH:], actual_data, label="Actual Close Prices")
plt.plot(test_df.index[SEQ_LENGTH:], predictions_rescaled, label="Predicted Close Prices")
plt.legend()
plt.title("Stock Price Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()
