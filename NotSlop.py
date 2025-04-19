from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import polars as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import seaborn as sns

#Read table
tqqq = pl.read_csv('tqqq.csv', try_parse_dates= True)
tqqq.shape
tqqq.describe()
tqqq = tqqq.sort("Date")
df = pl.DataFrame(tqqq)
df = df.sort('Date')

vix = pl.read_csv('vix.csv', try_parse_dates= True)
vix = vix.sort("Date")
vix_df = pl.DataFrame(vix)
vix_df = vix_df.sort('Date')

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='TQQQ vs VIX Close Price')
plt.plot(vix_df['Date'], vix_df['Close'], label='VIX Close Price')
plt.title('TQQQ vs VIX Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Preprocess data
# Scales data to be floats from 0-1 because lstms work better with similar values across the board
scaler = MinMaxScaler(feature_range=(0,1))
# Data we dont want scaled
noscale = ['Date','Volume']
# Filters out data that we want scaled from the columns on the DF
doscale = [col for col in df.columns if col not in noscale]
# Uses scaler object and array of columns to scale desired columns' data
scaled_tqqq = scaler.fit_transform(df[doscale])
# Converts scaled_tqqq(numpy array) to pandas dataframe
scaled_df = pl.DataFrame(scaled_tqqq, schema=doscale)

# Add VIX data to the scaled dataframe
merged = df.join(vix_df, on='Date', how='inner', suffix='_vix')

# Gradient Boosting Regressor
vix_features = [col for col in vix_df.columns if col not in ['Date', 'Volume']]
X = merged.select([f"{col}_vix" for col in vix_features]).to_numpy()
y = merged['Close'].to_numpy()

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train GBDT
gbdt = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbdt.fit(X_train, y_train)

# Predict using GBDT
gbdt_preds = gbdt.predict(X)
merged = merged.with_columns(pl.Series("GBDT_Pred", gbdt_preds))

# Include GBDT predictions in the scaled dataframe
noscale = ['Date','Volume','Volume_vix']
doscale = [col for col in merged.columns if col not in noscale]

# Scale the merged dataframe
scaler = MinMaxScaler(feature_range=(0,1))
scaled_merged = scaler.fit_transform(merged[doscale])
scaled_df = pl.DataFrame(scaled_merged, schema=doscale)

# Make new scaler
# The reason we need a new scaler because the first one operated on an array with shape (n,5)
# but now need one for array with shape (n,1) because thats our output shape
# we can use the .fit function to only fit it to close prices
close_scaler = MinMaxScaler(feature_range=(0,1))
close_scaler.fit(merged.select("Close").to_numpy())

# Make Sequences
# Partitions data into sequences meant to predict the value directly after them
# Takes numpy array or dataframe in data, and sequence length in seq_length
def make_sequences(data: pl.DataFrame, seq_length: int):
    sequences = []
    labels = []

    # Convert to numpy
    np_data = data.select(doscale).to_numpy()
    close_prices = tqqq['Close'].to_numpy()

    for i in range(len(np_data) - seq_length):
        sequences.append(np_data[i:i + seq_length])

        # Get close price (not scaled yet) and scale it
        price = close_prices[i + seq_length]
        scaled_price = close_scaler.transform([[price]])[0][0]
        labels.append(scaled_price)

    return np.array(sequences), np.array(labels)

# Length of the sequences we want for this LSTM (1 month)
length = 110

# Converting scaled dataframe into sets of sequences and labels
seqSet, labSet = make_sequences(scaled_df, length)

# Split into Training and Test Sets
# Determine the split of how much of the sets do we want to be training
# 80% of the data is training in this case
train_split = int(len(seqSet) * 0.8)

# Splitting each set into training and testing sets respectively
seqSetTrain, seqSetTest = seqSet[:train_split], seqSet[train_split:]
labSetTrain, labSetTest = labSet[:train_split], labSet[train_split:]
print(labSetTest.shape)
# Reshape Sequence Data
# Lstm model expects data in the form (Sample, Time, Features)
# Idk how this function works
seqSetTrain = seqSetTrain.reshape(seqSetTrain.shape[0], seqSetTrain.shape[1], seqSetTrain.shape[2])
seqSetTest = seqSetTest.reshape(seqSetTest.shape[0], seqSetTest.shape[1], seqSetTest.shape[2])

# Define Model
# Sequential Model seems to be the simplest model in the Keras API so we use that now
model = keras.models.Sequential()

# Adding Layers to the model so it actually does something
# See (https://keras.io/api/)
# Only adding a few layers because I barely understand how it works
# Values here can be changed to increase efficiency of model as well as more layers
# Currently has minimal layers as this is meant more for leaning
model.add(keras.layers.LSTM(50, input_shape=(length, seqSetTest.shape[2])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

# Compile Model
# The .compile method sets the configuration desired for training
# again, see (https://keras.io/api/)
model.compile(
    # Choosing the optimizer model for the model to run on
    # "Adam optimization is a stochastic gradient descent method that is 
    # based on adaptive estimation of first-order and second-order moments.
    # According to Kingma et al., 2014, the method is "computationally efficient, 
    # has little memory requirement, invariant to diagonal rescaling of gradients, 
    # and is well suited for problems that are large in terms of data/parameters".""
    # Whatever the fuck this means ^^^^^^
    # Learning Rate can be tweaked to attempt to improve model accuracy or training speed
    optimizer = keras.optimizers.Adam(learning_rate=0.005),

    # Choosing a loss equation in this case, compares the mean of the squares of errors between label and preduiction
    loss = keras.losses.MeanSquaredError()
)
# Train Model
# In order to train the model we use the .fit method to fit the model to our data
# in a set number of epochs

history = model.fit(
    # Setting the training sets, x is the sequences and the y is the predicted values
    x = seqSetTrain,
    y = labSetTrain,

    # Set number of epoch aka how many "cycles" the model trains before its done
    epochs = 50,

    # See (https://keras.io/api/models/model_training_apis/#fit-method)
    # for more info on the fit method
)

# Predict from Tests
# Now that the model is trained, we can use it to predict our data using the test data
# partitioned earlier in the code
# We will use the .predict method for the keras.model class
predictions = model.predict(seqSetTest)

# In order to make the predictions useful we will need to rescale
# the predictions back to the original size
# Using the .inverse_transform method for our close_scaler MinMaxScaler object this is possible
predict_rescale = close_scaler.inverse_transform(predictions)

# We also need the y test values to be transformed back to scale because they
# are still in transformed form
# We call the reshape method here because the shape of the array (how many dimensions it has)
# is different from the shape the .inverse_transform method expects
labSetTest_rescale = close_scaler.inverse_transform(labSetTest.reshape(-1,1))

# Graph Results
# Now with this we can compare the models predicted data and the actual values
test_dates = merged['Date'][train_split + length:].to_numpy()
plt.figure(figsize=(12,6))
plt.plot(test_dates, labSetTest_rescale, label = 'Actual Prices')
plt.plot(test_dates, predict_rescale, label = 'Predicted Prices')
plt.legend()
plt.title('Stock Price Prediction from LSTM - Test Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Get the last available sequence from your test data
last_sequence = seqSetTest[-1:]  # Shape: [1, 30, n_features]

# Number of future days to predict
future_steps = 110

# Make a copy of the last sequence to avoid modifying original
current_sequence = last_sequence.copy()

# Array to store predictions
future_predictions = []

for _ in range(future_steps):
    # Get the prediction (scaled between 0-1)
    current_pred = model.predict(current_sequence, verbose=0)
    
    # Unscale the prediction using close_scaler
    current_pred_unscaled = close_scaler.inverse_transform(current_pred)[0][0]
    future_predictions.append(current_pred_unscaled)
    
    # Uplate the sequence:
    # 1. Remove first element by shifting left
    current_sequence = np.roll(current_sequence, -1, axis=1)
    
    # 2. Uplate the Close price in the last position of the sequence
    # Assuming Close price is the 4th feature (index 3)
    current_sequence[0, -1, 3] = current_pred[0][0]

# Convert to numpy array
future_predictions = np.array(future_predictions)

# Generate future dates starting from day after last available date
last_date = merged['Date'][-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

# Plot historical prices and future predictions
plt.figure(figsize=(14,7))
plt.plot(merged['Date'][-100:], merged['Close'][-100:], 'b-', label='Historical Prices')
plt.plot(future_dates, future_predictions, 'r-', label='Predicted Future Prices')
plt.plot(merged['Date'][-100:], merged['Close_vix'][-100:], 'g-', label='VIX Historical Prices')
plt.plot(future_dates, merged['Close_vix'][-future_steps:], 'g-', label='VIX Actual Prices')
plt.axvline(x=last_date, color='k', linestyle='--', label='Prediction Start')
plt.legend()
plt.title(f'Stock Price Prediction - Next {future_steps} Days')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print predictions in a table
print("\nFuture Price Predictions:")
print("+" + "-"*32 + "+")
print("| Date       | Predicted Price |")
print("+" + "-"*32 + "+")
for date, price in zip(future_dates, future_predictions):
    print(f"| {date.strftime('%Y-%m-%d')} | ${price:>14.2f} |")
print("+" + "-"*32 + "+")