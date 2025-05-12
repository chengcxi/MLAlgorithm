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
scaled_vix = scaler.fit_transform(vix_df[doscale])
# Converts scaled_tqqq(numpy array) to polars dataframe
scaled_tqqq_df = pl.DataFrame(scaled_tqqq, schema=doscale)
scaled_vix_df = pl.DataFrame(scaled_vix, schema=doscale)
# Add VIX data to the scaled dataframe
merged_for_gbdt = df.join(vix_df, on='Date', how='inner', suffix='_vix')

# Define features for GBDT (original VIX column names, will be suffixed)
gbdt_vix_original_features = [col for col in vix_df.columns if col not in ['Date', 'Volume']]
# These are the feature names as they appear in merged_for_gbdt, used for selecting X
gbdt_X_feature_names = [f"{col}_vix" for col in gbdt_vix_original_features]

X_gbdt = merged_for_gbdt.select(gbdt_X_feature_names).to_numpy()
y_gbdt = merged_for_gbdt['Close'].to_numpy() # TQQQ Close is the target

split_idx_gbdt = int(len(X_gbdt) * 0.8)
X_train_gbdt, X_test_gbdt = X_gbdt[:split_idx_gbdt], X_gbdt[split_idx_gbdt:]
y_train_gbdt, y_test_gbdt = y_gbdt[:split_idx_gbdt], y_gbdt[split_idx_gbdt:]

# Train GBDT
gbdt = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbdt.fit(X_train_gbdt, y_train_gbdt)

# Predict using GBDT on the entire merged_for_gbdt set to get GBDT_Pred column
gbdt_preds_all = gbdt.predict(X_gbdt)
merged_for_gbdt = merged_for_gbdt.with_columns(pl.Series("GBDT_Pred", gbdt_preds_all))

merged = merged_for_gbdt

# --- LSTM Preprocessing ---
# Define features to scale for LSTM
# Noscale should include original VIX columns if they are still in 'merged' from a different join.
# Assuming 'merged' columns are: TQQQ_cols, VIX_cols_suffixed, GBDT_Pred
noscale = ['Date','Volume','Volume_vix'] # Volume from TQQQ, Volume_vix from VIX
doscale = [col for col in merged.columns if col not in noscale]

# Scale the 'merged' dataframe (which now includes GBDT_Pred)
scaler = MinMaxScaler(feature_range=(0,1))
# Important: Fit the scaler on the data that will be used to create sequences.
# Ensure 'doscale' only contains numeric columns intended for scaling.
# Convert to numpy for scaler, handling potential non-numeric types if any slipped into doscale
try:
    doscale_numeric_data = merged.select(doscale).to_numpy()
except pl.ComputeError as e:
    print(f"Error converting doscale columns to numpy. Check for non-numeric types in 'doscale': {doscale}")
    print(f"Columns in 'merged': {merged.columns}")
    print(f"doscale list: {doscale}")
    raise e

scaled_merged_values = scaler.fit_transform(doscale_numeric_data)
scaled_features_df = pl.DataFrame(scaled_merged_values, schema=doscale) # This df contains all features for LSTM, scaled

# Make new scaler
# The reason we need a new scaler because the first one operated on an array with shape (n,5)
# but now need one for array with shape (n,1) because thats our output shape
# we can use the .fit function to only fit it to close prices
close_scaler = MinMaxScaler(feature_range=(0,1))
close_scaler.fit(merged.select("Close").to_numpy())

# Make Sequences
# Partitions data into sequences meant to predict the value directly after them
# Takes numpy array or dataframe in data, and sequence length in seq_length
def make_sequences(scaled_data_df: pl.DataFrame, full_merged_df: pl.DataFrame, seq_length: int,
                   doscale_list: list, target_scaler: MinMaxScaler):
    sequences = []
    labels = []

    np_scaled_data = scaled_data_df.select(doscale_list).to_numpy()
    # Get unscaled close prices from the *full* merged dataframe to ensure alignment
    unscaled_close_prices_for_labels = full_merged_df['Close'].to_numpy()

    for i in range(len(np_scaled_data) - seq_length):
        sequences.append(np_scaled_data[i:i + seq_length])

        # Get the unscaled close price for the label
        price_label_unscaled = unscaled_close_prices_for_labels[i + seq_length]
        # Scale it using the dedicated close_scaler
        scaled_price_label = target_scaler.transform([[price_label_unscaled]])[0][0]
        labels.append(scaled_price_label)

    return np.array(sequences), np.array(labels)

# Length of the sequences we want for this LSTM (1 month)
length = 110

# Converting scaled dataframe into sets of sequences and labels
seqSet, labSet = make_sequences(scaled_features_df, merged, length, doscale, close_scaler)

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
    x = seqSetTrain,
    y = labSetTrain,
    epochs = 20, # more epochs for real training
    batch_size=32, # Added batch_size
    validation_split=0.1 # Optional: use part of training data for validation during training
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

# Get the last available sequence from test data
last_sequence = seqSetTest[-1:]  # Shape: [1, length, n_features] # length is 110

# Number of future days to predict
future_steps = 110

# Make a copy of the last sequence to avoid modifying original
current_sequence = last_sequence.copy()

# Array to store unscaled TQQQ close predictions
future_predictions_unscaled = []

# --- Preparation for the modified loop ---
# 'doscale' list defines the feature order in sequences.
# 'doscale' is:
# ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Open_vix', 'High_vix', 'Low_vix', 'Close_vix', 'AdjClose_vix', 'GBDT_Pred']

tqqq_close_idx = doscale.index('Close') # TQQQ's own close price
gbdt_pred_idx = doscale.index('GBDT_Pred')

# VIX feature names in 'doscale' (suffixed) and their original names in 'vix_df'
# These are the VIX columns from 'vix_df' that were used to create the GBDT features
# Make sure this list is in the same order as GBDT model expects its input features
# gbdt_vix_original_features was defined earlier: [col for col in vix_df.columns if col not in ['Date', 'Volume']]
# So, this should be ['Open', 'High', 'Low', 'Close', 'Adj Close'] (or similar, check vix.csv headers)

# Map from the suffixed names in 'doscale' to original names in 'vix_df'
# Example: if gbdt_vix_original_features = ['Open', 'High', ..., 'Adj Close']
# and corresponding doscale names are 'Open_vix', 'High_vix', ..., 'Adj Close_vix'
doscale_vix_feature_names_suffixed = [f"{col}_vix" for col in gbdt_vix_original_features]

# Check if all these suffixed names are actually in 'doscale'
missing_suffixed_vix_in_doscale = [name for name in doscale_vix_feature_names_suffixed if name not in doscale]
if missing_suffixed_vix_in_doscale:
    raise ValueError(f"Suffixed VIX feature names {missing_suffixed_vix_in_doscale} not found in 'doscale' list ({doscale}). Check suffixing and 'doscale' definition.")

vix_feature_indices_in_doscale = [doscale.index(name) for name in doscale_vix_feature_names_suffixed]

# Generate future dates
# Ensure 'merged' is sorted by 'Date' to get the correct last date.
# The initial sort of df and vix_df and inner join should maintain this.
last_date_in_merged_data = merged['Date'][-1]
future_dates = [last_date_in_merged_data + timedelta(days=i) for i in range(1, future_steps + 1)]

# --- Modified Future Prediction Loop ---
print(f"Starting future prediction loop for {future_steps} steps...")
future_predictions = []

for k in range(future_steps):
    # Predict TQQQ Close (scaled) for the next step
    predicted_tqqq_close_scaled = model.predict(current_sequence, verbose=0)[0, 0]

    # Unscale the prediction
    predicted_tqqq_close_unscaled = close_scaler.inverse_transform([[predicted_tqqq_close_scaled]])[0, 0]
    future_predictions_unscaled.append(predicted_tqqq_close_unscaled)

    # If this is the last prediction needed, break
    if k == future_steps - 1:
        break

    # --- Prepare the next input sequence ---
    # Roll the sequence to make space for the new day's features
    current_sequence = np.roll(current_sequence, -1, axis=1)
    # `new_last_slice_features_scaled` is a view into `current_sequence`
    new_last_slice_features_scaled = current_sequence[0, -1, :]

    # 1. Update TQQQ 'Close' feature with the latest (scaled) prediction
    new_last_slice_features_scaled[tqqq_close_idx] = predicted_tqqq_close_scaled

    # 2. Get VIX data for the current future day
    current_future_day_for_vix = future_dates[k] # This is the date for which we need VIX data
                                                 # to form features for predicting TQQQ on day k+1

    actual_vix_row_df = vix_df.filter(pl.col("Date") == current_future_day_for_vix)

    if not actual_vix_row_df.is_empty():
        if k < 3 or k % 20 == 0 or k == future_steps -2 :
             print(f"  Step {k+1}/{future_steps}: Using actual VIX for {current_future_day_for_vix.strftime('%Y-%m-%d')}")

        # Prepare unscaled VIX features for GBDT input
        gbdt_input_unscaled_vix_values = []
        for original_vix_name in gbdt_vix_original_features: # Iterate in the order GBDT expects
            gbdt_input_unscaled_vix_values.append(actual_vix_row_df.select(original_vix_name).item())

        # 2a. Update GBDT_Pred based on actual VIX
        # GBDT expects a 2D array of unscaled VIX values
        gbdt_pred_unscaled = gbdt.predict(np.array([gbdt_input_unscaled_vix_values]))[0]
        # Scale the new GBDT prediction using the main 'scaler'
        # Correct scaling: value * scaler.scale_[idx] + scaler.min_[idx]
        scaled_gbdt_pred = gbdt_pred_unscaled * scaler.scale_[gbdt_pred_idx] + scaler.min_[gbdt_pred_idx]
        new_last_slice_features_scaled[gbdt_pred_idx] = scaled_gbdt_pred

        # 2b. Update individual VIX features (Open_vix, High_vix, etc.)
        for i, suffixed_vix_name in enumerate(doscale_vix_feature_names_suffixed):
            original_vix_name = gbdt_vix_original_features[i] # Get corresponding original name
            feature_idx_in_doscale = vix_feature_indices_in_doscale[i] # Index in 'doscale'

            unscaled_actual_vix_value = actual_vix_row_df.select(original_vix_name).item()

            # Scale this VIX value using the main 'scaler'
            # Correct scaling: value * scaler.scale_[idx] + scaler.min_[idx]
            scaled_actual_vix_value = unscaled_actual_vix_value * scaler.scale_[feature_idx_in_doscale] + scaler.min_[feature_idx_in_doscale]
            new_last_slice_features_scaled[feature_idx_in_doscale] = scaled_actual_vix_value
    else:
        # If no actual VIX data, VIX features and GBDT_Pred in new_last_slice_features_scaled
        # will retain their rolled-over values from the previous timestep.
        if k < 3 or k % 20 == 0 or k == future_steps -2 :
            print(f"  Step {k+1}/{future_steps}: No VIX data for {current_future_day_for_vix.strftime('%Y-%m-%d')}. VIX features & GBDT_Pred rolled over.")
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

    # Other TQQQ features (Open, High, Low, Adj Close) in new_last_slice_features_scaled
    # are currently rolled over. If you have models to predict them, they could be updated here too.

# Convert list of unscaled predictions to numpy array
future_predictions_unscaled = np.array(future_predictions_unscaled)
# Plot historical prices and future predictions
plt.figure(figsize=(14,7))
plt.plot(merged['Date'][-100:], merged['Close'][-100:], 'b-', label='Historical Prices')
plt.plot(future_dates, future_predictions_unscaled, 'r-', label='Predicted Future Prices')
plt.plot(vix_df['Date'][-171:], vix_df['Close'][-171:], 'g-', label='VIX Historical Prices')
plt.axvline(x=last_date_in_merged_data, color='k', linestyle='--', label='Prediction Start')
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
for date, price in zip(future_dates, future_predictions_unscaled):
    print(f"| {date.strftime('%Y-%m-%d')} | ${price:>14.2f} |")
print("+" + "-"*32 + "+")