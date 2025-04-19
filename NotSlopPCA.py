from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

tqqq = pd.read_csv('uptodatetqqq.csv')
pca = pd.read_csv('pca_results.csv')
pca_df = pd.DataFrame(pca)
df = pd.DataFrame(tqqq)
pca_df = pca_df.sort_values('Date')
df = df.sort_values('Date')
pca_df = pca_df.set_index('Date')
df =  df.set_index('Date')
df = df.join(pca_df, how='inner')
df = df.bfill().ffill()
df.index = pd.to_datetime(df.index)
print(df)
scaler = MinMaxScaler(feature_range=(0,1))
noscale = ['Date']
doscale = [col for col in df.columns if col not in noscale]
scaled_tqqq = scaler.fit_transform(df[doscale])
scaled_df = pd.DataFrame(scaled_tqqq, columns=doscale)

close_scaler = MinMaxScaler(feature_range=(0,1))
close_scaler.fit(df[['Close']].values)

def make_sequences(data, seq_length):
    if isinstance(data, pd.DataFrame):
        data = data.values
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        price_2d = [[df['Close'].iloc[i + seq_length]]]

        scaled_2d = close_scaler.transform(price_2d)

        scaled_value = scaled_2d[0][0]
        
        labels.append(scaled_value)
    return np.array(sequences), np.array(labels)

length = 30

seqSet, labSet = make_sequences(scaled_df, length)

train_split = int(len(seqSet) * 0.8)
indices = np.random.permutation(len(seqSet))

seqSetTrain, seqSetTest = seqSet[indices[:train_split]], seqSet[indices[train_split:]]
labSetTrain, labSetTest = labSet[indices[:train_split]], labSet[indices[train_split:]]
print(labSetTest.shape)

seqSetTrain = seqSetTrain.reshape(seqSetTrain.shape[0], seqSetTrain.shape[1], seqSetTrain.shape[2])
seqSetTest = seqSetTest.reshape(seqSetTest.shape[0], seqSetTest.shape[1], seqSetTest.shape[2])

model = keras.models.Sequential([
    keras.layers.LSTM(128, input_shape=(length, seqSetTest.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.005),
    loss = keras.losses.MeanSquaredError()
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

    

history = model.fit(
    x=seqSetTrain,
    y=labSetTrain,
    validation_data=(seqSetTest, labSetTest),
    epochs=75,
    callbacks=[early_stopping],
    batch_size=32
)

predictions = model.predict(seqSetTest)

predict_rescale = close_scaler.inverse_transform(predictions)

labSetTest_rescale = close_scaler.inverse_transform(labSetTest.reshape(-1,1))

test_dates = df.index[train_split + length:]
plt.figure(figsize=(12,6))
plt.plot(test_dates, labSetTest_rescale, label = 'Actual Prices')
plt.plot(test_dates, predict_rescale, label = 'Predicted Prices')
plt.legend()
plt.title('Stock Price Prediction from LSTM - Test Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.index = pd.to_datetime(df.index)

last_sequence = seqSetTest[-1:]  # Shape: [1, 30, n_features]

future_steps = 30

current_sequence = last_sequence.copy()

future_predictions = []

for _ in range(future_steps):
    current_pred = model.predict(current_sequence, verbose=0)
    
    current_pred_unscaled = close_scaler.inverse_transform(current_pred)[0][0]
    future_predictions.append(current_pred_unscaled)
    
    current_sequence = np.roll(current_sequence, -1, axis=1)
    
    current_sequence[0, -1, 3] = current_pred[0][0]

future_predictions = np.array(future_predictions)

last_date = df.index[-1]
future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, future_steps+1)]

plt.figure(figsize=(14,7))
plt.plot(df.index[-100:], df['Close'][-100:], 'b-', label='Historical Prices')
plt.plot(future_dates, future_predictions, 'r-', label='Predicted Future Prices')
plt.axvline(x=last_date, color='k', linestyle='--', label='Prediction Start')
plt.legend()
plt.title(f'Stock Price Prediction - Next {future_steps} Days')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nFuture Price Predictions:")
print("+" + "-"*32 + "+")
print("| Date       | Predicted Price |")
print("+" + "-"*32 + "+")
for date, price in zip(future_dates, future_predictions):
    print(f"| {date.strftime('%Y-%m-%d')} | ${price:>14.2f} |")
print("+" + "-"*32 + "+")