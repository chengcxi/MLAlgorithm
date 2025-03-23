from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns

#Panda describe data
tqqq = pd.read_csv('tqqq.csv')
print(tqqq.head())
tqqq.shape
tqqq.info()
tqqq.describe()

#Open/Close price chart
tqqq['Date'] = pd.to_datetime(tqqq['Date'])
plt.plot(tqqq['Date'], 
        tqqq['Open'], 
        color="blue", 
        label="Open") 
plt.plot(tqqq['Date'], 
        tqqq['High'], 
        color="green", 
        label="High") 
plt.plot(tqqq['Date'], 
        tqqq['Low'], 
        color="red", 
        label="Low") 
plt.plot(tqqq['Date'], 
        tqqq['Close'], 
        color="black", 
        label="Close") 
plt.title("tqqq Open-Close Stock") 
plt.legend() 
plt.show()

# prepare the training set samples 
tqqqClose = tqqq.filter(['Close']) 
dataset = tqqqClose.values 
training = int(np.ceil(len(dataset) * .95)) 

# scale the data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :] 

x0_train = [] 
y_train = [] 

# considering 60 as the batch size, 
# create x1_train and y_train 
for i in range(60, len(train_data)): 
    x0_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0]) 

x_train, y_train = np.array(x0_train), np.array(y_train) 
x1_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Use x_train instead

#RNN Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x1_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

print(model.summary())

#Analyze loss & optimization

#Custom loss function
# def my_loss_fn(y_true, y_pred):
#     squared_difference = ops.square(y_true - y_pred)
#     return ops.mean(squared_difference, axis=-1)

optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mae', metrics=[keras.metrics.RootMeanSquaredError()])
history = model.fit(x1_train, y_train, epochs=10, batch_size=32, validation_split=0.1) 

#Evaluate prediction
testing = scaled_data[training-60:,:]
x0_test = []
y_test = dataset[training:, :]

for i in range(60, len(testing)):
        x0_test.append(testing[i-60:i, 0])

x0_test = np.array(x0_test)
x1_test = np.reshape(x0_test, (x0_test.shape[0], x0_test.shape[1], 1))

prediction = model.predict(x1_test)

#Predictions
train = tqqq[:training]
test = tqqq[training:].copy()
test.loc[:, 'Predictions'] = prediction

#Plot prediction
plt.figure(figsize=(12, 8))

# Convert Date column to datetime (if not already converted)
tqqq['Date'] = pd.to_datetime(tqqq['Date'])

# Plot Open and Close prices
plt.plot(tqqq['Date'], tqqq['Open'], color="blue", label="Open")
plt.plot(tqqq['Date'], tqqq['Close'], color="black", label="Close")
plt.plot(tqqq['Date'], tqqq['High'], color="green", label="High")
plt.plot(tqqq['Date'], tqqq['Low'], color="red", label="Low")

# Plot test and predictions with proper alignment
plt.plot(tqqq['Date'][training:], test['Close'], color="orange", label="Test (Actual)")
plt.plot(tqqq['Date'][training:], test['Predictions'], color="purple", label="Predictions")

# Title, labels, and legend
plt.title("TQQQ Stock Price with Test and Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()