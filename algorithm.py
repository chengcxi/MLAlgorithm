from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

#Panda describe data
tqqq = pd.read_csv('tqqq.csv', usecols=lambda col: col != 'Date')
print(tqqq.head())
tqqq.shape
tqqq.info()
tqqq.describe()

#Open/Close price chart
tqqq['date'] = pd.to_datetime(tqqq['date'])
plt.plot(tqqq['Date'], 
        tqqq['Open'], 
        color="blue", 
        label="Open") 
plt.plot(tqqq['Date'], 
        tqqq['Close'], 
        color="green", 
        label="Close") 
plt.title("tqqq Open-Close Stock") 
plt.legend() 

# prepare the training set samples 
tqqqClose = tqqq.filter(['close']) 
dataset = tqqqClose.values 
training = int(np.ceil(len(dataset) * .95)) 

# scale the data 
ss = StandardScaler() 
ss = ss.fit_transform(dataset) 

train_data = ss[0:int(training), :] 

x0_train = [] 
y_train = [] 

# considering 60 as the batch size, 
# create the X_train and y_train 
for i in range(60, len(train_data)): 
    x0_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0]) 

x_train, y_train = np.array(x0_train), np.array(y_train) 
x1_train = np.reshape(x0_train, 
                    (x0_train.shape[0], 
                    x0_train.shape[1], 1)) 

#RNN Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape
                            =(x1_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

print(model.summary())

#Analyze loss & optimization
model.compile(optimizer='adam', 
            loss='mae', 
            metrics=[keras.metrics.RootMeanSquaredError()]) 
history = model.fit(x1_train, y_train, 
                    epochs=20) 

#Evaluate prediction
testing = ss[training-60:,:]
x0_test = []
y_test = dataset[training:, :]

for i in range(60, len(testing)):
        x0_test.append(testing[i-60:i, 0])
x0_test = np.array(x0_test)
x1_test = np.reshape(x0_test, (x0_test.shape[0], x0_test.shape[1], 1))

prediction = model.predict(x1_test)

train = tqqq[:training]
test = tqqq[training:]
test['Predictions'] = prediction

plt.figure(figsize=(10, 8))
plt.plot(train['Close'], c="b")
plt.plot(test[['Close', 'Predictions']])
plt.title('TQQQ Stock Close Price')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])

