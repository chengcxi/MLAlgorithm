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
plt.plot(tqqq['date'], 
        tqqq['open'], 
        color="blue", 
        label="open") 
plt.plot(tqqq['date'], 
        tqqq['close'], 
        color="green", 
        label="close") 
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

x_train = [] 
y_train = [] 

# considering 60 as the batch size, 
# create the X_train and y_train 
for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0]) 

x_train, y_train = np.array(x_train), np.array(y_train) 
X_train = np.reshape(x_train, 
                    (x_train.shape[0], 
                    x_train.shape[1], 1)) 

#RNN Model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape
                            =(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

print(model.summary())


