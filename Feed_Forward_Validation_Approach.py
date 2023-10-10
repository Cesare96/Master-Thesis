# -*- coding: utf-8 -*-
"""
Created on Tue May 16 01:30:08 2023

@author: cesar
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.random import set_seed
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
# from keras.callbacks import EarlyStopping

#%% 

data = DJI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')

threshold_train = threshold_train_DJI
threshold_test = threshold_test_DJI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train_data, validation_data = train_test_split(train_data, test_size = .20, shuffle=False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train_data, look_back)
val_X, val_y = create_dataset(validation_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X= np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=train_X.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    #early_stop = EarlyStopping(monitior='val_loss', mode = 'min', patience = 15, verbose = 0)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes = node
    
print('The optimal number of node is: ', best_n_nodes)