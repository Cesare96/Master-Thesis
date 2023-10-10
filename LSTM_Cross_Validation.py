# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:47:38 2023

@author: cesar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense, LSTM

#%% LSTM 

# Load data

data = DJI
data = data.values.reshape(-1,1)
training_data_len = math.floor(len(data)*(1-0.3))
look_back = 60
batch_size = [8, 16, 32, 64, 128, 256] #64
epochs =[40, 50, 60, 100, 200, 300]#200

# Definisci il numero di fold per la Time Series Cross Validation
n_splits = 10 #5

# Crea uno scalatore per i dati
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create train data set
train_data = scaled_data[0:training_data_len,:]
test_data = scaled_data[training_data_len-look_back:,:]

# Crea la matrice delle feature e il target
train_X = []
train_y = []
for i in range(look_back, len(scaled_data)):
    train_X.append(scaled_data[i-look_back:i])
    train_y.append(scaled_data[i])
train_X, train_y = np.array(train_X), np.array(train_y)



# Crea il modello LSTM
def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# Esegui la Time Series Cross Validation
tscv = TimeSeriesSplit(n_splits=n_splits)
best_score = float('inf')
best_params = {}
for hidden_nodes in range(1,8):
    score = 0
    for train_index, val_index in tscv.split(train_X):
        X_train, y_train = train_X[train_index], train_y[train_index]
        X_val, y_val = train_X[val_index], train_y[val_index]
        model = create_model(hidden_nodes)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        score += model.evaluate(X_val, y_val, verbose=0)
    score /= n_splits 
    if score < best_score:
        best_score = score
        best_params = {'hidden_nodes': hidden_nodes, 'batch_size': batch_size, 'epochs': epochs}
print('I migliori parametri sono: ', best_params)

test_X = []
for i in range(look_back, len(test_data)):
    test_X.append(test_data[i-look_back:i,0])
    
test_X = np.array(test_X)
test_y = data[training_data_len:,:]

