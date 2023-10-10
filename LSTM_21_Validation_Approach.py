# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:04:06 2023

@author: cesar
"""
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.random import set_seed
from keras.models import Sequential 
from keras.layers import Dense, LSTM
from tensorflow.keras.initializers import GlorotUniform
from keras.callbacks import EarlyStopping

#%% 

data = DJI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_DJI
threshold_test = threshold_test_DJI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    #early_stop = EarlyStopping(monitior='val_loss', mode = 'min', patience = 15, verbose = 0)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False''', callbacks = [early_stop]''')
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_DJI = node
    
print('The optimal number of node is: ', best_n_nodes_DJI)

#%% 

data = FTSE
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_FTSE
threshold_test = threshold_test_FTSE
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_FTSE = node
    
print('The optimal number of node is: ', best_n_nodes_FTSE)

#%% 

data = FTSEMIB
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_FTSEMIB
threshold_test = threshold_test_FTSEMIB
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_FTSEMIB = node
    
print('The optimal number of node is: ', best_n_nodes_FTSEMIB)

#%% 

data = GDAXI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_GDAXI
threshold_test = threshold_test_GDAXI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_GDAXI = node
    
print('The optimal number of node is: ', best_n_nodes_GDAXI)

#%% 

data = SPX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_SPX
threshold_test = threshold_test_SPX
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_SPX = node
    
print('The optimal number of node is: ', best_n_nodes_SPX)

#%% 

data = HSI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_HSI
threshold_test = threshold_test_HSI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_HSI = node
    
print('The optimal number of node is: ', best_n_nodes_HSI)

#%% 

data = IBEX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_IBEX
threshold_test = threshold_test_IBEX
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_IBEX = node
    
print('The optimal number of node is: ', best_n_nodes_IBEX)

#%% 

data = IXIC
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_IXIC
threshold_test = threshold_test_IXIC
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_IXIC = node
    
print('The optimal number of node is: ', best_n_nodes_IXIC)

#%% 

data = N225
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_N225
threshold_test = threshold_test_N225
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_N225 = node
    
print('The optimal number of node is: ', best_n_nodes_N225)

#%% 

data = OMXC20
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = np.arange(1,8)
min_val_loss = float('inf')
best_n_nodes = None

threshold_train = threshold_train_OMXC20
threshold_test = threshold_test_OMXC20
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]
train, val = train_test_split(train_data, test_size = .20, shuffle = False)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(train, look_back)
val_X, val_y = create_dataset(val, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
val_X = np.reshape(val_X, (val_X.shape[0], val_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(node, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

for node in hidden_nodes:
    random.seed(1)
    set_seed(1)
    model = create_model(node)
    history = model.fit(train_X, train_y, batch_size = batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose = 0, shuffle = False)
    val_loss = np.min(history.history['val_loss'])
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_n_nodes_OMXC20 = node
    
print('The optimal number of node is: ', best_n_nodes_OMXC20)
