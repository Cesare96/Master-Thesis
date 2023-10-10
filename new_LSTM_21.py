# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:24:58 2023

@author: cesar
"""

import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.random import set_seed
from keras.models import Sequential 
from keras.layers import Dense, LSTM
from tensorflow.keras.initializers import GlorotUniform
from keras import regularizers

#%% DJI

data = DJI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

threshold_train = threshold_train_DJI
threshold_test = threshold_test_DJI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_DJI, train_y_DJI = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_DJI = np.reshape(train_X_DJI, (train_X_DJI.shape[0], train_X_DJI.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_DJI.shape[1], train_X_DJI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_DJI = create_model(hidden_nodes)
model_DJI.fit(train_X_DJI, train_y_DJI, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_DJI.predict(test_X)
predictions_DJI = scaler.inverse_transform(predictions)
test_y_DJI = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_DJI = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_DJI = mean_squared_error(test_y_DJI, predictions_DJI)
print(MSE_LSTM_1_DJI)

# QLIKE

y_forecastvalues = predictions_DJI
y_actualvalues = test_y_DJI
qlike_DJI_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_LSTM_1_Out.append(iteration)
qlike_DJI_LSTM_1_Out = np.array(qlike_DJI_LSTM_1_Out)
QLIKE_LSTM_1_DJI = sum(qlike_DJI_LSTM_1_Out)/len(y_actualvalues)

predictions = model_DJI.predict(train_X_DJI)
predictions_DJI_In = scaler.inverse_transform(predictions)
train_y_DJI_In = np.array(y_train_DJI)
MSE_LSTM_1_DJI_In = mean_squared_error(train_y_DJI_In, predictions_DJI_In)
mseDJI_LSTM_In = []
for i in np.arange(len(train_y_DJI_In)):
    mse = (predictions_DJI_In[i]-train_y_DJI_In[i])**2
    mseDJI_LSTM_In.append(mse)
mseDJI_LSTM_In = np.array(mseDJI_LSTM_In)

#%% FTSE

data = FTSE
data1 = np.array(FTSE)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

threshold_train = threshold_train_FTSE
threshold_test = threshold_test_FTSE
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_FTSE, train_y_FTSE = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_FTSE = np.reshape(train_X_FTSE, (train_X_FTSE.shape[0], train_X_FTSE.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_FTSE.shape[1], train_X_FTSE.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSE = create_model(hidden_nodes)
model_FTSE.fit(train_X_FTSE, train_y_FTSE, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_FTSE.predict(test_X)
predictions_FTSE = scaler.inverse_transform(predictions)
test_y_FTSE = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSE = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_FTSE = mean_squared_error(test_y_FTSE, predictions_FTSE)
print(MSE_LSTM_1_FTSE)

# QLIKE

y_forecastvalues = predictions_FTSE
y_actualvalues = test_y_FTSE
qlike_FTSE_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_LSTM_1_Out.append(iteration)
qlike_FTSE_LSTM_1_Out = np.array(qlike_FTSE_LSTM_1_Out)
QLIKE_LSTM_1_FTSE = sum(qlike_FTSE_LSTM_1_Out)/len(y_actualvalues)

predictions = model_FTSE.predict(train_X_FTSE)
predictions_FTSE_In = scaler.inverse_transform(predictions)
train_y_FTSE_In = np.array(y_train_FTSE)
MSE_LSTM_1_FTSE_In = mean_squared_error(train_y_FTSE_In, predictions_FTSE_In)
mseFTSE_LSTM_In = []
for i in np.arange(len(train_y_FTSE_In)):
    mse = (predictions_FTSE_In[i]-train_y_FTSE_In[i])**2
    mseFTSE_LSTM_In.append(mse)
mseFTSE_LSTM_In = np.array(mseFTSE_LSTM_In)

#%% FTSEMIB

data = FTSEMIB
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

threshold_train = threshold_train_FTSEMIB
threshold_test = threshold_test_FTSEMIB
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_FTSEMIB, train_y_FTSEMIB = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_FTSEMIB = np.reshape(train_X_FTSEMIB, (train_X_FTSEMIB.shape[0], train_X_FTSEMIB.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_FTSEMIB.shape[1], train_X_FTSEMIB.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSEMIB = create_model(hidden_nodes)
model_FTSEMIB.fit(train_X_FTSEMIB, train_y_FTSEMIB, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_FTSEMIB.predict(test_X)
predictions_FTSEMIB = scaler.inverse_transform(predictions)
test_y_FTSEMIB = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSEMIB = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_FTSEMIB = mean_squared_error(test_y_FTSEMIB, predictions_FTSEMIB)
print(MSE_LSTM_1_FTSEMIB)

# QLIKE 

y_forecastvalues = predictions_FTSEMIB
y_actualvalues = test_y_FTSEMIB
qlike_FTSEMIB_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_LSTM_1_Out.append(iteration)
qlike_FTSEMIB_LSTM_1_Out = np.array(qlike_FTSEMIB_LSTM_1_Out)
QLIKE_LSTM_1_FTSEMIB = sum(qlike_FTSEMIB_LSTM_1_Out)/len(y_actualvalues)

predictions = model_FTSEMIB.predict(train_X_FTSEMIB)
predictions_FTSEMIB_In = scaler.inverse_transform(predictions)
train_y_FTSEMIB_In = np.array(y_train_FTSEMIB)
MSE_LSTM_1_FTSEMIB_In = mean_squared_error(train_y_FTSEMIB_In, predictions_FTSEMIB_In)
mseFTSEMIB_LSTM_In = []
for i in np.arange(len(train_y_FTSEMIB_In)):
    mse = (predictions_FTSEMIB_In[i]-train_y_FTSEMIB_In[i])**2
    mseFTSEMIB_LSTM_In.append(mse)
mseFTSEMIB_LSTM_In = np.array(mseFTSEMIB_LSTM_In)

#%% GDAXI

data = GDAXI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

threshold_train = threshold_train_GDAXI
threshold_test = threshold_test_GDAXI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_GDAXI, train_y_GDAXI = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_GDAXI = np.reshape(train_X_GDAXI, (train_X_GDAXI.shape[0], train_X_GDAXI.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_GDAXI.shape[1], train_X_GDAXI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_GDAXI = create_model(hidden_nodes)
model_GDAXI.fit(train_X_GDAXI, train_y_GDAXI, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_GDAXI.predict(test_X)
predictions_GDAXI = scaler.inverse_transform(predictions)
test_y_GDAXI = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_GDAXI = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_GDAXI = mean_squared_error(test_y_GDAXI, predictions_GDAXI)
print(MSE_LSTM_1_GDAXI)

# QLIKE

y_forecastvalues = predictions_GDAXI
y_actualvalues = test_y_GDAXI
qlike_GDAXI_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_LSTM_1_Out.append(iteration)
qlike_GDAXI_LSTM_1_Out = np.array(qlike_GDAXI_LSTM_1_Out)
QLIKE_LSTM_1_GDAXI = sum(qlike_GDAXI_LSTM_1_Out)/len(y_actualvalues)

predictions = model_GDAXI.predict(train_X_GDAXI)
predictions_GDAXI_In = scaler.inverse_transform(predictions)
train_y_GDAXI_In = np.array(y_train_GDAXI)
MSE_LSTM_1_GDAXI_In = mean_squared_error(train_y_GDAXI_In, predictions_GDAXI_In)
mseGDAXI_LSTM_In = []
for i in np.arange(len(train_y_GDAXI_In)):
    mse = (predictions_GDAXI_In[i]-train_y_GDAXI_In[i])**2
    mseGDAXI_LSTM_In.append(mse)
mseGDAXI_LSTM_In = np.array(mseGDAXI_LSTM_In)

#%% SPX

data = SPX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7
threshold_train = threshold_train_SPX
threshold_test = threshold_test_SPX
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_SPX, train_y_SPX = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_SPX = np.reshape(train_X_SPX, (train_X_SPX.shape[0], train_X_SPX.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_SPX.shape[1], train_X_SPX.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_SPX = create_model(hidden_nodes)
model_SPX.fit(train_X_SPX, train_y_SPX, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_SPX.predict(test_X)
predictions_SPX = scaler.inverse_transform(predictions)
test_y_SPX = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_SPX = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_SPX = mean_squared_error(test_y_SPX, predictions_SPX)
print(MSE_LSTM_1_SPX)

# QLIKE

y_forecastvalues = predictions_SPX
y_actualvalues = test_y_SPX
qlike_SPX_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_LSTM_1_Out.append(iteration)
qlike_SPX_LSTM_1_Out = np.array(qlike_SPX_LSTM_1_Out)
QLIKE_LSTM_1_SPX = sum(qlike_SPX_LSTM_1_Out)/len(y_actualvalues)

predictions = model_SPX.predict(train_X_SPX)
predictions_SPX_In = scaler.inverse_transform(predictions)
train_y_SPX_In = np.array(y_train_SPX)
MSE_LSTM_1_SPX_In = mean_squared_error(train_y_SPX_In, predictions_SPX_In)
mseSPX_LSTM_In = []
for i in np.arange(len(train_y_SPX_In)):
    mse = (predictions_SPX_In[i]-train_y_SPX_In[i])**2
    mseSPX_LSTM_In.append(mse)
mseSPX_LSTM_In = np.array(mseSPX_LSTM_In)

#%% HSI

data = HSI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

threshold_train = threshold_train_HSI
threshold_test = threshold_test_HSI
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_HSI, train_y_HSI = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_HSI = np.reshape(train_X_HSI, (train_X_HSI.shape[0], train_X_HSI.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_HSI.shape[1], train_X_HSI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_HSI = create_model(hidden_nodes)
model_HSI.fit(train_X_HSI, train_y_HSI, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_HSI.predict(test_X)
predictions_HSI = scaler.inverse_transform(predictions)
test_y_HSI = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_HSI = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_HSI = mean_squared_error(test_y_HSI, predictions_HSI)
print(MSE_LSTM_1_HSI)

# QLIKE

y_forecastvalues = predictions_HSI
y_actualvalues = test_y_HSI
qlike_HSI_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_LSTM_1_Out.append(iteration)
qlike_HSI_LSTM_1_Out = np.array(qlike_HSI_LSTM_1_Out)
QLIKE_LSTM_1_HSI = sum(qlike_HSI_LSTM_1_Out)/len(y_actualvalues)

predictions = model_HSI.predict(train_X_HSI)
predictions_HSI_In = scaler.inverse_transform(predictions)
train_y_HSI_In = np.array(y_train_HSI)
MSE_LSTM_1_HSI_In = mean_squared_error(train_y_HSI_In, predictions_HSI_In)
mseHSI_LSTM_In = []
for i in np.arange(len(train_y_HSI_In)):
    mse = (predictions_HSI_In[i]-train_y_HSI_In[i])**2
    mseHSI_LSTM_In.append(mse)
mseHSI_LSTM_In = np.array(mseHSI_LSTM_In)

#%% IBEX

data = IBEX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

threshold_train = threshold_train_IBEX
threshold_test = threshold_test_IBEX
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_IBEX, train_y_IBEX = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_IBEX = np.reshape(train_X_IBEX, (train_X_IBEX.shape[0], train_X_IBEX.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_IBEX.shape[1], train_X_IBEX.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IBEX = create_model(hidden_nodes)
model_IBEX.fit(train_X_IBEX, train_y_IBEX, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_IBEX.predict(test_X)
predictions_IBEX = scaler.inverse_transform(predictions)
test_y_IBEX = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IBEX = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_IBEX = mean_squared_error(test_y_IBEX, predictions_IBEX)
print(MSE_LSTM_1_IBEX)

# QLIKE

y_forecastvalues = predictions_IBEX
y_actualvalues = test_y_IBEX
qlike_IBEX_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_LSTM_1_Out.append(iteration)
qlike_IBEX_LSTM_1_Out = np.array(qlike_IBEX_LSTM_1_Out)
QLIKE_LSTM_1_IBEX = sum(qlike_IBEX_LSTM_1_Out)/len(y_actualvalues)

predictions = model_IBEX.predict(train_X_IBEX)
predictions_IBEX_In = scaler.inverse_transform(predictions)
train_y_IBEX_In = np.array(y_train_IBEX)
MSE_LSTM_1_IBEX_In = mean_squared_error(train_y_IBEX_In, predictions_IBEX_In)
mseIBEX_LSTM_In = []
for i in np.arange(len(train_y_IBEX_In)):
    mse = (predictions_IBEX_In[i]-train_y_IBEX_In[i])**2
    mseIBEX_LSTM_In.append(mse)
mseIBEX_LSTM_In = np.array(mseIBEX_LSTM_In)

#%% IXIC

data = IXIC
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

threshold_train = threshold_train_IXIC
threshold_test = threshold_test_IXIC
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_IXIC, train_y_IXIC = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_IXIC = np.reshape(train_X_IXIC, (train_X_IXIC.shape[0], train_X_IXIC.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0007), recurrent_regularizer=regularizers.l2(0.0007),  input_shape=(train_X_IXIC.shape[1], train_X_IXIC.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IXIC = create_model(hidden_nodes)
model_IXIC.fit(train_X_IXIC, train_y_IXIC, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_IXIC.predict(test_X)
predictions_IXIC = scaler.inverse_transform(predictions)
test_y_IXIC = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IXIC = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_IXIC = mean_squared_error(test_y_IXIC, predictions_IXIC)
print(MSE_LSTM_1_IXIC)

# QLIKE

y_forecastvalues = predictions_IXIC
y_actualvalues = test_y_IXIC
qlike_IXIC_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_LSTM_1_Out.append(iteration)
qlike_IXIC_LSTM_1_Out = np.array(qlike_IXIC_LSTM_1_Out)
QLIKE_LSTM_1_IXIC = sum(qlike_IXIC_LSTM_1_Out)/len(y_actualvalues)

predictions = model_IXIC.predict(train_X_IXIC)
predictions_IXIC_In = scaler.inverse_transform(predictions)
train_y_IXIC_In = np.array(y_train_IXIC)
MSE_LSTM_1_IXIC_In = mean_squared_error(train_y_IXIC_In, predictions_IXIC_In)
mseIXIC_LSTM_In = []
for i in np.arange(len(train_y_IXIC_In)):
    mse = (predictions_IXIC_In[i]-train_y_IXIC_In[i])**2
    mseIXIC_LSTM_In.append(mse)
mseIXIC_LSTM_In = np.array(mseIXIC_LSTM_In)

#%% N225

data = N225
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

threshold_train = threshold_train_N225
threshold_test = threshold_test_N225
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_N225, train_y_N225 = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_N225 = np.reshape(train_X_N225, (train_X_N225.shape[0], train_X_N225.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_N225.shape[1], train_X_N225.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_N225 = create_model(hidden_nodes)
model_N225.fit(train_X_N225, train_y_N225, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_N225.predict(test_X)
predictions_N225 = scaler.inverse_transform(predictions)
test_y_N225 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_N225 = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_N225 = mean_squared_error(test_y_N225, predictions_N225)
print(MSE_LSTM_1_N225)

# QLIKE

y_forecastvalues = predictions_N225
y_actualvalues = test_y_N225
qlike_N225_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_LSTM_1_Out.append(iteration)
qlike_N225_LSTM_1_Out = np.array(qlike_N225_LSTM_1_Out)
QLIKE_LSTM_1_N225 = sum(qlike_N225_LSTM_1_Out)/len(y_actualvalues)

predictions = model_N225.predict(train_X_N225)
predictions_N225_In = scaler.inverse_transform(predictions)
train_y_N225_In = np.array(y_train_N225)
MSE_LSTM_1_N225_In = mean_squared_error(train_y_N225_In, predictions_N225_In)
mseN225_LSTM_In = []
for i in np.arange(len(train_y_N225_In)):
    mse = (predictions_N225_In[i]-train_y_N225_In[i])**2
    mseN225_LSTM_In.append(mse)
mseN225_LSTM_In = np.array(mseN225_LSTM_In)

#%% 0MXC20

data = OMXC20
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 1

threshold_train = threshold_train_OMXC20
threshold_test = threshold_test_OMXC20
    
train_data = data[:threshold_train]
test_data = data[threshold_test:]
train_data_len = len(train_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data1.reshape(-1,1))

train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

train_X_OMXC20, train_y_OMXC20 = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)
train_X_OMXC20 = np.reshape(train_X_OMXC20, (train_X_OMXC20.shape[0], train_X_OMXC20.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_OMXC20.shape[1], train_X_OMXC20.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_OMXC20 = create_model(hidden_nodes)
model_OMXC20.fit(train_X_OMXC20, train_y_OMXC20, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_OMXC20.predict(test_X)
predictions_OMXC20 = scaler.inverse_transform(predictions)
test_y_OMXC20 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_OMXC20 = data.tail(len(test_X)).index

# MSE

MSE_LSTM_1_OMXC20 = mean_squared_error(test_y_OMXC20, predictions_OMXC20)
print(MSE_LSTM_1_OMXC20)

# QLIKE

y_forecastvalues = predictions_OMXC20
y_actualvalues = test_y_OMXC20
qlike_OMXC20_LSTM_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_LSTM_1_Out.append(iteration)
qlike_OMXC20_LSTM_1_Out = np.array(qlike_OMXC20_LSTM_1_Out)
QLIKE_LSTM_1_OMXC20 = sum(qlike_OMXC20_LSTM_1_Out)/len(y_actualvalues)

predictions = model_OMXC20.predict(train_X_OMXC20)
predictions_OMXC20_In = scaler.inverse_transform(predictions)
train_y_OMXC20_In = np.array(y_train_OMXC20)
# posizione = np.where(predictions_OMXC20_In<=0)[0]
MSE_LSTM_1_OMXC20_In = mean_squared_error(train_y_OMXC20_In, predictions_OMXC20_In)
mseOMXC20_LSTM_In = []
for i in np.arange(len(train_y_OMXC20_In)):
    mse = (predictions_OMXC20_In[i]-train_y_OMXC20_In[i])**2
    mseOMXC20_LSTM_In.append(mse)
mseOMXC20_LSTM_In = np.array(mseOMXC20_LSTM_In)

#%% In-sample Loss Function

predictions = model_DJI.predict(train_X_DJI)
predictions_DJI_In = scaler.inverse_transform(predictions)
train_y_DJI_In = np.array(y_train_DJI)
MSE_LSTM_1_DJI_In = mean_squared_error(train_y_DJI_In, predictions_DJI_In)
mseDJI_LSTM_In = []
for i in np.arange(len(train_y_DJI_In)):
    mse = (predictions_DJI_In[i]-train_y_DJI_In[i])**2
    mseDJI_LSTM_In.append(mse)
mseDJI_LSTM_In = np.array(mseDJI_LSTM_In)

predictions = model_FTSE.predict(train_X_FTSE)
predictions_FTSE_In = scaler.inverse_transform(predictions)
train_y_FTSE_In = np.array(y_train_FTSE)
MSE_LSTM_1_FTSE_In = mean_squared_error(train_y_FTSE_In, predictions_FTSE_In)
mseFTSE_LSTM_In = []
for i in np.arange(len(train_y_FTSE_In)):
    mse = (predictions_FTSE_In[i]-train_y_FTSE_In[i])**2
    mseFTSE_LSTM_In.append(mse)
mseFTSE_LSTM_In = np.array(mseFTSE_LSTM_In)

predictions = model_FTSEMIB.predict(train_X_FTSEMIB)
predictions_FTSEMIB_In = scaler.inverse_transform(predictions)
train_y_FTSEMIB_In = np.array(y_train_FTSEMIB)
MSE_LSTM_1_FTSEMIB_In = mean_squared_error(train_y_FTSEMIB_In, predictions_FTSEMIB_In)
mseFTSEMIB_LSTM_In = []
for i in np.arange(len(train_y_FTSEMIB_In)):
    mse = (predictions_FTSEMIB_In[i]-train_y_FTSEMIB_In[i])**2
    mseFTSEMIB_LSTM_In.append(mse)
mseFTSEMIB_LSTM_In = np.array(mseFTSEMIB_LSTM_In)

predictions = model_GDAXI.predict(train_X_GDAXI)
predictions_GDAXI_In = scaler.inverse_transform(predictions)
train_y_GDAXI_In = np.array(y_train_GDAXI)
MSE_LSTM_1_GDAXI_In = mean_squared_error(train_y_GDAXI_In, predictions_GDAXI_In)
mseGDAXI_LSTM_In = []
for i in np.arange(len(train_y_GDAXI_In)):
    mse = (predictions_GDAXI_In[i]-train_y_GDAXI_In[i])**2
    mseGDAXI_LSTM_In.append(mse)
mseGDAXI_LSTM_In = np.array(mseGDAXI_LSTM_In)

predictions = model_SPX.predict(train_X_SPX)
predictions_SPX_In = scaler.inverse_transform(predictions)
train_y_SPX_In = np.array(y_train_SPX)
MSE_LSTM_1_SPX_In = mean_squared_error(train_y_SPX_In, predictions_SPX_In)
mseSPX_LSTM_In = []
for i in np.arange(len(train_y_SPX_In)):
    mse = (predictions_SPX_In[i]-train_y_SPX_In[i])**2
    mseSPX_LSTM_In.append(mse)
mseSPX_LSTM_In = np.array(mseSPX_LSTM_In)

predictions = model_HSI.predict(train_X_HSI)
predictions_HSI_In = scaler.inverse_transform(predictions)
train_y_HSI_In = np.array(y_train_HSI)
MSE_LSTM_1_HSI_In = mean_squared_error(train_y_HSI_In, predictions_HSI_In)
mseHSI_LSTM_In = []
for i in np.arange(len(train_y_HSI_In)):
    mse = (predictions_HSI_In[i]-train_y_HSI_In[i])**2
    mseHSI_LSTM_In.append(mse)
mseHSI_LSTM_In = np.array(mseHSI_LSTM_In)

predictions = model_IBEX.predict(train_X_IBEX)
predictions_IBEX_In = scaler.inverse_transform(predictions)
train_y_IBEX_In = np.array(y_train_IBEX)
MSE_LSTM_1_IBEX_In = mean_squared_error(train_y_IBEX_In, predictions_IBEX_In)
mseIBEX_LSTM_In = []
for i in np.arange(len(train_y_IBEX_In)):
    mse = (predictions_IBEX_In[i]-train_y_IBEX_In[i])**2
    mseIBEX_LSTM_In.append(mse)
mseIBEX_LSTM_In = np.array(mseIBEX_LSTM_In)

predictions = model_IXIC.predict(train_X_IXIC)
predictions_IXIC_In = scaler.inverse_transform(predictions)
train_y_IXIC_In = np.array(y_train_IXIC)
MSE_LSTM_1_IXIC_In = mean_squared_error(train_y_IXIC_In, predictions_IXIC_In)
mseIXIC_LSTM_In = []
for i in np.arange(len(train_y_IXIC_In)):
    mse = (predictions_IXIC_In[i]-train_y_IXIC_In[i])**2
    mseIXIC_LSTM_In.append(mse)
mseIXIC_LSTM_In = np.array(mseIXIC_LSTM_In)

predictions = model_N225.predict(train_X_N225)
predictions_N225_In = scaler.inverse_transform(predictions)
train_y_N225_In = np.array(y_train_N225)
MSE_LSTM_1_N225_In = mean_squared_error(train_y_N225_In, predictions_N225_In)
mseN225_LSTM_In = []
for i in np.arange(len(train_y_N225_In)):
    mse = (predictions_N225_In[i]-train_y_N225_In[i])**2
    mseN225_LSTM_In.append(mse)
mseN225_LSTM_In = np.array(mseN225_LSTM_In)

predictions = model_OMXC20.predict(train_X_OMXC20)
predictions_OMXC20_In = scaler.inverse_transform(predictions)
train_y_OMXC20_In = np.array(y_train_OMXC20)
# posizione = np.where(predictions_OMXC20_In<=0)[0]
MSE_LSTM_1_OMXC20_In = mean_squared_error(train_y_OMXC20_In, predictions_OMXC20_In)
mseOMXC20_LSTM_In = []
for i in np.arange(len(train_y_OMXC20_In)):
    mse = (predictions_OMXC20_In[i]-train_y_OMXC20_In[i])**2
    mseOMXC20_LSTM_In.append(mse)
mseOMXC20_LSTM_In = np.array(mseOMXC20_LSTM_In)

#%% Out of sample Loss function

mseDJI_LSTM_Out = []
for i in np.arange(len(test_y_DJI)):
    mse = (predictions_DJI[i]-test_y_DJI[i])**2
    mseDJI_LSTM_Out.append(mse)
mseDJI_LSTM_Out = np.array(mseDJI_LSTM_Out)

mseFTSE_LSTM_Out = []
for i in np.arange(len(test_y_FTSE)):
    mse = (predictions_FTSE[i]-test_y_FTSE[i])**2
    mseFTSE_LSTM_Out.append(mse)
mseFTSE_LSTM_Out = np.array(mseFTSE_LSTM_Out)

mseFTSEMIB_LSTM_Out = []
for i in np.arange(len(test_y_FTSEMIB)):
    mse = (predictions_FTSEMIB[i]-test_y_FTSEMIB[i])**2
    mseFTSEMIB_LSTM_Out.append(mse)
mseFTSEMIB_LSTM_Out = np.array(mseFTSEMIB_LSTM_Out)

mseGDAXI_LSTM_Out = []
for i in np.arange(len(test_y_GDAXI)):
    mse = (predictions_GDAXI[i]-test_y_GDAXI[i])**2
    mseGDAXI_LSTM_Out.append(mse)
mseGDAXI_LSTM_Out = np.array(mseGDAXI_LSTM_Out)

mseSPX_LSTM_Out = []
for i in np.arange(len(test_y_SPX)):
    mse = (predictions_SPX[i]-test_y_SPX[i])**2
    mseSPX_LSTM_Out.append(mse)
mseSPX_LSTM_Out = np.array(mseSPX_LSTM_Out)

mseHSI_LSTM_Out = []
for i in np.arange(len(test_y_HSI)):
    mse = (predictions_HSI[i]-test_y_HSI[i])**2
    mseHSI_LSTM_Out.append(mse)
mseHSI_LSTM_Out = np.array(mseHSI_LSTM_Out)

mseIBEX_LSTM_Out = []
for i in np.arange(len(test_y_IBEX)):
    mse = (predictions_IBEX[i]-test_y_IBEX[i])**2
    mseIBEX_LSTM_Out.append(mse)
mseIBEX_LSTM_Out = np.array(mseIBEX_LSTM_Out)

mseIXIC_LSTM_Out = []
for i in np.arange(len(test_y_IXIC)):
    mse = (predictions_IXIC[i]-test_y_IXIC[i])**2
    mseIXIC_LSTM_Out.append(mse)
mseIXIC_LSTM_Out = np.array(mseIXIC_LSTM_Out)

mseN225_LSTM_Out = []
for i in np.arange(len(test_y_N225)):
    mse = (predictions_N225[i]-test_y_N225[i])**2
    mseN225_LSTM_Out.append(mse)
mseN225_LSTM_Out = np.array(mseN225_LSTM_Out)

mseOMXC20_LSTM_Out = []
for i in np.arange(len(test_y_OMXC20)):
    mse = (predictions_OMXC20[i]-test_y_OMXC20[i])**2
    mseOMXC20_LSTM_Out.append(mse)
mseOMXC20_LSTM_Out = np.array(mseOMXC20_LSTM_Out)

#%% In Sample MSE
'''
arrays_In = {
    'mse_LSTM': {
        'DJI': mseDJI_LSTM_In,
        'FTSE': mseFTSE_LSTM_In,
        'FTSEMIB': mseFTSEMIB_LSTM_In,
        'GDAXI': mseGDAXI_LSTM_In,
        'SPX': mseSPX_LSTM_In,
        'HSI': mseHSI_LSTM_In,
        'IBEX': mseIBEX_LSTM_In,
        'IXIC': mseIXIC_LSTM_In,
        'N225': mseN225_LSTM_In,
        'OMXC20': mseOMXC20_LSTM_In
                }
        }

for k1 in arrays_In:
    if k1 == 'mse_LSTM':    
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_LSTM_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
 '''           
#%% Out of Sample MSE
'''
arrays_Out = {
    'mse_LSTM': {
        'DJI': mseDJI_LSTM_Out,
        'FTSE': mseFTSE_LSTM_Out,
        'FTSEMIB': mseFTSEMIB_LSTM_Out,
        'GDAXI': mseGDAXI_LSTM_Out,
        'SPX': mseSPX_LSTM_Out,
        'HSI': mseHSI_LSTM_Out,
        'IBEX': mseIBEX_LSTM_Out,
        'IXIC': mseIXIC_LSTM_Out,
        'N225': mseN225_LSTM_Out,
        'OMXC20': mseOMXC20_LSTM_Out
                }
        }

for k1 in arrays_Out:
    if k1 == 'mse_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_LSTM_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
'''
#%% Qlike Out
'''
arrays_Out = {
    'qlike_LSTM': {
        'DJI': qlike_DJI_LSTM_1_Out,
        'FTSE': qlike_FTSE_LSTM_1_Out,
        'FTSEMIB': qlike_FTSEMIB_LSTM_1_Out,
        'GDAXI': qlike_GDAXI_LSTM_1_Out,
        'SPX': qlike_SPX_LSTM_1_Out,
        'HSI': qlike_HSI_LSTM_1_Out,
        'IBEX': qlike_IBEX_LSTM_1_Out,
        'IXIC': qlike_IXIC_LSTM_1_Out,
        'N225': qlike_N225_LSTM_1_Out,
        'OMXC20': qlike_OMXC20_LSTM_1_Out
                }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_LSTM_1_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
'''
#%% Qlike In

y_forecastvalues = predictions_DJI_In
y_actualvalues = train_y_DJI_In
qlike_DJI_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_LSTM_1_In.append(iteration)
qlike_DJI_LSTM_1_In = np.array(qlike_DJI_LSTM_1_In)
QLIKE_LSTM_1_DJI_In = sum(qlike_DJI_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_FTSE_In
y_actualvalues = train_y_FTSE_In
qlike_FTSE_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_LSTM_1_In.append(iteration)
qlike_FTSE_LSTM_1_In = np.array(qlike_FTSE_LSTM_1_In)
QLIKE_LSTM_1_FTSE_In = sum(qlike_FTSE_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_FTSEMIB_In
y_actualvalues = train_y_FTSEMIB_In
qlike_FTSEMIB_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_LSTM_1_In.append(iteration)
qlike_FTSEMIB_LSTM_1_In = np.array(qlike_FTSEMIB_LSTM_1_In)
QLIKE_LSTM_1_FTSEMIB_In = sum(qlike_FTSEMIB_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_GDAXI_In
y_actualvalues = train_y_GDAXI_In
qlike_GDAXI_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_LSTM_1_In.append(iteration)
qlike_GDAXI_LSTM_1_In = np.array(qlike_GDAXI_LSTM_1_In)
QLIKE_LSTM_1_GDAXI_In = sum(qlike_GDAXI_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_SPX_In
y_actualvalues = train_y_SPX_In
qlike_SPX_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_LSTM_1_In.append(iteration)
qlike_SPX_LSTM_1_In = np.array(qlike_SPX_LSTM_1_In)
QLIKE_LSTM_1_SPX_In = sum(qlike_SPX_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_HSI_In
y_actualvalues = train_y_HSI_In
qlike_HSI_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_LSTM_1_In.append(iteration)
qlike_HSI_LSTM_1_In = np.array(qlike_HSI_LSTM_1_In)
QLIKE_LSTM_1_HSI_In = sum(qlike_HSI_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_IBEX_In
y_actualvalues = train_y_IBEX_In
qlike_IBEX_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_LSTM_1_In.append(iteration)
qlike_IBEX_LSTM_1_In = np.array(qlike_IBEX_LSTM_1_In)
QLIKE_LSTM_1_IBEX_In = sum(qlike_IBEX_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_IXIC_In
y_actualvalues = train_y_IXIC_In
qlike_IXIC_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_LSTM_1_In.append(iteration)
qlike_IXIC_LSTM_1_In = np.array(qlike_IXIC_LSTM_1_In)
QLIKE_LSTM_1_IXIC_In = sum(qlike_IXIC_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_N225_In
y_actualvalues = train_y_N225_In
qlike_N225_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_LSTM_1_In.append(iteration)
qlike_N225_LSTM_1_In = np.array(qlike_N225_LSTM_1_In)
QLIKE_LSTM_1_N225_In = sum(qlike_N225_LSTM_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_OMXC20_In
y_actualvalues = train_y_OMXC20_In
qlike_OMXC20_LSTM_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_LSTM_1_In.append(iteration)
qlike_OMXC20_LSTM_1_In = np.array(qlike_OMXC20_LSTM_1_In)
QLIKE_LSTM_1_OMXC20_In = sum(qlike_OMXC20_LSTM_1_In)/len(y_actualvalues)

#%% Qlike In
'''
arrays_In = {
    'qlike_LSTM': {
        'DJI': qlike_DJI_LSTM_1_In,
        'FTSE': qlike_FTSE_LSTM_1_In,
        'FTSEMIB': qlike_FTSEMIB_LSTM_1_In,
        'GDAXI': qlike_GDAXI_LSTM_1_In,
        'SPX': qlike_SPX_LSTM_1_In,
        'HSI': qlike_HSI_LSTM_1_In,
        'IBEX': qlike_IBEX_LSTM_1_In,
        'IXIC': qlike_IXIC_LSTM_1_In,
        'N225': qlike_N225_LSTM_1_In,
        'OMXC20': qlike_OMXC20_LSTM_1_In
                }
        }

for k1 in arrays_In:
    if k1 == 'qlike_LSTM':    
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_LSTM_1_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
'''

#%% 

symbols = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']
predictions = [predictions_DJI, predictions_FTSE, predictions_FTSEMIB, predictions_GDAXI, predictions_SPX, predictions_HSI, predictions_IBEX, predictions_IXIC, predictions_N225, predictions_OMXC20]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastlstm_{symbol}.csv', prediction, delimiter=',')


symbols = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']
predictions = [predictions_DJI_In, predictions_FTSE_In, predictions_FTSEMIB_In, predictions_GDAXI_In, predictions_SPX_In, predictions_HSI_In, predictions_IBEX_In, predictions_IXIC_In, predictions_N225_In, predictions_OMXC20_In]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastlstm_{symbol}_in.csv', prediction, delimiter=',')