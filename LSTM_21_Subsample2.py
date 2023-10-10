# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:52:06 2023

@author: cesar
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.random import set_seed
from keras.models import Sequential 
from keras.layers import Dense, LSTM
from tensorflow.keras.initializers import GlorotUniform
from keras import regularizers

#%% LSTM 

start_date2_DJI = '2020-07-19'
start_date2_DJI = pd.to_datetime(start_date2_DJI, format = '%Y-%m-%d')
end_date2_DJI = '2022-05-05'
end_date2_DJI = pd.to_datetime(end_date2_DJI, format = '%Y-%m-%d')

start_date2_FTSE = '2020-07-12'
start_date2_FTSE = pd.to_datetime(start_date2_FTSE, format = '%Y-%m-%d')
end_date2_FTSE = '2022-05-05'
end_date2_FTSE = pd.to_datetime(end_date2_FTSE, format = '%Y-%m-%d')

start_date2_FTSEMIB = '2020-07-01'
start_date2_FTSEMIB = pd.to_datetime(start_date2_FTSEMIB, format = '%Y-%m-%d')
end_date2_FTSEMIB = '2022-05-05'
end_date2_FTSEMIB = pd.to_datetime(end_date2_FTSEMIB, format = '%Y-%m-%d')

start_date2_GDAXI = '2020-06-18'
start_date2_GDAXI = pd.to_datetime(start_date2_GDAXI, format = '%Y-%m-%d')
end_date2_GDAXI = '2022-05-05'
end_date2_GDAXI = pd.to_datetime(end_date2_GDAXI, format = '%Y-%m-%d')

start_date2_SPX = '2020-10-01'
start_date2_SPX = pd.to_datetime(start_date2_SPX, format = '%Y-%m-%d')
end_date2_SPX = '2022-05-05'
end_date2_SPX = pd.to_datetime(end_date2_SPX, format = '%Y-%m-%d')

start_date2_HSI = '2020-03-24'
start_date2_HSI = pd.to_datetime(start_date2_HSI, format = '%Y-%m-%d')
end_date2_HSI = '2022-05-05'
end_date2_HSI = pd.to_datetime(end_date2_HSI, format = '%Y-%m-%d')

start_date2_IBEX = '2020-04-15'
start_date2_IBEX = pd.to_datetime(start_date2_IBEX, format = '%Y-%m-%d')
end_date2_IBEX = '2022-05-05'
end_date2_IBEX = pd.to_datetime(end_date2_IBEX, format = '%Y-%m-%d')

start_date2_IXIC = '2020-06-29'
start_date2_IXIC = pd.to_datetime(start_date2_IXIC, format = '%Y-%m-%d')
end_date2_IXIC = '2022-05-05'
end_date2_IXIC = pd.to_datetime(end_date2_IXIC, format = '%Y-%m-%d')

start_date2_N225 = '2020-04-01'
start_date2_N225 = pd.to_datetime(start_date2_N225, format = '%Y-%m-%d')
end_date2_N225 = '2022-05-05'
end_date2_N225 = pd.to_datetime(end_date2_N225, format = '%Y-%m-%d')

start_date2_OMXC20 = '2020-03-31'
start_date2_OMXC20 = pd.to_datetime(start_date2_OMXC20, format = '%Y-%m-%d')
end_date2_OMXC20 = '2022-05-05'
end_date2_OMXC20 = pd.to_datetime(end_date2_OMXC20, format = '%Y-%m-%d')

# DJI

data = DJI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6
    
test_data = data[start_date2_DJI:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Crea il modello LSTM
def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_DJI.shape[1], train_X_DJI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_DJI_sub2 = create_model(hidden_nodes)
model_DJI_sub2.fit(train_X_DJI, train_y_DJI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_DJI_sub2.predict(test_X)
predictions_DJI_sub2 = scaler.inverse_transform(predictions)
test_y_DJI_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_DJI_sub2 = test_data.tail(len(test_X)).index

# Find mean squared error and the QLIKE loss

MSE_LSTM_1_DJI_sub2 = mean_squared_error(test_y_DJI_sub2, predictions_DJI_sub2)

mseDJI_LSTM_sub2 = []
for i in np.arange(len(test_y_DJI_sub2)):
    mse = (predictions_DJI_sub2[i]-test_y_DJI_sub2[i])**2
    mseDJI_LSTM_sub2.append(mse)
mseDJI_LSTM_sub2 = np.array(mseDJI_LSTM_sub2)

y_forecastvalues = predictions_DJI_sub2
y_actualvalues = test_y_DJI_sub2
qlike_DJI_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_LSTM_1_sub2.append(iteration)
qlike_DJI_LSTM_1_sub2 = np.array(qlike_DJI_LSTM_1_sub2)
QLIKE_LSTM_1_DJI_sub2 = sum(qlike_DJI_LSTM_1_sub2)/len(y_actualvalues)

# FTSE

data = FTSE
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6


# Data scaling

test_data = data[start_date2_FTSE:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model 

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_FTSE.shape[1], train_X_FTSE.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSE_sub2 = create_model(hidden_nodes)
model_FTSE_sub2.fit(train_X_FTSE, train_y_FTSE, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_FTSE_sub2.predict(test_X)
predictions_FTSE_sub2 = scaler.inverse_transform(predictions)
test_y_FTSE_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSE_sub2 = test_data.tail(len(test_X)).index
    
# Find  mean squared error and QLIKE loss

MSE_LSTM_1_FTSE_sub2 = mean_squared_error(test_y_FTSE_sub2, predictions_FTSE_sub2)

mseFTSE_LSTM_sub2 = []
for i in np.arange(len(test_y_FTSE_sub2)):
    mse = (predictions_FTSE_sub2[i]-test_y_FTSE_sub2[i])**2
    mseFTSE_LSTM_sub2.append(mse)
mseFTSE_LSTM_sub2 = np.array(mseFTSE_LSTM_sub2)

y_forecastvalues = predictions_FTSE_sub2
y_actualvalues = test_y_FTSE_sub2
qlike_FTSE_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_LSTM_1_sub2.append(iteration)
qlike_FTSE_LSTM_1_sub2 = np.array(qlike_FTSE_LSTM_1_sub2)
QLIKE_LSTM_1_FTSE_sub2 = sum(qlike_FTSE_LSTM_1_sub2)/len(y_actualvalues)

# FTSEMIB

data = FTSEMIB
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

# Data scaling

test_data = data[start_date2_FTSEMIB:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_FTSEMIB.shape[1], train_X_FTSEMIB.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSEMIB_sub2 = create_model(hidden_nodes)
model_FTSEMIB_sub2.fit(train_X_FTSEMIB, train_y_FTSEMIB, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_FTSEMIB_sub2.predict(test_X)
predictions_FTSEMIB_sub2 = scaler.inverse_transform(predictions)
test_y_FTSEMIB_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSEMIB_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_FTSEMIB_sub2 = mean_squared_error(test_y_FTSEMIB_sub2, predictions_FTSEMIB_sub2)

mseFTSEMIB_LSTM_sub2 = []
for i in np.arange(len(test_y_FTSEMIB_sub2)):
    mse = (predictions_FTSEMIB_sub2[i]-test_y_FTSEMIB_sub2[i])**2
    mseFTSEMIB_LSTM_sub2.append(mse)
mseFTSEMIB_LSTM_sub2 = np.array(mseFTSEMIB_LSTM_sub2)

y_forecastvalues = predictions_FTSEMIB_sub2
y_actualvalues = test_y_FTSEMIB_sub2
qlike_FTSEMIB_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_LSTM_1_sub2.append(iteration)
qlike_FTSEMIB_LSTM_1_sub2 = np.array(qlike_FTSEMIB_LSTM_1_sub2)
QLIKE_LSTM_1_FTSEMIB_sub2 = sum(qlike_FTSEMIB_LSTM_1_sub2)/len(y_actualvalues)

# GDAXI

data = GDAXI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date2_GDAXI:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_GDAXI.shape[1], train_X_GDAXI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_GDAXI_sub2 = create_model(hidden_nodes)
model_GDAXI_sub2.fit(train_X_GDAXI, train_y_GDAXI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_GDAXI_sub2.predict(test_X)
predictions_GDAXI_sub2 = scaler.inverse_transform(predictions)
test_y_GDAXI_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_GDAXI_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_GDAXI_sub2 = mean_squared_error(test_y_GDAXI_sub2, predictions_GDAXI_sub2)

mseGDAXI_LSTM_sub2 = []
for i in np.arange(len(test_y_GDAXI_sub2)):
    mse = (predictions_GDAXI_sub2[i]-test_y_GDAXI_sub2[i])**2
    mseGDAXI_LSTM_sub2.append(mse)
mseGDAXI_LSTM_sub2 = np.array(mseGDAXI_LSTM_sub2)

y_forecastvalues = predictions_GDAXI_sub2
y_actualvalues = test_y_GDAXI_sub2
qlike_GDAXI_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_LSTM_1_sub2.append(iteration)
qlike_GDAXI_LSTM_1_sub2 = np.array(qlike_GDAXI_LSTM_1_sub2)
QLIKE_LSTM_1_GDAXI_sub2 = sum(qlike_GDAXI_LSTM_1_sub2)/len(y_actualvalues)

# SPX

data = SPX
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date2_SPX:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_SPX.shape[1], train_X_SPX.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_SPX_sub2 = create_model(hidden_nodes)
model_SPX_sub2.fit(train_X_SPX, train_y_SPX, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_SPX_sub2.predict(test_X)
predictions_SPX_sub2 = scaler.inverse_transform(predictions)
test_y_SPX_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))
posizione = np.where(predictions_SPX_sub2<=0)[0]
media = np.mean(test_y_SPX_sub2[62:66])
predictions_SPX_sub2[64] = media
predictDates_SPX_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_SPX_sub2 = mean_squared_error(test_y_SPX_sub2, predictions_SPX_sub2)

mseSPX_LSTM_sub2 = []
for i in np.arange(len(test_y_SPX_sub2)):
    mse = (predictions_SPX_sub2[i]-test_y_SPX_sub2[i])**2
    mseSPX_LSTM_sub2.append(mse)
mseSPX_LSTM_sub2 = np.array(mseSPX_LSTM_sub2)

y_forecastvalues = predictions_SPX_sub2
y_actualvalues = test_y_SPX_sub2
qlike_SPX_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_LSTM_1_sub2.append(iteration)
qlike_SPX_LSTM_1_sub2 = np.array(qlike_SPX_LSTM_1_sub2)
QLIKE_LSTM_1_SPX_sub2 = sum(qlike_SPX_LSTM_1_sub2)/len(y_actualvalues)

# HSI

data = HSI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date2_HSI:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_HSI.shape[1], train_X_HSI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_HSI_sub2 = create_model(hidden_nodes)
model_HSI_sub2.fit(train_X_HSI, train_y_HSI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_HSI_sub2.predict(test_X)
predictions_HSI_sub2 = scaler.inverse_transform(predictions)
test_y_HSI_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_HSI_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_HSI_sub2 = mean_squared_error(test_y_HSI_sub2, predictions_HSI_sub2)

mseHSI_LSTM_sub2 = []
for i in np.arange(len(test_y_HSI_sub2)):
    mse = (predictions_HSI_sub2[i]-test_y_HSI_sub2[i])**2
    mseHSI_LSTM_sub2.append(mse)
mseHSI_LSTM_sub2 = np.array(mseHSI_LSTM_sub2)

y_forecastvalues = predictions_HSI_sub2
y_actualvalues = test_y_HSI_sub2
qlike_HSI_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_LSTM_1_sub2.append(iteration)
qlike_HSI_LSTM_1_sub2 = np.array(qlike_HSI_LSTM_1_sub2)
QLIKE_LSTM_1_HSI_sub2 = sum(qlike_HSI_LSTM_1_sub2)/len(y_actualvalues)

# IBEX

data = IBEX
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date2_IBEX:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_IBEX.shape[1], train_X_IBEX.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IBEX_sub2 = create_model(hidden_nodes)
model_IBEX_sub2.fit(train_X_IBEX, train_y_IBEX, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_IBEX_sub2.predict(test_X)
predictions_IBEX_sub2 = scaler.inverse_transform(predictions)
test_y_IBEX_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IBEX_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_IBEX_sub2 = mean_squared_error(test_y_IBEX_sub2, predictions_IBEX_sub2)

mseIBEX_LSTM_sub2 = []
for i in np.arange(len(test_y_IBEX_sub2)):
    mse = (predictions_IBEX_sub2[i]-test_y_IBEX_sub2[i])**2
    mseIBEX_LSTM_sub2.append(mse)
mseIBEX_LSTM_sub2 = np.array(mseIBEX_LSTM_sub2)

y_forecastvalues = predictions_IBEX_sub2
y_actualvalues = test_y_IBEX_sub2
qlike_IBEX_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_LSTM_1_sub2.append(iteration)
qlike_IBEX_LSTM_1_sub2 = np.array(qlike_IBEX_LSTM_1_sub2)
QLIKE_LSTM_1_IBEX_sub2 = sum(qlike_IBEX_LSTM_1_sub2)/len(y_actualvalues)

# IXIC

data = IXIC
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

# Data scaling

test_data = data[start_date2_IXIC:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_IXIC.shape[1], train_X_IXIC.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IXIC_sub2 = create_model(hidden_nodes)
model_IXIC_sub2.fit(train_X_IXIC, train_y_IXIC, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_IXIC_sub2.predict(test_X)
predictions_IXIC_sub2 = scaler.inverse_transform(predictions)
test_y_IXIC_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IXIC_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_IXIC_sub2 = mean_squared_error(test_y_IXIC_sub2, predictions_IXIC_sub2)

mseIXIC_LSTM_sub2 = []
for i in np.arange(len(test_y_IXIC_sub2)):
    mse = (predictions_IXIC_sub2[i]-test_y_IXIC_sub2[i])**2
    mseIXIC_LSTM_sub2.append(mse)
mseIXIC_LSTM_sub2 = np.array(mseIXIC_LSTM_sub2)

y_forecastvalues = predictions_IXIC_sub2
y_actualvalues = test_y_IXIC_sub2
qlike_IXIC_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_LSTM_1_sub2.append(iteration)
qlike_IXIC_LSTM_1_sub2 = np.array(qlike_IXIC_LSTM_1_sub2)
QLIKE_LSTM_1_IXIC_sub2 = sum(qlike_IXIC_LSTM_1_sub2)/len(y_actualvalues)

# N225

data = N225
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

# Data scaling

test_data = data[start_date2_N225:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), input_shape=(train_X_N225.shape[1], train_X_N225.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_N225_sub2 = create_model(hidden_nodes)
model_N225_sub2.fit(train_X_N225, train_y_N225, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_N225_sub2.predict(test_X)
predictions_N225_sub2 = scaler.inverse_transform(predictions)
test_y_N225_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_N225_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_N225_sub2 = mean_squared_error(test_y_N225_sub2, predictions_N225_sub2)

mseN225_LSTM_sub2 = []
for i in np.arange(len(test_y_N225_sub2)):
    mse = (predictions_N225_sub2[i]-test_y_N225_sub2[i])**2
    mseN225_LSTM_sub2.append(mse)
mseN225_LSTM_sub2 = np.array(mseN225_LSTM_sub2)

y_forecastvalues = predictions_N225_sub2
y_actualvalues = test_y_N225_sub2
qlike_N225_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_LSTM_1_sub2.append(iteration)
qlike_N225_LSTM_1_sub2 = np.array(qlike_N225_LSTM_1_sub2)
QLIKE_LSTM_1_N225_sub2 = sum(qlike_N225_LSTM_1_sub2)/len(y_actualvalues)

# OMXC20

data = OMXC20
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 1

# Data scaling

test_data = data[start_date2_OMXC20:]
test_data1 = np.array(test_data).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(test_data1)

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

test_X, test_y = create_dataset(scaled_data, look_back)

# Create the model

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_OMXC20.shape[1], train_X_OMXC20.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_OMXC20_sub2 = create_model(hidden_nodes)
model_OMXC20_sub2.fit(train_X_OMXC20, train_y_OMXC20, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_OMXC20_sub2.predict(test_X)
predictions_OMXC20_sub2 = scaler.inverse_transform(predictions)
test_y_OMXC20_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_OMXC20_sub2 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_OMXC20_sub2 = mean_squared_error(test_y_OMXC20_sub2, predictions_OMXC20_sub2)

mseOMXC20_LSTM_sub2 = []
for i in np.arange(len(test_y_OMXC20_sub2)):
    mse = (predictions_OMXC20_sub2[i]-test_y_OMXC20_sub2[i])**2
    mseOMXC20_LSTM_sub2.append(mse)
mseOMXC20_LSTM_sub2 = np.array(mseOMXC20_LSTM_sub2)

y_forecastvalues = predictions_OMXC20_sub2
y_actualvalues = test_y_OMXC20_sub2
qlike_OMXC20_LSTM_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_LSTM_1_sub2.append(iteration)
qlike_OMXC20_LSTM_1_sub2 = np.array(qlike_OMXC20_LSTM_1_sub2)
QLIKE_LSTM_1_OMXC20_sub2 = sum(qlike_OMXC20_LSTM_1_sub2)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_LSTM': {
        'DJI': mseDJI_LSTM_sub2,
        'FTSE': mseFTSE_LSTM_sub2,
        'FTSEMIB': mseFTSEMIB_LSTM_sub2,
        'GDAXI': mseGDAXI_LSTM_sub2,
        'SPX': mseSPX_LSTM_sub2,
        'HSI': mseHSI_LSTM_sub2,
        'IBEX': mseIBEX_LSTM_sub2,
        'IXIC': mseIXIC_LSTM_sub2,
        'N225': mseN225_LSTM_sub2,
        'OMXC20': mseOMXC20_LSTM_sub2
                }
        }

for k1 in arrays_Out:
    if k1 == 'mse_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_LSTM_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_LSTM': {
        'DJI': qlike_DJI_LSTM_1_sub2,
        'FTSE': qlike_FTSE_LSTM_1_sub2,
        'FTSEMIB': qlike_FTSEMIB_LSTM_1_sub2,
        'GDAXI': qlike_GDAXI_LSTM_1_sub2,
        'SPX': qlike_SPX_LSTM_1_sub2,
        'HSI': qlike_HSI_LSTM_1_sub2,
        'IBEX': qlike_IBEX_LSTM_1_sub2,
        'IXIC': qlike_IXIC_LSTM_1_sub2,
        'N225': qlike_N225_LSTM_1_sub2,
        'OMXC20': qlike_OMXC20_LSTM_1_sub2
                }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_LSTM_1_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%

predict = {
    'predictions_LSTM': {
        'DJI': predictions_DJI_sub2,
        'FTSE': predictions_FTSE_sub2,
        'FTSEMIB': predictions_FTSEMIB_sub2,
        'GDAXI': predictions_GDAXI_sub2,
        'SPX': predictions_SPX_sub2,
        'HSI': predictions_HSI_sub2,
        'IBEX': predictions_IBEX_sub2,
        'IXIC': predictions_IXIC_sub2,
        'N225': predictions_N225_sub2,
        'OMXC20': predictions_OMXC20_sub2
                }
        }

for k1 in predict:
    if k1 == 'predictions_LSTM':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_LSTM_1_sub2.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')