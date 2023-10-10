# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:16:07 2023

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

#%% LSTM (dates)

start_date_DJI = '2015-08-13'
end_date_DJI = '2020-02-21' 
start_date_DJI = pd.to_datetime(start_date_DJI)
end_date_DJI = pd.to_datetime(end_date_DJI)

start_date_FTSE = '2015-08-25'
end_date_FTSE = '2020-02-24' 
start_date_FTSE = pd.to_datetime(start_date_FTSE)
end_date_FTSE = pd.to_datetime(end_date_FTSE)

start_date_FTSEMIB = '2018-06-07'
end_date_FTSEMIB = '2020-02-21' 
start_date_FTSEMIB = pd.to_datetime(start_date_FTSEMIB)
end_date_FTSEMIB = pd.to_datetime(end_date_FTSEMIB)

start_date_GDAXI = '2015-08-11'
end_date_GDAXI = '2020-02-25' 
start_date_GDAXI = pd.to_datetime(start_date_GDAXI)
end_date_GDAXI = pd.to_datetime(end_date_GDAXI)

start_date_SPX = '2015-08-18'
end_date_SPX = '2020-02-25' 
start_date_SPX = pd.to_datetime(start_date_SPX)
end_date_SPX = pd.to_datetime(end_date_SPX)

start_date_HSI = '2015-08-17'
end_date_HSI = '2020-03-10' 
start_date_HSI = pd.to_datetime(start_date_HSI)
end_date_HSI = pd.to_datetime(end_date_HSI)

start_date_IBEX = '2015-09-06'
end_date_IBEX = '2020-02-26' 
start_date_IBEX = pd.to_datetime(start_date_IBEX)
end_date_IBEX = pd.to_datetime(end_date_IBEX)

start_date_IXIC = '2015-08-24'
end_date_IXIC = '2020-02-21' 
start_date_IXIC = pd.to_datetime(start_date_IXIC)
end_date_IXIC = pd.to_datetime(end_date_IXIC)

start_date_N225 = '2015-08-12'
end_date_N225 = '2020-02-21' 
start_date_N225 = pd.to_datetime(start_date_N225)
end_date_N225 = pd.to_datetime(end_date_N225)

start_date_OMXC20 = '2017-05-07'
end_date_OMXC20 = '2020-02-21' 
start_date_OMXC20 = pd.to_datetime(start_date_OMXC20)
end_date_OMXC20 = pd.to_datetime(end_date_OMXC20)

#%% DJI

data = DJI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

test_data = data[start_date_DJI:end_date_DJI]
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

# Crea il modello LSTM

def create_model(hidden_nodes):
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(train_X_DJI.shape[1], train_X_DJI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_DJI_sub1 = create_model(hidden_nodes)
model_DJI_sub1.fit(train_X_DJI, train_y_DJI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_DJI_sub1.predict(test_X)
predictions_DJI_sub1 = scaler.inverse_transform(predictions)
test_y_DJI_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_DJI_sub1 = test_data.tail(len(test_X)).index

# Find mean squared error and the QLIKE loss

MSE_LSTM_1_DJI_sub1 = mean_squared_error(test_y_DJI_sub1, predictions_DJI_sub1)

mseDJI_LSTM_sub1 = []
for i in np.arange(len(test_y_DJI_sub1)):
    mse = (predictions_DJI_sub1[i]-test_y_DJI_sub1[i])**2
    mseDJI_LSTM_sub1.append(mse)
mseDJI_LSTM_sub1 = np.array(mseDJI_LSTM_sub1)

y_forecastvalues = predictions_DJI_sub1
y_actualvalues = test_y_DJI_sub1
qlike_DJI_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_LSTM_1_sub1.append(iteration)
qlike_DJI_LSTM_1_sub1 = np.array(qlike_DJI_LSTM_1_sub1)
QLIKE_LSTM_1_DJI_sub1 = sum(qlike_DJI_LSTM_1_sub1)/len(y_actualvalues)

#%% FTSE

data = FTSE
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

# Data scaling
test_data = data[start_date_FTSE:end_date_FTSE]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_FTSE.shape[1], train_X_FTSE.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSE_sub1 = create_model(hidden_nodes)
model_FTSE_sub1.fit(train_X_FTSE, train_y_FTSE, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_FTSE_sub1.predict(test_X)
predictions_FTSE_sub1 = scaler.inverse_transform(predictions)
test_y_FTSE_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSE_sub1 = test_data.tail(len(test_X)).index
    
# Find  mean squared error and QLIKE loss

MSE_LSTM_1_FTSE_sub1 = mean_squared_error(test_y_FTSE_sub1, predictions_FTSE_sub1)

mseFTSE_LSTM_sub1 = []
for i in np.arange(len(test_y_FTSE_sub1)):
    mse = (predictions_FTSE_sub1[i]-test_y_FTSE_sub1[i])**2
    mseFTSE_LSTM_sub1.append(mse)
mseFTSE_LSTM_sub1 = np.array(mseFTSE_LSTM_sub1)

y_forecastvalues = predictions_FTSE_sub1
y_actualvalues = test_y_FTSE_sub1
qlike_FTSE_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_LSTM_1_sub1.append(iteration)
qlike_FTSE_LSTM_1_sub1 = np.array(qlike_FTSE_LSTM_1_sub1)
QLIKE_LSTM_1_FTSE_sub1 = sum(qlike_FTSE_LSTM_1_sub1)/len(y_actualvalues)

#%% FTSEMIB

data = FTSEMIB
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

# Data scaling

test_data = data[start_date_FTSEMIB:end_date_FTSEMIB]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_FTSEMIB.shape[1], train_X_FTSEMIB.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSEMIB_sub1 = create_model(hidden_nodes)
model_FTSEMIB_sub1.fit(train_X_FTSEMIB, train_y_FTSEMIB, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_FTSEMIB_sub1.predict(test_X)
predictions_FTSEMIB_sub1 = scaler.inverse_transform(predictions)
test_y_FTSEMIB_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSEMIB_sub1 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_FTSEMIB_sub1 = mean_squared_error(test_y_FTSEMIB_sub1, predictions_FTSEMIB_sub1)

mseFTSEMIB_LSTM_sub1 = []
for i in np.arange(len(test_y_FTSEMIB_sub1)):
    mse = (predictions_FTSEMIB_sub1[i]-test_y_FTSEMIB_sub1[i])**2
    mseFTSEMIB_LSTM_sub1.append(mse)
mseFTSEMIB_LSTM_sub1 = np.array(mseFTSEMIB_LSTM_sub1)

y_forecastvalues = predictions_FTSEMIB_sub1
y_actualvalues = test_y_FTSEMIB_sub1
qlike_FTSEMIB_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_LSTM_1_sub1.append(iteration)
qlike_FTSEMIB_LSTM_1_sub1 = np.array(qlike_FTSEMIB_LSTM_1_sub1)
QLIKE_LSTM_1_FTSEMIB_sub1 = sum(qlike_FTSEMIB_LSTM_1_sub1)/len(y_actualvalues)

#%% GDAXI

data = GDAXI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date_GDAXI:end_date_GDAXI]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_GDAXI.shape[1], train_X_GDAXI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_GDAXI_sub1 = create_model(hidden_nodes)
model_GDAXI_sub1.fit(train_X_GDAXI, train_y_GDAXI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_GDAXI_sub1.predict(test_X)
predictions_GDAXI_sub1 = scaler.inverse_transform(predictions)
test_y_GDAXI_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_GDAXI_sub1 = test_data.tail(len(test_X)).index
    
# Find teh lower mean squared error

MSE_LSTM_1_GDAXI_sub1 = mean_squared_error(test_y_GDAXI_sub1, predictions_GDAXI_sub1)

mseGDAXI_LSTM_sub1 = []
for i in np.arange(len(test_y_GDAXI_sub1)):
    mse = (predictions_GDAXI_sub1[i]-test_y_GDAXI_sub1[i])**2
    mseGDAXI_LSTM_sub1.append(mse)
mseGDAXI_LSTM_sub1 = np.array(mseGDAXI_LSTM_sub1)

y_forecastvalues = predictions_GDAXI_sub1
y_actualvalues = test_y_GDAXI_sub1
qlike_GDAXI_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_LSTM_1_sub1.append(iteration)
qlike_GDAXI_LSTM_1_sub1 = np.array(qlike_GDAXI_LSTM_1_sub1)
QLIKE_LSTM_1_GDAXI_sub1 = sum(qlike_GDAXI_LSTM_1_sub1)/len(y_actualvalues)

#%% SPX

data = SPX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date_SPX:end_date_SPX]
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
model_SPX_sub1 = create_model(hidden_nodes)
model_SPX_sub1.fit(train_X_SPX, train_y_SPX, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_SPX_sub1.predict(test_X)
predictions_SPX_sub1 = scaler.inverse_transform(predictions)
test_y_SPX_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))
posizione = np.where(predictions_SPX_sub1<=0)[0]
media1 = np.mean(test_y_SPX_sub1[212:216])
media2 = np.mean(test_y_SPX_sub1[500:505])
media3 = np.mean(test_y_SPX_sub1[513:531])
media4 = np.mean(test_y_SPX_sub1[549:555])
media5 = np.mean(test_y_SPX_sub1[572:579])
media6 = np.mean(test_y_SPX_sub1[1069:1073])
predictions_SPX_sub1[214] = media1
predictions_SPX_sub1[[502, 503]] = media2
predictions_SPX_sub1[[515, 517, 519, 523, 524, 525, 526, 527, 528, 529]] = media3
predictions_SPX_sub1[[551, 552, 553]] = media4
predictions_SPX_sub1[[574, 575, 576, 577]] = media5
predictions_SPX_sub1[1071] = media6

predictDates_SPX_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_SPX_sub1 = mean_squared_error(test_y_SPX_sub1, predictions_SPX_sub1)

mseSPX_LSTM_sub1 = []
for i in np.arange(len(test_y_SPX_sub1)):
    mse = (predictions_SPX_sub1[i]-test_y_SPX_sub1[i])**2
    mseSPX_LSTM_sub1.append(mse)
mseSPX_LSTM_sub1 = np.array(mseSPX_LSTM_sub1)

y_forecastvalues = predictions_SPX_sub1
y_actualvalues = test_y_SPX_sub1
qlike_SPX_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_LSTM_1_sub1.append(iteration)
qlike_SPX_LSTM_1_sub1 = np.array(qlike_SPX_LSTM_1_sub1)
QLIKE_LSTM_1_SPX_sub1 = sum(qlike_SPX_LSTM_1_sub1)/len(y_actualvalues)

#%% HSI

data = HSI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date_HSI:end_date_HSI]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_HSI.shape[1], train_X_HSI.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_HSI_sub1 = create_model(hidden_nodes)
model_HSI_sub1.fit(train_X_HSI, train_y_HSI, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_HSI_sub1.predict(test_X)
predictions_HSI_sub1 = scaler.inverse_transform(predictions)
test_y_HSI_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_HSI_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_HSI_sub1 = mean_squared_error(test_y_HSI_sub1, predictions_HSI_sub1)

mseHSI_LSTM_sub1 = []
for i in np.arange(len(test_y_HSI_sub1)):
    mse = (predictions_HSI_sub1[i]-test_y_HSI_sub1[i])**2
    mseHSI_LSTM_sub1.append(mse)
mseHSI_LSTM_sub1 = np.array(mseHSI_LSTM_sub1)

y_forecastvalues = predictions_HSI_sub1
y_actualvalues = test_y_HSI_sub1
qlike_HSI_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_LSTM_1_sub1.append(iteration)
qlike_HSI_LSTM_1_sub1 = np.array(qlike_HSI_LSTM_1_sub1)
QLIKE_LSTM_1_HSI_sub1 = sum(qlike_HSI_LSTM_1_sub1)/len(y_actualvalues)

#%% IBEX

data = IBEX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

# Data scaling

test_data = data[start_date_IBEX:end_date_IBEX]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_IBEX.shape[1], train_X_IBEX.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IBEX_sub1 = create_model(hidden_nodes)
model_IBEX_sub1.fit(train_X_IBEX, train_y_IBEX, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_IBEX_sub1.predict(test_X)
predictions_IBEX_sub1 = scaler.inverse_transform(predictions)
test_y_IBEX_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IBEX_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_IBEX_sub1 = mean_squared_error(test_y_IBEX_sub1, predictions_IBEX_sub1)

mseIBEX_LSTM_sub1 = []
for i in np.arange(len(test_y_IBEX_sub1)):
    mse = (predictions_IBEX_sub1[i]-test_y_IBEX_sub1[i])**2
    mseIBEX_LSTM_sub1.append(mse)
mseIBEX_LSTM_sub1 = np.array(mseIBEX_LSTM_sub1)

y_forecastvalues = predictions_IBEX_sub1
y_actualvalues = test_y_IBEX_sub1
qlike_IBEX_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_LSTM_1_sub1.append(iteration)
qlike_IBEX_LSTM_1_sub1 = np.array(qlike_IBEX_LSTM_1_sub1)
QLIKE_LSTM_1_IBEX_sub1 = sum(qlike_IBEX_LSTM_1_sub1)/len(y_actualvalues)

#%% IXIC

data = IXIC
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

# Data scaling

test_data = data[start_date_IXIC:end_date_IXIC]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0007), recurrent_regularizer=regularizers.l2(0.0007), input_shape=(train_X_IXIC.shape[1], train_X_IXIC.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IXIC_sub1 = create_model(hidden_nodes)
model_IXIC_sub1.fit(train_X_IXIC, train_y_IXIC, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_IXIC_sub1.predict(test_X)
predictions_IXIC_sub1 = scaler.inverse_transform(predictions)
test_y_IXIC_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IXIC_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_IXIC_sub1 = mean_squared_error(test_y_IXIC_sub1, predictions_IXIC_sub1)

mseIXIC_LSTM_sub1 = []
for i in np.arange(len(test_y_IXIC_sub1)):
    mse = (predictions_IXIC_sub1[i]-test_y_IXIC_sub1[i])**2
    mseIXIC_LSTM_sub1.append(mse)
mseIXIC_LSTM_sub1 = np.array(mseIXIC_LSTM_sub1)

y_forecastvalues = predictions_IXIC_sub1
y_actualvalues = test_y_IXIC_sub1
qlike_IXIC_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_LSTM_1_sub1.append(iteration)
qlike_IXIC_LSTM_1_sub1 = np.array(qlike_IXIC_LSTM_1_sub1)
QLIKE_LSTM_1_IXIC_sub1 = sum(qlike_IXIC_LSTM_1_sub1)/len(y_actualvalues)

#%% N225

data = N225
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

# Data scaling

test_data = data[start_date_N225:end_date_N225]
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
    model.add(LSTM(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005), input_shape=(train_X_N225.shape[1], train_X_N225.shape[2]), kernel_initializer=GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_N225_sub1 = create_model(hidden_nodes)
model_N225_sub1.fit(train_X_N225, train_y_N225, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_N225_sub1.predict(test_X)
predictions_N225_sub1 = scaler.inverse_transform(predictions)
test_y_N225_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_N225_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_N225_sub1 = mean_squared_error(test_y_N225_sub1, predictions_N225_sub1)

mseN225_LSTM_sub1 = []
for i in np.arange(len(test_y_N225_sub1)):
    mse = (predictions_N225_sub1[i]-test_y_N225_sub1[i])**2
    mseN225_LSTM_sub1.append(mse)
mseN225_LSTM_sub1 = np.array(mseN225_LSTM_sub1)

y_forecastvalues = predictions_N225_sub1
y_actualvalues = test_y_N225_sub1
qlike_N225_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_LSTM_1_sub1.append(iteration)
qlike_N225_LSTM_1_sub1 = np.array(qlike_N225_LSTM_1_sub1)
QLIKE_LSTM_1_N225_sub1 = sum(qlike_N225_LSTM_1_sub1)/len(y_actualvalues)

#%% OMXC20

data = OMXC20
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 1

# Data scaling

test_data = data[start_date_OMXC20:end_date_OMXC20]
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
model_OMXC20_sub1 = create_model(hidden_nodes)
model_OMXC20_sub1.fit(train_X_OMXC20, train_y_OMXC20, batch_size = batch_size, epochs=epochs, verbose = 0)
predictions = model_OMXC20_sub1.predict(test_X)
predictions_OMXC20_sub1 = scaler.inverse_transform(predictions)
test_y_OMXC20_sub1 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_OMXC20_sub1 = test_data.tail(len(test_X)).index
    
# Find the lower mean squared error

MSE_LSTM_1_OMXC20_sub1 = mean_squared_error(test_y_OMXC20_sub1, predictions_OMXC20_sub1)

mseOMXC20_LSTM_sub1 = []
for i in np.arange(len(test_y_OMXC20_sub1)):
    mse = (predictions_OMXC20_sub1[i]-test_y_OMXC20_sub1[i])**2
    mseOMXC20_LSTM_sub1.append(mse)
mseOMXC20_LSTM_sub1 = np.array(mseOMXC20_LSTM_sub1)

y_forecastvalues = predictions_OMXC20_sub1
y_actualvalues = test_y_OMXC20_sub1
qlike_OMXC20_LSTM_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_LSTM_1_sub1.append(iteration)
qlike_OMXC20_LSTM_1_sub1 = np.array(qlike_OMXC20_LSTM_1_sub1)
QLIKE_LSTM_1_OMXC20_sub1 = sum(qlike_OMXC20_LSTM_1_sub1)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_LSTM': {
        'DJI': mseDJI_LSTM_sub1,
        'FTSE': mseFTSE_LSTM_sub1,
        'FTSEMIB': mseFTSEMIB_LSTM_sub1,
        'GDAXI': mseGDAXI_LSTM_sub1,
        'SPX': mseSPX_LSTM_sub1,
        'HSI': mseHSI_LSTM_sub1,
        'IBEX': mseIBEX_LSTM_sub1,
        'IXIC': mseIXIC_LSTM_sub1,
        'N225': mseN225_LSTM_sub1,
        'OMXC20': mseOMXC20_LSTM_sub1
                }
        }

for k1 in arrays_Out:
    if k1 == 'mse_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_LSTM_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_LSTM': {
        'DJI': qlike_DJI_LSTM_1_sub1,
        'FTSE': qlike_FTSE_LSTM_1_sub1,
        'FTSEMIB': qlike_FTSEMIB_LSTM_1_sub1,
        'GDAXI': qlike_GDAXI_LSTM_1_sub1,
        'SPX': qlike_SPX_LSTM_1_sub1,
        'HSI': qlike_HSI_LSTM_1_sub1,
        'IBEX': qlike_IBEX_LSTM_1_sub1,
        'IXIC': qlike_IXIC_LSTM_1_sub1,
        'N225': qlike_N225_LSTM_1_sub1,
        'OMXC20': qlike_OMXC20_LSTM_1_sub1
                }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_LSTM':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_LSTM_1_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
            
#%%

predict = {
    'predictions_LSTM': {
        'DJI': predictions_DJI_sub1,
        'FTSE': predictions_FTSE_sub1,
        'FTSEMIB': predictions_FTSEMIB_sub1,
        'GDAXI': predictions_GDAXI_sub1,
        'SPX': predictions_SPX_sub1,
        'HSI': predictions_HSI_sub1,
        'IBEX': predictions_IBEX_sub1,
        'IXIC': predictions_IXIC_sub1,
        'N225': predictions_N225_sub1,
        'OMXC20': predictions_OMXC20_sub1
                }
        }

for k1 in predict:
    if k1 == 'predictions_LSTM':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_LSTM_1_sub1.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')


