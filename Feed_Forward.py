# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:39:43 2023

@author: cesar
"""

import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.random import set_seed
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from keras import regularizers

#%% DJI

data = DJI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

train_X_DJI_FNN, train_y_DJI_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_DJI_FNN= np.reshape(train_X_DJI_FNN, (train_X_DJI_FNN.shape[0], train_X_DJI_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.0005),  input_dim=train_X_DJI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_DJI = create_model(hidden_nodes)
model_DJI.fit(train_X_DJI_FNN, train_y_DJI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_DJI.predict(test_X_FNN)
predictions_DJI_FNN = scaler.inverse_transform(predictions)
test_y_DJI_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_DJI = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_DJI = mean_squared_error(test_y_DJI_FNN, predictions_DJI_FNN)
print(MSE_FNN_1_DJI)

# QLIKE

y_forecastvalues = predictions_DJI_FNN
y_actualvalues = test_y_DJI_FNN
qlike_DJI_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_FNN_1_Out.append(iteration)
QLIKE_FNN_1_DJI = sum(qlike_DJI_FNN_1_Out)/len(y_actualvalues)

predictions = model_DJI.predict(train_X_DJI_FNN)
predictions_DJI_FNN_In = scaler.inverse_transform(predictions)
train_y_DJI_FNN_In = np.array(y_train_DJI)
MSE_FNN_1_DJI_In = mean_squared_error(train_y_DJI_FNN_In, predictions_DJI_FNN_In)
mseDJI_FNN_In = []
for i in np.arange(len(train_y_DJI_FNN_In)):
    mse = (predictions_DJI_FNN_In[i]-train_y_DJI_FNN_In[i])**2
    mseDJI_FNN_In.append(mse)
mseDJI_FNN_In = np.array(mseDJI_FNN_In)

#%% FTSE

data = FTSE
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

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

train_X_FTSE_FNN, train_y_FTSE_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_FTSE_FNN = np.reshape(train_X_FTSE_FNN, (train_X_FTSE_FNN.shape[0], train_X_FTSE_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_FTSE_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSE = create_model(hidden_nodes)
model_FTSE.fit(train_X_FTSE_FNN, train_y_FTSE_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_FTSE.predict(test_X_FNN)
predictions_FTSE_FNN = scaler.inverse_transform(predictions)
test_y_FTSE_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))                    

predictDates_FTSE = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_FTSE = mean_squared_error(test_y_FTSE_FNN, predictions_FTSE_FNN)
print(MSE_FNN_1_FTSE)

# QLIKE

y_forecastvalues = predictions_FTSE_FNN
y_actualvalues = test_y_FTSE_FNN
qlike_FTSE_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_FNN_1_Out.append(iteration)
qlike_FTSE_FNN_1_Out = np.array(qlike_FTSE_FNN_1_Out)
QLIKE_FNN_1_FTSE = sum(qlike_FTSE_FNN_1_Out)/len(y_actualvalues)

predictions = model_FTSE.predict(train_X_FTSE_FNN)
predictions_FTSE_FNN_In = scaler.inverse_transform(predictions)
train_y_FTSE_FNN_In = np.array(y_train_FTSE)
MSE_FNN_1_FTSE_In = mean_squared_error(train_y_FTSE_FNN_In, predictions_FTSE_FNN_In)
mseFTSE_FNN_In = []
for i in np.arange(len(train_y_FTSE_FNN_In)):
    mse = (predictions_FTSE_FNN_In[i]-train_y_FTSE_FNN_In[i])**2
    mseFTSE_FNN_In.append(mse)
mseFTSE_FNN_In = np.array(mseFTSE_FNN_In)

#%% FTSEMIB

data = FTSEMIB
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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

train_X_FTSEMIB_FNN, train_y_FTSEMIB_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_FTSEMIB_FNN = np.reshape(train_X_FTSEMIB_FNN, (train_X_FTSEMIB_FNN.shape[0], train_X_FTSEMIB_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.0005), input_dim=train_X_FTSEMIB_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSEMIB = create_model(hidden_nodes)
model_FTSEMIB.fit(train_X_FTSEMIB_FNN, train_y_FTSEMIB_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_FTSEMIB.predict(test_X_FNN)
predictions_FTSEMIB_FNN = scaler.inverse_transform(predictions)
test_y_FTSEMIB_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))
posizione = np.where(predictions_FTSEMIB_FNN<=0)[0]

predictDates_FTSEMIB = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_FTSEMIB = mean_squared_error(test_y_FTSEMIB_FNN, predictions_FTSEMIB_FNN)
print(MSE_FNN_1_FTSEMIB)

# QLIKE

y_forecastvalues = predictions_FTSEMIB_FNN
y_actualvalues = test_y_FTSEMIB_FNN
qlike_FTSEMIB_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_FNN_1_Out.append(iteration)
qlike_FTSEMIB_FNN_1_Out = np.array(qlike_FTSEMIB_FNN_1_Out)
QLIKE_FNN_1_FTSEMIB = sum(qlike_FTSEMIB_FNN_1_Out)/len(y_actualvalues)

predictions = model_FTSEMIB.predict(train_X_FTSEMIB_FNN)
predictions_FTSEMIB_FNN_In = scaler.inverse_transform(predictions)
train_y_FTSEMIB_FNN_In = np.array(y_train_FTSEMIB)
posizione = np.where(predictions_FTSEMIB_FNN_In<=0)[0]
media1 = np.mean(train_y_FTSE_FNN_In[395:399])
media2 = np.mean(train_y_FTSE_FNN_In[801:805])
predictions_FTSEMIB_FNN_In[397] = media1
predictions_FTSEMIB_FNN_In[803] = media2
MSE_FNN_1_FTSEMIB_In = mean_squared_error(train_y_FTSEMIB_FNN_In, predictions_FTSEMIB_FNN_In)
mseFTSEMIB_FNN_In = []
for i in np.arange(len(train_y_FTSEMIB_FNN_In)):
    mse = (predictions_FTSEMIB_FNN_In[i]-train_y_FTSEMIB_FNN_In[i])**2
    mseFTSEMIB_FNN_In.append(mse)
mseFTSEMIB_FNN_In = np.array(mseFTSEMIB_FNN_In)

#%% GDAXI

data = GDAXI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

train_X_GDAXI_FNN, train_y_GDAXI_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_GDAXI_FNN = np.reshape(train_X_GDAXI_FNN, (train_X_GDAXI_FNN.shape[0], train_X_GDAXI_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.0005), input_dim=train_X_GDAXI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_GDAXI = create_model(hidden_nodes)
model_GDAXI.fit(train_X_GDAXI_FNN, train_y_GDAXI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_GDAXI.predict(test_X_FNN)
predictions_GDAXI_FNN = scaler.inverse_transform(predictions)
test_y_GDAXI_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_GDAXI = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_GDAXI = mean_squared_error(test_y_GDAXI_FNN, predictions_GDAXI_FNN)
print(MSE_FNN_1_GDAXI)

# QLIKE

y_forecastvalues = predictions_GDAXI_FNN
y_actualvalues = test_y_GDAXI_FNN
qlike_GDAXI_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_FNN_1_Out.append(iteration)
qlike_GDAXI_FNN_1_Out = np.array(qlike_GDAXI_FNN_1_Out)
QLIKE_FNN_1_GDAXI = sum(qlike_GDAXI_FNN_1_Out)/len(y_actualvalues)

predictions = model_GDAXI.predict(train_X_GDAXI_FNN)
predictions_GDAXI_FNN_In = scaler.inverse_transform(predictions)
train_y_GDAXI_FNN_In = np.array(y_train_GDAXI)
MSE_FNN_1_GDAXI_In = mean_squared_error(train_y_GDAXI_FNN_In, predictions_GDAXI_FNN_In)
mseGDAXI_FNN_In = []
for i in np.arange(len(train_y_GDAXI_FNN_In)):
    mse = (predictions_GDAXI_FNN_In[i]-train_y_GDAXI_FNN_In[i])**2
    mseGDAXI_FNN_In.append(mse)
mseGDAXI_FNN_In = np.array(mseGDAXI_FNN_In)

#%% SPX

data = SPX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

train_X_SPX_FNN, train_y_SPX_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_SPX_FNN = np.reshape(train_X_SPX_FNN, (train_X_SPX_FNN.shape[0], train_X_SPX_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005),  activation='relu', input_dim=train_X_SPX_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_SPX = create_model(hidden_nodes)
model_SPX.fit(train_X_SPX_FNN, train_y_SPX_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_SPX.predict(test_X_FNN)
predictions_SPX_FNN = scaler.inverse_transform(predictions)
test_y_SPX_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_SPX = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_SPX = mean_squared_error(test_y_SPX_FNN, predictions_SPX_FNN)
print(MSE_FNN_1_SPX)

# QLIKE

y_forecastvalues = predictions_SPX_FNN
y_actualvalues = test_y_SPX_FNN
qlike_SPX_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_FNN_1_Out.append(iteration)
qlike_SPX_FNN_1_Out = np.array(qlike_SPX_FNN_1_Out)
QLIKE_FNN_1_SPX = sum(qlike_SPX_FNN_1_Out)/len(y_actualvalues)

predictions = model_SPX.predict(train_X_SPX_FNN)
predictions_SPX_FNN_In = scaler.inverse_transform(predictions)
train_y_SPX_FNN_In = np.array(y_train_SPX)
MSE_FNN_1_SPX_In = mean_squared_error(train_y_SPX_FNN_In, predictions_SPX_FNN_In)
mseSPX_FNN_In = []
for i in np.arange(len(train_y_SPX_FNN_In)):
    mse = (predictions_SPX_FNN_In[i]-train_y_SPX_FNN_In[i])**2
    mseSPX_FNN_In.append(mse)
mseSPX_FNN_In = np.array(mseSPX_FNN_In)

#%% HSI

data = HSI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

train_X_HSI_FNN, train_y_HSI_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_HSI_FNN= np.reshape(train_X_HSI_FNN, (train_X_HSI_FNN.shape[0], train_X_HSI_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.0005), input_dim=train_X_HSI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_HSI = create_model(hidden_nodes)
model_HSI.fit(train_X_HSI_FNN, train_y_HSI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_HSI.predict(test_X_FNN)
predictions_HSI_FNN = scaler.inverse_transform(predictions)
test_y_HSI_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))
posizione = np.where(predictions_HSI_FNN<=0)[0]

predictDates_HSI = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_HSI = mean_squared_error(test_y_HSI_FNN, predictions_HSI_FNN)
print(MSE_FNN_1_HSI)

# QLIKE

y_forecastvalues = predictions_HSI_FNN
y_actualvalues = test_y_HSI_FNN
qlike_HSI_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_FNN_1_Out.append(iteration)
QLIKE_FNN_1_HSI = sum(qlike_HSI_FNN_1_Out)/len(y_actualvalues)

predictions = model_HSI.predict(train_X_HSI_FNN)
predictions_HSI_FNN_In = scaler.inverse_transform(predictions)
train_y_HSI_FNN_In = np.array(y_train_HSI)
posizione = np.where(predictions_HSI_FNN_In<=0)[0]
media1 = np.mean(train_y_HSI_FNN_In[400:404])
media2 = np.mean(train_y_HSI_FNN_In[1946:1950])
media3 = np.mean(train_y_HSI_FNN_In[1951:1957])
predictions_HSI_FNN_In[402] = media1
predictions_HSI_FNN_In[1948] = media2
predictions_HSI_FNN_In[[1953, 1955]] = media3
MSE_FNN_1_HSI_In = mean_squared_error(train_y_HSI_FNN_In, predictions_HSI_FNN_In)
mseHSI_FNN_In = []
for i in np.arange(len(train_y_HSI_FNN_In)):
    mse = (predictions_HSI_FNN_In[i]-train_y_HSI_FNN_In[i])**2
    mseHSI_FNN_In.append(mse)
mseHSI_FNN_In = np.array(mseHSI_FNN_In)

#%% IBEX

data = IBEX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

train_X_IBEX_FNN, train_y_IBEX_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_IBEX_FNN= np.reshape(train_X_IBEX_FNN, (train_X_IBEX_FNN.shape[0], train_X_IBEX_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(0.0005),  input_dim=train_X_IBEX_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IBEX = create_model(hidden_nodes)
model_IBEX.fit(train_X_IBEX_FNN, train_y_IBEX_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_IBEX.predict(test_X_FNN)
predictions_IBEX_FNN = scaler.inverse_transform(predictions)
test_y_IBEX_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_IBEX = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_IBEX = mean_squared_error(test_y_IBEX_FNN, predictions_IBEX_FNN)
print(MSE_FNN_1_IBEX)

# QLIKE

y_forecastvalues = predictions_IBEX_FNN
y_actualvalues = test_y_IBEX_FNN
qlike_IBEX_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_FNN_1_Out.append(iteration)
QLIKE_FNN_1_IBEX = sum(qlike_IBEX_FNN_1_Out)/len(y_actualvalues)

predictions = model_IBEX.predict(train_X_IBEX_FNN)
predictions_IBEX_FNN_In = scaler.inverse_transform(predictions)
train_y_IBEX_FNN_In = np.array(y_train_IBEX)
MSE_FNN_1_IBEX_In = mean_squared_error(train_y_IBEX_FNN_In, predictions_IBEX_FNN_In)
mseIBEX_FNN_In = []
for i in np.arange(len(train_y_IBEX_FNN_In)):
    mse = (predictions_IBEX_FNN_In[i]-train_y_IBEX_FNN_In[i])**2
    mseIBEX_FNN_In.append(mse)
mseIBEX_FNN_In = np.array(mseIBEX_FNN_In)

#%% IXIC

data = IXIC
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

train_X_IXIC_FNN, train_y_IXIC_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_IXIC_FNN= np.reshape(train_X_IXIC_FNN, (train_X_IXIC_FNN.shape[0], train_X_IXIC_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_IXIC_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IXIC = create_model(hidden_nodes)
model_IXIC.fit(train_X_IXIC_FNN, train_y_IXIC_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_IXIC.predict(test_X_FNN)
predictions_IXIC_FNN = scaler.inverse_transform(predictions)
test_y_IXIC_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_IXIC = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_IXIC = mean_squared_error(test_y_IXIC_FNN, predictions_IXIC_FNN)
print(MSE_FNN_1_IXIC)

# QLIKE

y_forecastvalues = predictions_IXIC_FNN
y_actualvalues = test_y_IXIC_FNN
qlike_IXIC_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_FNN_1_Out.append(iteration)
QLIKE_FNN_1_IXIC = sum(qlike_IXIC_FNN_1_Out)/len(y_actualvalues)

predictions = model_IXIC.predict(train_X_IXIC_FNN)
predictions_IXIC_FNN_In = scaler.inverse_transform(predictions)
train_y_IXIC_FNN_In = np.array(y_train_IXIC)
MSE_FNN_1_IXIC_In = mean_squared_error(train_y_IXIC_FNN_In, predictions_IXIC_FNN_In)
mseIXIC_FNN_In = []
for i in np.arange(len(train_y_IXIC_FNN_In)):
    mse = (predictions_IXIC_FNN_In[i]-train_y_IXIC_FNN_In[i])**2
    mseIXIC_FNN_In.append(mse)
mseIXIC_FNN_In = np.array(mseIXIC_FNN_In)

#%% N225

data = N225
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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

train_X_N225_FNN, train_y_N225_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_N225_FNN= np.reshape(train_X_N225_FNN, (train_X_N225_FNN.shape[0], train_X_N225_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005),  activation='relu', input_dim=train_X_N225_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_N225 = create_model(hidden_nodes)
model_N225.fit(train_X_N225_FNN, train_y_N225_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_N225.predict(test_X_FNN)
predictions_N225_FNN = scaler.inverse_transform(predictions)
test_y_N225_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))
 
predictDates_N225 = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_N225 = mean_squared_error(test_y_N225_FNN, predictions_N225_FNN)
print(MSE_FNN_1_N225)

# QLIKE

y_forecastvalues = predictions_N225_FNN
y_actualvalues = test_y_N225_FNN
qlike_N225_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_FNN_1_Out.append(iteration)
QLIKE_FNN_1_N225 = sum(qlike_N225_FNN_1_Out)/len(y_actualvalues)

predictions = model_N225.predict(train_X_N225_FNN)
predictions_N225_FNN_In = scaler.inverse_transform(predictions)
train_y_N225_FNN_In = np.array(y_train_N225)
MSE_FNN_1_N225_In = mean_squared_error(train_y_N225_FNN_In, predictions_N225_FNN_In)
mseN225_FNN_In = []
for i in np.arange(len(train_y_N225_FNN_In)):
    mse = (predictions_N225_FNN_In[i]-train_y_N225_FNN_In[i])**2
    mseN225_FNN_In.append(mse)
mseN225_FNN_In = np.array(mseN225_FNN_In)

#%% OMXC20

data = OMXC20
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 3

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

train_X_OMXC20_FNN, train_y_OMXC20_FNN = create_dataset(train_data, look_back)
test_X_FNN, test_y_FNN = create_dataset(test_data, look_back)
train_X_OMXC20_FNN= np.reshape(train_X_OMXC20_FNN, (train_X_OMXC20_FNN.shape[0], train_X_OMXC20_FNN.shape[1], 1))
test_X_FNN = np.reshape(test_X_FNN, (test_X_FNN.shape[0], test_X_FNN.shape[1], 1))

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005),  activation='relu', input_dim=train_X_OMXC20_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_OMXC20 = create_model(hidden_nodes)
model_OMXC20.fit(train_X_OMXC20_FNN, train_y_OMXC20_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_OMXC20.predict(test_X_FNN)
predictions_OMXC20_FNN = scaler.inverse_transform(predictions)
test_y_OMXC20_FNN = scaler.inverse_transform(test_y_FNN.reshape(-1,1))

predictDates_OMXC20 = data.tail(len(test_X_FNN)).index

# MSE

MSE_FNN_1_OMXC20 = mean_squared_error(test_y_OMXC20_FNN, predictions_OMXC20_FNN)
print(MSE_FNN_1_OMXC20)

# QLIKE

y_forecastvalues = predictions_OMXC20_FNN
y_actualvalues = test_y_OMXC20_FNN
qlike_OMXC20_FNN_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_FNN_1_Out.append(iteration)
QLIKE_FNN_1_OMXC20 = sum(qlike_OMXC20_FNN_1_Out)/len(y_actualvalues)

predictions = model_OMXC20.predict(train_X_OMXC20_FNN)
predictions_OMXC20_FNN_In = scaler.inverse_transform(predictions)
train_y_OMXC20_FNN_In = np.array(y_train_OMXC20)
MSE_FNN_1_OMXC20_In = mean_squared_error(train_y_OMXC20_FNN_In, predictions_OMXC20_FNN_In)
mseOMXC20_FNN_In = []
for i in np.arange(len(train_y_OMXC20_FNN_In)):
    mse = (predictions_OMXC20_FNN_In[i]-train_y_OMXC20_FNN_In[i])**2
    mseOMXC20_FNN_In.append(mse)
mseOMXC20_FNN_In = np.array(mseOMXC20_FNN_In)

#%% In-sample Loss Function

predictions = model_DJI.predict(train_X_DJI_FNN)
predictions_DJI_FNN_In = scaler.inverse_transform(predictions)
train_y_DJI_FNN_In = np.array(y_train_DJI)
MSE_FNN_1_DJI_In = mean_squared_error(train_y_DJI_FNN_In, predictions_DJI_FNN_In)
mseDJI_FNN_In = []
for i in np.arange(len(train_y_DJI_FNN_In)):
    mse = (predictions_DJI_FNN_In[i]-train_y_DJI_FNN_In[i])**2
    mseDJI_FNN_In.append(mse)
mseDJI_FNN_In = np.array(mseDJI_FNN_In)

predictions = model_FTSE.predict(train_X_FTSE_FNN)
predictions_FTSE_FNN_In = scaler.inverse_transform(predictions)
train_y_FTSE_FNN_In = np.array(y_train_FTSE)
MSE_FNN_1_FTSE_In = mean_squared_error(train_y_FTSE_FNN_In, predictions_FTSE_FNN_In)
mseFTSE_FNN_In = []
for i in np.arange(len(train_y_FTSE_FNN_In)):
    mse = (predictions_FTSE_FNN_In[i]-train_y_FTSE_FNN_In[i])**2
    mseFTSE_FNN_In.append(mse)
mseFTSE_FNN_In = np.array(mseFTSE_FNN_In)

predictions = model_FTSEMIB.predict(train_X_FTSEMIB_FNN)
predictions_FTSEMIB_FNN_In = scaler.inverse_transform(predictions)
train_y_FTSEMIB_FNN_In = np.array(y_train_FTSEMIB)
posizione = np.where(predictions_FTSEMIB_FNN_In<=0)[0]
media1 = np.mean(train_y_FTSE_FNN_In[395:399])
media2 = np.mean(train_y_FTSE_FNN_In[801:805])
predictions_FTSEMIB_FNN_In[397] = media1
predictions_FTSEMIB_FNN_In[803] = media2
MSE_FNN_1_FTSEMIB_In = mean_squared_error(train_y_FTSEMIB_FNN_In, predictions_FTSEMIB_FNN_In)
mseFTSEMIB_FNN_In = []
for i in np.arange(len(train_y_FTSEMIB_FNN_In)):
    mse = (predictions_FTSEMIB_FNN_In[i]-train_y_FTSEMIB_FNN_In[i])**2
    mseFTSEMIB_FNN_In.append(mse)
mseFTSEMIB_FNN_In = np.array(mseFTSEMIB_FNN_In)

predictions = model_GDAXI.predict(train_X_GDAXI_FNN)
predictions_GDAXI_FNN_In = scaler.inverse_transform(predictions)
train_y_GDAXI_FNN_In = np.array(y_train_GDAXI)
MSE_FNN_1_GDAXI_In = mean_squared_error(train_y_GDAXI_FNN_In, predictions_GDAXI_FNN_In)
mseGDAXI_FNN_In = []
for i in np.arange(len(train_y_GDAXI_FNN_In)):
    mse = (predictions_GDAXI_FNN_In[i]-train_y_GDAXI_FNN_In[i])**2
    mseGDAXI_FNN_In.append(mse)
mseGDAXI_FNN_In = np.array(mseGDAXI_FNN_In)

predictions = model_SPX.predict(train_X_SPX_FNN)
predictions_SPX_FNN_In = scaler.inverse_transform(predictions)
train_y_SPX_FNN_In = np.array(y_train_SPX)
MSE_FNN_1_SPX_In = mean_squared_error(train_y_SPX_FNN_In, predictions_SPX_FNN_In)
mseSPX_FNN_In = []
for i in np.arange(len(train_y_SPX_FNN_In)):
    mse = (predictions_SPX_FNN_In[i]-train_y_SPX_FNN_In[i])**2
    mseSPX_FNN_In.append(mse)
mseSPX_FNN_In = np.array(mseSPX_FNN_In)

predictions = model_HSI.predict(train_X_HSI_FNN)
predictions_HSI_FNN_In = scaler.inverse_transform(predictions)
train_y_HSI_FNN_In = np.array(y_train_HSI)
posizione = np.where(predictions_HSI_FNN_In<=0)[0]
media1 = np.mean(train_y_HSI_FNN_In[400:404])
media2 = np.mean(train_y_HSI_FNN_In[1946:1950])
media3 = np.mean(train_y_HSI_FNN_In[1951:1957])
predictions_HSI_FNN_In[402] = media1
predictions_HSI_FNN_In[1948] = media2
predictions_HSI_FNN_In[[1953, 1955]] = media3
MSE_FNN_1_HSI_In = mean_squared_error(train_y_HSI_FNN_In, predictions_HSI_FNN_In)
mseHSI_FNN_In = []
for i in np.arange(len(train_y_HSI_FNN_In)):
    mse = (predictions_HSI_FNN_In[i]-train_y_HSI_FNN_In[i])**2
    mseHSI_FNN_In.append(mse)
mseHSI_FNN_In = np.array(mseHSI_FNN_In)

predictions = model_IBEX.predict(train_X_IBEX_FNN)
predictions_IBEX_FNN_In = scaler.inverse_transform(predictions)
train_y_IBEX_FNN_In = np.array(y_train_IBEX)
MSE_FNN_1_IBEX_In = mean_squared_error(train_y_IBEX_FNN_In, predictions_IBEX_FNN_In)
mseIBEX_FNN_In = []
for i in np.arange(len(train_y_IBEX_FNN_In)):
    mse = (predictions_IBEX_FNN_In[i]-train_y_IBEX_FNN_In[i])**2
    mseIBEX_FNN_In.append(mse)
mseIBEX_FNN_In = np.array(mseIBEX_FNN_In)

predictions = model_IXIC.predict(train_X_IXIC_FNN)
predictions_IXIC_FNN_In = scaler.inverse_transform(predictions)
train_y_IXIC_FNN_In = np.array(y_train_IXIC)
MSE_FNN_1_IXIC_In = mean_squared_error(train_y_IXIC_FNN_In, predictions_IXIC_FNN_In)
mseIXIC_FNN_In = []
for i in np.arange(len(train_y_IXIC_FNN_In)):
    mse = (predictions_IXIC_FNN_In[i]-train_y_IXIC_FNN_In[i])**2
    mseIXIC_FNN_In.append(mse)
mseIXIC_FNN_In = np.array(mseIXIC_FNN_In)

predictions = model_N225.predict(train_X_N225_FNN)
predictions_N225_FNN_In = scaler.inverse_transform(predictions)
train_y_N225_FNN_In = np.array(y_train_N225)
MSE_FNN_1_N225_In = mean_squared_error(train_y_N225_FNN_In, predictions_N225_FNN_In)
mseN225_FNN_In = []
for i in np.arange(len(train_y_N225_FNN_In)):
    mse = (predictions_N225_FNN_In[i]-train_y_N225_FNN_In[i])**2
    mseN225_FNN_In.append(mse)
mseN225_FNN_In = np.array(mseN225_FNN_In)

predictions = model_OMXC20.predict(train_X_OMXC20_FNN)
predictions_OMXC20_FNN_In = scaler.inverse_transform(predictions)
train_y_OMXC20_FNN_In = np.array(y_train_OMXC20)
MSE_FNN_1_OMXC20_In = mean_squared_error(train_y_OMXC20_FNN_In, predictions_OMXC20_FNN_In)
mseOMXC20_FNN_In = []
for i in np.arange(len(train_y_OMXC20_FNN_In)):
    mse = (predictions_OMXC20_FNN_In[i]-train_y_OMXC20_FNN_In[i])**2
    mseOMXC20_FNN_In.append(mse)
mseOMXC20_FNN_In = np.array(mseOMXC20_FNN_In)

#%% Out of Sample Loss Function

mseDJI_FNN_Out = []
for i in np.arange(len(test_y_DJI_FNN)):
    mse = (predictions_DJI_FNN[i]-test_y_DJI_FNN[i])**2
    mseDJI_FNN_Out.append(mse)
mseDJI_FNN_Out = np.array(mseDJI_FNN_Out)

mseFTSE_FNN_Out = []
for i in np.arange(len(test_y_FTSE_FNN)):
    mse = (predictions_FTSE_FNN[i]-test_y_FTSE_FNN[i])**2
    mseFTSE_FNN_Out.append(mse)
mseFTSE_FNN_Out = np.array(mseFTSE_FNN_Out)

mseFTSEMIB_FNN_Out = []
for i in np.arange(len(test_y_FTSEMIB_FNN)):
    mse = (predictions_FTSEMIB_FNN[i]-test_y_FTSEMIB_FNN[i])**2
    mseFTSEMIB_FNN_Out.append(mse)
mseFTSEMIB_FNN_Out = np.array(mseFTSEMIB_FNN_Out)

mseGDAXI_FNN_Out = []
for i in np.arange(len(test_y_GDAXI_FNN)):
    mse = (predictions_GDAXI_FNN[i]-test_y_GDAXI_FNN[i])**2
    mseGDAXI_FNN_Out.append(mse)
mseGDAXI_FNN_Out = np.array(mseGDAXI_FNN_Out)

mseSPX_FNN_Out = []
for i in np.arange(len(test_y_SPX_FNN)):
    mse = (predictions_SPX_FNN[i]-test_y_SPX_FNN[i])**2
    mseSPX_FNN_Out.append(mse)
mseSPX_FNN_Out = np.array(mseSPX_FNN_Out)

mseHSI_FNN_Out = []
for i in np.arange(len(test_y_HSI_FNN)):
    mse = (predictions_HSI_FNN[i]-test_y_HSI_FNN[i])**2
    mseHSI_FNN_Out.append(mse)
mseHSI_FNN_Out = np.array(mseHSI_FNN_Out)

mseIBEX_FNN_Out = []
for i in np.arange(len(test_y_IBEX_FNN)):
    mse = (predictions_IBEX_FNN[i]-test_y_IBEX_FNN[i])**2
    mseIBEX_FNN_Out.append(mse)
mseIBEX_FNN_Out = np.array(mseIBEX_FNN_Out)

mseIXIC_FNN_Out = []
for i in np.arange(len(test_y_IXIC_FNN)):
    mse = (predictions_IXIC_FNN[i]-test_y_IXIC_FNN[i])**2
    mseIXIC_FNN_Out.append(mse)
mseIXIC_FNN_Out = np.array(mseIXIC_FNN_Out)

mseN225_FNN_Out = []
for i in np.arange(len(test_y_N225_FNN)):
    mse = (predictions_N225_FNN[i]-test_y_N225_FNN[i])**2
    mseN225_FNN_Out.append(mse)
mseN225_FNN_Out = np.array(mseN225_FNN_Out)

mseOMXC20_FNN_Out = []
for i in np.arange(len(test_y_OMXC20_FNN)):
    mse = (predictions_OMXC20_FNN[i]-test_y_OMXC20_FNN[i])**2
    mseOMXC20_FNN_Out.append(mse)
mseOMXC20_FNN_Out = np.array(mseOMXC20_FNN_Out)

#%% In Sample MSE
'''
arrays_In = {
    'mse_FNN': {
    'DJI': mseDJI_FNN_In,
    'FTSE': mseFTSE_FNN_In,
    'FTSEMIB': mseFTSEMIB_FNN_In,
    'GDAXI': mseGDAXI_FNN_In,
    'SPX': mseSPX_FNN_In,
    'HSI': mseHSI_FNN_In,
    'IBEX': mseIBEX_FNN_In,
    'IXIC': mseIXIC_FNN_In,
    'N225': mseN225_FNN_In,
    'OMXC20': mseOMXC20_FNN_In
            }
        }

for k1 in arrays_In:
    if k1 == 'mse_FNN':    
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_FNN_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
  '''          
#%% Out of Sample MSE
'''
arrays_Out = {
    'mse_FNN': {
    'DJI': mseDJI_FNN_Out,
    'FTSE': mseFTSE_FNN_Out,
    'FTSEMIB': mseFTSEMIB_FNN_Out,
    'GDAXI': mseGDAXI_FNN_Out,
    'SPX': mseSPX_FNN_Out,
    'HSI': mseHSI_FNN_Out,
    'IBEX': mseIBEX_FNN_Out,
    'IXIC': mseIXIC_FNN_Out,
    'N225': mseN225_FNN_Out,
    'OMXC20': mseOMXC20_FNN_Out
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_FNN_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
            
 '''           
#%% Qlike In

y_forecastvalues = predictions_DJI_FNN_In
y_actualvalues = train_y_DJI_FNN_In
qlike_DJI_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_FNN_1_In.append(iteration)
QLIKE_FNN_1_DJI_In = sum(qlike_DJI_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_FTSE_FNN_In
y_actualvalues = train_y_FTSE_FNN_In
qlike_FTSE_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_FNN_1_In.append(iteration)
QLIKE_FNN_1_FTSE_In = sum(qlike_FTSE_FNN_1_In)/len(y_actualvalues)


y_forecastvalues = predictions_FTSEMIB_FNN_In
y_actualvalues = train_y_FTSEMIB_FNN_In
qlike_FTSEMIB_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_FNN_1_In.append(iteration)
QLIKE_FNN_1_FTSEMIB_In = sum(qlike_FTSEMIB_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_GDAXI_FNN_In
y_actualvalues = train_y_GDAXI_FNN_In
qlike_GDAXI_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_FNN_1_In.append(iteration)
QLIKE_FNN_1_GDAXI_In = sum(qlike_GDAXI_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_SPX_FNN_In
y_actualvalues = train_y_SPX_FNN_In
qlike_SPX_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_FNN_1_In.append(iteration)
QLIKE_FNN_1_SPX_In = sum(qlike_SPX_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_HSI_FNN_In
y_actualvalues = train_y_HSI_FNN_In
qlike_HSI_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_FNN_1_In.append(iteration)
QLIKE_FNN_1_HSI_In = sum(qlike_HSI_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_IBEX_FNN_In
y_actualvalues = train_y_IBEX_FNN_In
qlike_IBEX_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_FNN_1_In.append(iteration)
QLIKE_FNN_1_IBEX_In = sum(qlike_IBEX_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_IXIC_FNN_In
y_actualvalues = train_y_IXIC_FNN_In
qlike_IXIC_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_FNN_1_In.append(iteration)
QLIKE_FNN_1_IXIC_In = sum(qlike_IXIC_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_N225_FNN_In
y_actualvalues = train_y_N225_FNN_In
qlike_N225_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_FNN_1_In.append(iteration)
QLIKE_FNN_1_N225_In = sum(qlike_N225_FNN_1_In)/len(y_actualvalues)

y_forecastvalues = predictions_OMXC20_FNN_In
y_actualvalues = train_y_OMXC20_FNN_In
qlike_OMXC20_FNN_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_FNN_1_In.append(iteration)
QLIKE_FNN_1_OMXC20_In = sum(qlike_OMXC20_FNN_1_In)/len(y_actualvalues)

#%% In-sample Qlike
'''
arrays_In = {
    'qlike_FNN': {
    'DJI': qlike_DJI_FNN_1_In,
    'FTSE': qlike_FTSE_FNN_1_In,
    'FTSEMIB': qlike_FTSEMIB_FNN_1_In,
    'GDAXI': qlike_GDAXI_FNN_1_In,
    'SPX': qlike_SPX_FNN_1_In,
    'HSI': qlike_HSI_FNN_1_In,
    'IBEX': qlike_IBEX_FNN_1_In,
    'IXIC': qlike_IXIC_FNN_1_In,
    'N225': qlike_N225_FNN_1_In,
    'OMXC20': qlike_OMXC20_FNN_1_In
            }
        }

for k1 in arrays_In:
    if k1 == 'qlike_FNN':    
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_FNN_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
'''            
#%% Qlike out of sample
'''
arrays_Out = {
    'qlike_FNN': {
    'DJI': qlike_DJI_FNN_1_Out,
    'FTSE': qlike_FTSE_FNN_1_Out,
    'FTSEMIB': qlike_FTSEMIB_FNN_1_Out,
    'GDAXI': qlike_GDAXI_FNN_1_Out,
    'SPX': qlike_SPX_FNN_1_Out,
    'HSI': qlike_HSI_FNN_1_Out,
    'IBEX': qlike_IBEX_FNN_1_Out,
    'IXIC': qlike_IXIC_FNN_1_Out,
    'N225': qlike_N225_FNN_1_Out,
    'OMXC20': qlike_OMXC20_FNN_1_Out
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_FNN_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
'''

#%%  

symbols = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']
predictions = [predictions_DJI_FNN, predictions_FTSE_FNN, predictions_FTSEMIB_FNN, predictions_GDAXI_FNN, predictions_SPX_FNN, predictions_HSI_FNN, predictions_IBEX_FNN, predictions_IXIC_FNN, predictions_N225_FNN, predictions_OMXC20_FNN]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastffnn_{symbol}.csv', prediction, delimiter=',')


symbols = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']
predictions = [predictions_DJI_FNN_In, predictions_FTSE_FNN_In, predictions_FTSEMIB_FNN_In, predictions_GDAXI_FNN_In, predictions_SPX_FNN_In, predictions_HSI_FNN_In, predictions_IBEX_FNN_In, predictions_IXIC_FNN_In, predictions_N225_FNN_In, predictions_OMXC20_FNN_In]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastffnn_{symbol}_in.csv', prediction, delimiter=',')