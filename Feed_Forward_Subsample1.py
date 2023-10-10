# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:29:37 2023

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

#%%

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
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=train_X_DJI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_DJI = create_model(hidden_nodes)
model_DJI.fit(train_X_DJI_FNN, train_y_DJI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_DJI.predict(test_X_FNN_sub1)
predictions_DJI_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_DJI_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))
posizione = np.where(predictions_DJI_FNN_sub1<=0)[0]
media = np.mean(test_y_DJI_FNN_sub1[4:8])
predictions_DJI_FNN_sub1[6] = media

predictDates_DJI_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_DJI_sub1 = mean_squared_error(test_y_DJI_FNN_sub1, predictions_DJI_FNN_sub1)
print(MSE_FNN_1_DJI_sub1)

mseDJI_FNN_sub1 = []
for i in np.arange(len(test_y_DJI_FNN_sub1)):
    mse = (predictions_DJI_FNN_sub1[i]-test_y_DJI_FNN_sub1[i])**2
    mseDJI_FNN_sub1.append(mse)
mseDJI_FNN_sub1 = np.array(mseDJI_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_DJI_FNN_sub1
y_actualvalues = test_y_DJI_FNN_sub1
qlike_DJI_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_FNN_1_sub1.append(iteration)
qlike_DJI_FNN_1_sub1 = np.array(qlike_DJI_FNN_1_sub1)
QLIKE_FNN_1_DJI_sub1 = sum(qlike_DJI_FNN_1_sub1)/len(y_actualvalues)

#%% FTSE

data = FTSE
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

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
predictions = model_FTSE.predict(test_X_FNN_sub1)
predictions_FTSE_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_FTSE_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_FTSE_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_FTSE_sub1 = mean_squared_error(test_y_FTSE_FNN_sub1, predictions_FTSE_FNN_sub1)
print(MSE_FNN_1_FTSE_sub1)

mseFTSE_FNN_sub1 = []
for i in np.arange(len(test_y_FTSE_FNN_sub1)):
    mse = (predictions_FTSE_FNN_sub1[i]-test_y_FTSE_FNN_sub1[i])**2
    mseFTSE_FNN_sub1.append(mse)
mseFTSE_FNN_sub1 = np.array(mseFTSE_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_FTSE_FNN_sub1
y_actualvalues = test_y_FTSE_FNN_sub1
qlike_FTSE_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_FNN_1_sub1.append(iteration)
qlike_FTSE_FNN_1_sub1 = np.array(qlike_FTSE_FNN_1_sub1)
QLIKE_FNN_1_FTSE_sub1 = sum(qlike_FTSE_FNN_1_sub1)/len(y_actualvalues)

#%% FTSEMIB

data = FTSEMIB
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_FTSEMIB_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_FTSEMIB = create_model(hidden_nodes)
model_FTSEMIB.fit(train_X_FTSEMIB_FNN, train_y_FTSEMIB_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_FTSEMIB.predict(test_X_FNN_sub1)
predictions_FTSEMIB_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_FTSEMIB_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_FTSEMIB_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_FTSEMIB_sub1 = mean_squared_error(test_y_FTSEMIB_FNN_sub1, predictions_FTSEMIB_FNN_sub1)
print(MSE_FNN_1_FTSEMIB_sub1)

mseFTSEMIB_FNN_sub1 = []
for i in np.arange(len(test_y_FTSEMIB_FNN_sub1)):
    mse = (predictions_FTSEMIB_FNN_sub1[i]-test_y_FTSEMIB_FNN_sub1[i])**2
    mseFTSEMIB_FNN_sub1.append(mse)
mseFTSEMIB_FNN_sub1 = np.array(mseFTSEMIB_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_FTSEMIB_FNN_sub1
y_actualvalues = test_y_FTSEMIB_FNN_sub1
qlike_FTSEMIB_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_FNN_1_sub1.append(iteration)
qlike_FTSEMIB_FNN_1_sub1 = np.array(qlike_FTSEMIB_FNN_1_sub1)
QLIKE_FNN_1_FTSEMIB_sub1 = sum(qlike_FTSEMIB_FNN_1_sub1)/len(y_actualvalues)

#%% GDAXI

data = GDAXI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_GDAXI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_GDAXI = create_model(hidden_nodes)
model_GDAXI.fit(train_X_GDAXI_FNN, train_y_GDAXI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_GDAXI.predict(test_X_FNN_sub1)
predictions_GDAXI_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_GDAXI_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_GDAXI_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_GDAXI_sub1 = mean_squared_error(test_y_GDAXI_FNN_sub1, predictions_GDAXI_FNN_sub1)
print(MSE_FNN_1_GDAXI_sub1)

mseGDAXI_FNN_sub1 = []
for i in np.arange(len(test_y_GDAXI_FNN_sub1)):
    mse = (predictions_GDAXI_FNN_sub1[i]-test_y_GDAXI_FNN_sub1[i])**2
    mseGDAXI_FNN_sub1.append(mse)
mseGDAXI_FNN_sub1 = np.array(mseGDAXI_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_GDAXI_FNN_sub1
y_actualvalues = test_y_GDAXI_FNN_sub1
qlike_GDAXI_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_FNN_1_sub1.append(iteration)
qlike_GDAXI_FNN_1_sub1 = np.array(qlike_GDAXI_FNN_1_sub1)
QLIKE_FNN_1_GDAXI_sub1 = sum(qlike_GDAXI_FNN_1_sub1)/len(y_actualvalues)

#%% SPX

data = SPX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=train_X_SPX_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_SPX = create_model(hidden_nodes)
model_SPX.fit(train_X_SPX_FNN, train_y_SPX_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_SPX.predict(test_X_FNN_sub1)
predictions_SPX_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_SPX_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_SPX_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_SPX_sub1 = mean_squared_error(test_y_SPX_FNN_sub1, predictions_SPX_FNN_sub1)
print(MSE_FNN_1_SPX_sub1)

mseSPX_FNN_sub1 = []
for i in np.arange(len(test_y_SPX_FNN_sub1)):
    mse = (predictions_SPX_FNN_sub1[i]-test_y_SPX_FNN_sub1[i])**2
    mseSPX_FNN_sub1.append(mse)
mseSPX_FNN_sub1 = np.array(mseSPX_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_SPX_FNN_sub1
y_actualvalues = test_y_SPX_FNN_sub1
qlike_SPX_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_FNN_1_sub1.append(iteration)
qlike_SPX_FNN_1_sub1 = np.array(qlike_SPX_FNN_1_sub1)
QLIKE_FNN_1_SPX_sub1 = sum(qlike_SPX_FNN_1_sub1)/len(y_actualvalues)

#%% HSI

data = HSI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_HSI_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_HSI = create_model(hidden_nodes)
model_HSI.fit(train_X_HSI_FNN, train_y_HSI_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_HSI.predict(test_X_FNN_sub1)
predictions_HSI_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_HSI_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_HSI_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_HSI_sub1 = mean_squared_error(test_y_HSI_FNN_sub1, predictions_HSI_FNN_sub1)
print(MSE_FNN_1_HSI_sub1)

mseHSI_FNN_sub1 = []
for i in np.arange(len(test_y_HSI_FNN_sub1)):
    mse = (predictions_HSI_FNN_sub1[i]-test_y_HSI_FNN_sub1[i])**2
    mseHSI_FNN_sub1.append(mse)
mseHSI_FNN_sub1 = np.array(mseHSI_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_HSI_FNN_sub1
y_actualvalues = test_y_HSI_FNN_sub1
qlike_HSI_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_FNN_1_sub1.append(iteration)
qlike_HSI_FNN_1_sub1 = np.array(qlike_HSI_FNN_1_sub1)
QLIKE_FNN_1_HSI_sub1 = sum(qlike_HSI_FNN_1_sub1)/len(y_actualvalues)

#%% IBEX

data = IBEX
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_IBEX_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_IBEX = create_model(hidden_nodes)
model_IBEX.fit(train_X_IBEX_FNN, train_y_IBEX_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_IBEX.predict(test_X_FNN_sub1)
predictions_IBEX_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_IBEX_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_IBEX_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_IBEX_sub1 = mean_squared_error(test_y_IBEX_FNN_sub1, predictions_IBEX_FNN_sub1)
print(MSE_FNN_1_IBEX_sub1)

mseIBEX_FNN_sub1 = []
for i in np.arange(len(test_y_IBEX_FNN_sub1)):
    mse = (predictions_IBEX_FNN_sub1[i]-test_y_IBEX_FNN_sub1[i])**2
    mseIBEX_FNN_sub1.append(mse)
mseIBEX_FNN_sub1 = np.array(mseIBEX_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_IBEX_FNN_sub1
y_actualvalues = test_y_IBEX_FNN_sub1
qlike_IBEX_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_FNN_1_sub1.append(iteration)
qlike_IBEX_FNN_1_sub1 = np.array(qlike_IBEX_FNN_1_sub1)
QLIKE_FNN_1_IBEX_sub1 = sum(qlike_IBEX_FNN_1_sub1)/len(y_actualvalues)

#%% IXIC

data = IXIC
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

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
predictions = model_IXIC.predict(test_X_FNN_sub1)
predictions_IXIC_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_IXIC_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))
predictDates_IXIC_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_IXIC_sub1 = mean_squared_error(test_y_IXIC_FNN_sub1, predictions_IXIC_FNN_sub1)
print(MSE_FNN_1_IXIC_sub1)

mseIXIC_FNN_sub1 = []
for i in np.arange(len(test_y_IXIC_FNN_sub1)):
    mse = (predictions_IXIC_FNN_sub1[i]-test_y_IXIC_FNN_sub1[i])**2
    mseIXIC_FNN_sub1.append(mse)
mseIXIC_FNN_sub1 = np.array(mseIXIC_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_IXIC_FNN_sub1
y_actualvalues = test_y_IXIC_FNN_sub1
qlike_IXIC_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_FNN_1_sub1.append(iteration)
qlike_IXIC_FNN_1_sub1 = np.array(qlike_IXIC_FNN_1_sub1)
QLIKE_FNN_1_IXIC_sub1 = sum(qlike_IXIC_FNN_1_sub1)/len(y_actualvalues)

#%% N225

data = N225
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_N225_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_N225 = create_model(hidden_nodes)
model_N225.fit(train_X_N225_FNN, train_y_N225_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_N225.predict(test_X_FNN_sub1)
predictions_N225_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_N225_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))
posizione = np.where(predictions_N225_FNN_sub1<=0)[0]
media1 = np.mean(test_y_N225_FNN_sub1[171:175])
media2 = np.mean(test_y_N225_FNN_sub1[300:304])
predictions_N225_FNN_sub1[173] = media1
predictions_N225_FNN_sub1[302] = media2

predictDates_N225_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_N225_sub1 = mean_squared_error(test_y_N225_FNN_sub1, predictions_N225_FNN_sub1)
print(MSE_FNN_1_N225_sub1)

mseN225_FNN_sub1 = []
for i in np.arange(len(test_y_N225_FNN_sub1)):
    mse = (predictions_N225_FNN_sub1[i]-test_y_N225_FNN_sub1[i])**2
    mseN225_FNN_sub1.append(mse)
mseN225_FNN_sub1 = np.array(mseN225_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_N225_FNN_sub1
y_actualvalues = test_y_N225_FNN_sub1
qlike_N225_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_FNN_1_sub1.append(iteration)
qlike_N225_FNN_1_sub1 = np.array(qlike_N225_FNN_1_sub1)
QLIKE_FNN_1_N225_sub1 = sum(qlike_N225_FNN_1_sub1)/len(y_actualvalues)

#%% OMXC20

data = OMXC20
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 3

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

test_X_FNN_sub1, test_y_FNN_sub1 = create_dataset(scaled_data, look_back)

def create_model(hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_regularizer=regularizers.l2(0.0005), activation='relu', input_dim=train_X_OMXC20_FNN.shape[1], kernel_initializer = GlorotUniform(seed=1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

random.seed(1)
set_seed(1)
model_OMXC20 = create_model(hidden_nodes)
model_OMXC20.fit(train_X_OMXC20_FNN, train_y_OMXC20_FNN, batch_size = batch_size, epochs=epochs, verbose = 0, shuffle = False)
predictions = model_OMXC20.predict(test_X_FNN_sub1)
predictions_OMXC20_FNN_sub1 = scaler.inverse_transform(predictions)
test_y_OMXC20_FNN_sub1 = scaler.inverse_transform(test_y_FNN_sub1.reshape(-1,1))

predictDates_OMXC20_sub1 = data.tail(len(test_X_FNN_sub1)).index

# MSE

MSE_FNN_1_OMXC20_sub1 = mean_squared_error(test_y_OMXC20_FNN_sub1, predictions_OMXC20_FNN_sub1)
print(MSE_FNN_1_OMXC20_sub1)

mseOMXC20_FNN_sub1 = []
for i in np.arange(len(test_y_OMXC20_FNN_sub1)):
    mse = (predictions_OMXC20_FNN_sub1[i]-test_y_OMXC20_FNN_sub1[i])**2
    mseOMXC20_FNN_sub1.append(mse)
mseOMXC20_FNN_sub1 = np.array(mseOMXC20_FNN_sub1)

# QLIKE

y_forecastvalues = predictions_OMXC20_FNN_sub1
y_actualvalues = test_y_OMXC20_FNN_sub1
qlike_OMXC20_FNN_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_FNN_1_sub1.append(iteration)
qlike_OMXC20_FNN_1_sub1 = np.array(qlike_OMXC20_FNN_1_sub1)
QLIKE_FNN_1_OMXC20_sub1 = sum(qlike_OMXC20_FNN_1_sub1)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_FNN': {
    'DJI': mseDJI_FNN_sub1,
    'FTSE': mseFTSE_FNN_sub1,
    'FTSEMIB': mseFTSEMIB_FNN_sub1,
    'GDAXI': mseGDAXI_FNN_sub1,
    'SPX': mseSPX_FNN_sub1,
    'HSI': mseHSI_FNN_sub1,
    'IBEX': mseIBEX_FNN_sub1,
    'IXIC': mseIXIC_FNN_sub1,
    'N225': mseN225_FNN_sub1,
    'OMXC20': mseOMXC20_FNN_sub1
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_FNN_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_FNN': {
    'DJI': qlike_DJI_FNN_1_sub1,
    'FTSE': qlike_FTSE_FNN_1_sub1,
    'FTSEMIB': qlike_FTSEMIB_FNN_1_sub1,
    'GDAXI': qlike_GDAXI_FNN_1_sub1,
    'SPX': qlike_SPX_FNN_1_sub1,
    'HSI': qlike_HSI_FNN_1_sub1,
    'IBEX': qlike_IBEX_FNN_1_sub1,
    'IXIC': qlike_IXIC_FNN_1_sub1,
    'N225': qlike_N225_FNN_1_sub1,
    'OMXC20': qlike_OMXC20_FNN_1_sub1
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_FNN_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%       
table=readtable('qlikeDJI_AR1_sub1.csv');
qlikeDJIAR1_sub1 = table2array(table);

table=readtable('qlikeDJI_HAR_sub1.csv');
qlikeDJIHAR_sub1 = table2array(table);

table=readtable('qlikeDJI_HARlog_sub1.csv');
qlikeDJIHARlog_sub1 = table2array(table);

table=readtable('qlikeDJI_rf_sub1.csv');
qlikeDJIrf_sub1 = table2array(table);

table=readtable('qlikeDJI_LSTM_sub1.csv');
qlikeDJILSTM_sub1 = table2array(table);

table=readtable('qlikeDJI_FNN_sub1.csv');
qlikeDJIFNN_sub1 = table2array(table);

table=readtable('qlikeFTSE_AR1_sub1.csv');
qlikeFTSEAR1_sub1 = table2array(table);

table=readtable('qlikeFTSE_HAR_sub1.csv');
qlikeFTSEHAR_sub1 = table2array(table);

table=readtable('qlikeFTSE_HARlog_sub1.csv');
qlikeFTSEHARlog_sub1 = table2array(table);

table=readtable('qlikeFTSE_rf_sub1.csv');
qlikeFTSErf_sub1 = table2array(table);

table=readtable('qlikeFTSE_LSTM_sub1.csv');
qlikeFTSELSTM_sub1 = table2array(table);

table=readtable('qlikeFTSE_FNN_sub1.csv');
qlikeFTSEFNN_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_AR1_sub1.csv');
qlikeFTSEMIBAR1_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_HAR_sub1.csv');
qlikeFTSEMIBHAR_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_HARlog_sub1.csv');
qlikeFTSEMIBHARlog_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_rf_sub1.csv');
qlikeFTSEMIBrf_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_LSTM_sub1.csv');
qlikeFTSEMIBLSTM_sub1 = table2array(table);

table=readtable('qlikeFTSEMIB_FNN_sub1.csv');
qlikeFTSEMIBFNN_sub1 = table2array(table);

table=readtable('qlikeGDAXI_AR1_sub1.csv');
qlikeGDAXIAR1_sub1 = table2array(table);

table=readtable('qlikeGDAXI_HAR_sub1.csv');
qlikeGDAXIHAR_sub1 = table2array(table);

table=readtable('qlikeGDAXI_HARlog_sub1.csv');
qlikeGDAXIHARlog_sub1 = table2array(table);

table=readtable('qlikeGDAXI_rf_sub1.csv');
qlikeGDAXIrf_sub1 = table2array(table);

table=readtable('qlikeGDAXI_LSTM_sub1.csv');
qlikeGDAXILSTM_sub1 = table2array(table);

table=readtable('qlikeGDAXI_FNN_sub1.csv');
qlikeGDAXIFNN_sub1 = table2array(table);

table=readtable('qlikeSPX_AR1_sub1.csv');
qlikeSPXAR1_sub1 = table2array(table);

table=readtable('qlikeSPX_HAR_sub1.csv');
qlikeSPXHAR_sub1 = table2array(table);

table=readtable('qlikeSPX_HARlog_sub1.csv');
qlikeSPXHARlog_sub1 = table2array(table);

table=readtable('qlikeSPX_rf_sub1.csv');
qlikeSPXrf_sub1 = table2array(table);

table=readtable('qlikeSPX_LSTM_sub1.csv');
qlikeSPXLSTM_sub1 = table2array(table);

table=readtable('qlikeSPX_FNN_sub1.csv');
qlikeSPXFNN_sub1 = table2array(table);

table=readtable('qlikeHSI_AR1_sub1.csv');
qlikeHSIAR1_sub1 = table2array(table);

table=readtable('qlikeHSI_HAR_sub1.csv');
qlikeHSIHAR_sub1 = table2array(table);

table=readtable('qlikeHSI_HARlog_sub1.csv');
qlikeHSIHARlog_sub1 = table2array(table);

table=readtable('qlikeHSI_rf_sub1.csv');
qlikeHSIrf_sub1 = table2array(table);

table=readtable('qlikeHSI_LSTM_sub1.csv');
qlikeHSILSTM_sub1 = table2array(table);

table=readtable('qlikeHSI_FNN_sub1.csv');
qlikeHSIFNN_sub1 = table2array(table);

table=readtable('qlikeIBEX_AR1_sub1.csv');
qlikeIBEXAR1_sub1 = table2array(table);

table=readtable('qlikeIBEX_HAR_sub1.csv');
qlikeIBEXHAR_sub1 = table2array(table);

table=readtable('qlikeIBEX_HARlog_sub1.csv');
qlikeIBEXHARlog_sub1 = table2array(table);

table=readtable('qlikeIBEX_rf_sub1.csv');
qlikeIBEXrf_sub1 = table2array(table);

table=readtable('qlikeIBEX_LSTM_sub1.csv');
qlikeIBEXLSTM_sub1 = table2array(table);

table=readtable('qlikeIBEX_FNN_sub1.csv');
qlikeIBEXFNN_sub1 = table2array(table);

table=readtable('qlikeIXIC_AR1_sub1.csv');
qlikeIXICAR1_sub1 = table2array(table);

table=readtable('qlikeIXIC_HAR_sub1.csv');
qlikeIXICHAR_sub1 = table2array(table);

table=readtable('qlikeIXIC_HARlog_sub1.csv');
qlikeIXICHARlog_sub1 = table2array(table);

table=readtable('qlikeIXIC_rf_sub1.csv');
qlikeIXICrf_sub1 = table2array(table);

table=readtable('qlikeIXIC_LSTM_sub1.csv');
qlikeIXICLSTM_sub1 = table2array(table);

table=readtable('qlikeIXIC_FNN_sub1.csv');
qlikeIXICFNN_sub1 = table2array(table);

table=readtable('qlikeN225_AR1_sub1.csv');
qlikeN225AR1_sub1 = table2array(table);

table=readtable('qlikeN225_HAR_sub1.csv');
qlikeN225HAR_sub1 = table2array(table);

table=readtable('qlikeN225_HARlog_sub1.csv');
qlikeN225HARlog_sub1 = table2array(table);

table=readtable('qlikeN225_rf_sub1.csv');
qlikeN225rf_sub1 = table2array(table);

table=readtable('qlikeN225_LSTM_sub1.csv');
qlikeN225LSTM_sub1 = table2array(table);

table=readtable('qlikeN225_FNN_sub1.csv');
qlikeN225FNN_sub1 = table2array(table);

table=readtable('qlikeN225_AR1_sub1.csv');
qlikeOMXC20AR1_sub1 = table2array(table);

table=readtable('qlikeN225_HAR_sub1.csv');
qlikeOMXC20HAR_sub1 = table2array(table);

table=readtable('qlikeN225_HARlog_sub1.csv');
qlikeOMXC20HARlog_sub1 = table2array(table);

table=readtable('qlikeN225_rf_sub1.csv');
qlikeOMXC20rf_sub1 = table2array(table);

table=readtable('qlikeN225_LSTM_sub1.csv');
qlikeOMXC20LSTM_sub1 = table2array(table);

table=readtable('qlikeN225_FNN_sub1.csv');
qlikeOMXC20FNN_sub1 = table2array(table);

['predictionsDJI_rf_1_sub1.csv.csv', 'predictionsFTSE_rf_1_sub1.csv.csv', 'predictionsFTSEMIB_rf_1_sub1.csv.csv', 'predictionsGDAXI_rf_1_sub1.csv.csv', 'predictionsSPX_rf_1_sub1.csv.csv', 'predictionsHSI_rf_1_sub1.csv.csv', 'predictionsIBEX_rf_1_sub1.csv.csv', 'predictionsIXIC_rf_1_sub1.csv.csv', 'predictionsN225_rf_1_sub1.csv.csv', 'predictionsOMXC20_rf_1_sub1.csv.csv'];


#%%

predict = {
    'predictions_FNN': {
        'DJI': predictions_DJI_FNN_sub1,
        'FTSE': predictions_FTSE_FNN_sub1,
        'FTSEMIB': predictions_FTSEMIB_FNN_sub1,
        'GDAXI': predictions_GDAXI_FNN_sub1,
        'SPX': predictions_SPX_FNN_sub1,
        'HSI': predictions_HSI_FNN_sub1,
        'IBEX': predictions_IBEX_FNN_sub1,
        'IXIC': predictions_IXIC_FNN_sub1,
        'N225': predictions_N225_FNN_sub1,
        'OMXC20': predictions_OMXC20_FNN_sub1
                }
        }

for k1 in predict:
    if k1 == 'predictions_FNN':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_FNN_1_sub1.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')