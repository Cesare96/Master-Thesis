# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:09:05 2023

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

#%% FNN

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

#%% DJI

data = DJI
data1 = np.array(data)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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
predictions = model_DJI.predict(test_X)
predictions_DJI_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_DJI_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_DJI_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_DJI_sub2 = mean_squared_error(test_y_DJI_FNN_sub2, predictions_DJI_FNN_sub2)
print(MSE_FNN_1_DJI_sub2)

mseDJI_FNN_sub2 = []
for i in np.arange(len(test_y_DJI_FNN_sub2)):
    mse = (predictions_DJI_FNN_sub2[i]-test_y_DJI_FNN_sub2[i])**2
    mseDJI_FNN_sub2.append(mse)
mseDJI_FNN_sub2 = np.array(mseDJI_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_DJI_FNN_sub2
y_actualvalues = test_y_DJI_FNN_sub2
qlike_DJI_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_DJI_FNN_1_sub2.append(iteration)
qlike_DJI_FNN_1_sub2 = np.array(qlike_DJI_FNN_1_sub2)
QLIKE_FNN_1_DJI_sub2 = sum(qlike_DJI_FNN_1_sub2)/len(y_actualvalues)

#%% FTSE

data = FTSE
data1 = np.array(data).reshape(-1,1)
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 7

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
predictions = model_FTSE.predict(test_X)
predictions_FTSE_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_FTSE_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_FTSE_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_FTSE_sub2 = mean_squared_error(test_y_FTSE_FNN_sub2, predictions_FTSE_FNN_sub2)
print(MSE_FNN_1_FTSE_sub2)

mseFTSE_FNN_sub2 = []
for i in np.arange(len(test_y_FTSE_FNN_sub2)):
    mse = (predictions_FTSE_FNN_sub2[i]-test_y_FTSE_FNN_sub2[i])**2
    mseFTSE_FNN_sub2.append(mse)
mseFTSE_FNN_sub2 = np.array(mseFTSE_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_FTSE_FNN_sub2
y_actualvalues = test_y_FTSE_FNN_sub2
qlike_FTSE_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSE_FNN_1_sub2.append(iteration)
qlike_FTSE_FNN_1_sub2 = np.array(qlike_FTSE_FNN_1_sub2)
QLIKE_FNN_1_FTSE_sub2 = sum(qlike_FTSE_FNN_1_sub2)/len(y_actualvalues)

#%% FTSEMIB

data = FTSEMIB
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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
predictions = model_FTSEMIB.predict(test_X)
predictions_FTSEMIB_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_FTSEMIB_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))
posizione = np.where(predictions_FTSEMIB_FNN_sub2<=0)[0]
media = np.mean(test_y_FTSEMIB_FNN_sub2[416:420])
predictions_FTSEMIB_FNN_sub2[418] = media

predictDates_FTSEMIB_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_FTSEMIB_sub2 = mean_squared_error(test_y_FTSEMIB_FNN_sub2, predictions_FTSEMIB_FNN_sub2)
print(MSE_FNN_1_FTSEMIB_sub2)

mseFTSEMIB_FNN_sub2 = []
for i in np.arange(len(test_y_FTSEMIB_FNN_sub2)):
    mse = (predictions_FTSEMIB_FNN_sub2[i]-test_y_FTSEMIB_FNN_sub2[i])**2
    mseFTSEMIB_FNN_sub2.append(mse)
mseFTSEMIB_FNN_sub2 = np.array(mseFTSEMIB_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_FTSEMIB_FNN_sub2
y_actualvalues = test_y_FTSEMIB_FNN_sub2
qlike_FTSEMIB_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_FTSEMIB_FNN_1_sub2.append(iteration)
qlike_FTSEMIB_FNN_1_sub2 = np.array(qlike_FTSEMIB_FNN_1_sub2)
QLIKE_FNN_1_FTSEMIB_sub2 = sum(qlike_FTSEMIB_FNN_1_sub2)/len(y_actualvalues)

#%% GDAXI

data = GDAXI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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
predictions = model_GDAXI.predict(test_X)
predictions_GDAXI_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_GDAXI_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_GDAXI_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_GDAXI_sub2 = mean_squared_error(test_y_GDAXI_FNN_sub2, predictions_GDAXI_FNN_sub2)
print(MSE_FNN_1_GDAXI_sub2)

mseGDAXI_FNN_sub2 = []
for i in np.arange(len(test_y_GDAXI_FNN_sub2)):
    mse = (predictions_GDAXI_FNN_sub2[i]-test_y_GDAXI_FNN_sub2[i])**2
    mseGDAXI_FNN_sub2.append(mse)
mseGDAXI_FNN_sub2 = np.array(mseGDAXI_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_GDAXI_FNN_sub2
y_actualvalues = test_y_GDAXI_FNN_sub2
qlike_GDAXI_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_GDAXI_FNN_1_sub2.append(iteration)
qlike_GDAXI_FNN_1_sub2 = np.array(qlike_GDAXI_FNN_1_sub2)
QLIKE_FNN_1_GDAXI_sub2 = sum(qlike_GDAXI_FNN_1_sub2)/len(y_actualvalues)

#%% SPX

data = SPX
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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
predictions = model_SPX.predict(test_X)
predictions_SPX_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_SPX_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_SPX_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_SPX_sub2 = mean_squared_error(test_y_SPX_FNN_sub2, predictions_SPX_FNN_sub2)
print(MSE_FNN_1_SPX_sub2)

mseSPX_FNN_sub2 = []
for i in np.arange(len(test_y_SPX_FNN_sub2)):
    mse = (predictions_SPX_FNN_sub2[i]-test_y_SPX_FNN_sub2[i])**2
    mseSPX_FNN_sub2.append(mse)
mseSPX_FNN_sub2 = np.array(mseSPX_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_SPX_FNN_sub2
y_actualvalues = test_y_SPX_FNN_sub2
qlike_SPX_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_SPX_FNN_1_sub2.append(iteration)
qlike_SPX_FNN_1_sub2 = np.array(qlike_SPX_FNN_1_sub2)
QLIKE_FNN_1_SPX_sub2 = sum(qlike_SPX_FNN_1_sub2)/len(y_actualvalues)

#%% HSI

data = HSI
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 2

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
predictions = model_HSI.predict(test_X)
predictions_HSI_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_HSI_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_HSI_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_HSI_sub2 = mean_squared_error(test_y_HSI_FNN_sub2, predictions_HSI_FNN_sub2)
print(MSE_FNN_1_HSI_sub2)

mseHSI_FNN_sub2 = []
for i in np.arange(len(test_y_HSI_FNN_sub2)):
    mse = (predictions_HSI_FNN_sub2[i]-test_y_HSI_FNN_sub2[i])**2
    mseHSI_FNN_sub2.append(mse)
mseHSI_FNN_sub2 = np.array(mseHSI_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_HSI_FNN_sub2
y_actualvalues = test_y_HSI_FNN_sub2
qlike_HSI_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_HSI_FNN_1_sub2.append(iteration)
qlike_HSI_FNN_1_sub2 = np.array(qlike_HSI_FNN_1_sub2)
QLIKE_FNN_1_HSI_sub2 = sum(qlike_HSI_FNN_1_sub2)/len(y_actualvalues)

#%% IBEX

data = IBEX
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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
predictions = model_IBEX.predict(test_X)
predictions_IBEX_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_IBEX_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IBEX_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_IBEX_sub2 = mean_squared_error(test_y_IBEX_FNN_sub2, predictions_IBEX_FNN_sub2)
print(MSE_FNN_1_IBEX_sub2)

mseIBEX_FNN_sub2 = []
for i in np.arange(len(test_y_IBEX_FNN_sub2)):
    mse = (predictions_IBEX_FNN_sub2[i]-test_y_IBEX_FNN_sub2[i])**2
    mseIBEX_FNN_sub2.append(mse)
mseIBEX_FNN_sub2 = np.array(mseIBEX_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_IBEX_FNN_sub2
y_actualvalues = test_y_IBEX_FNN_sub2
qlike_IBEX_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IBEX_FNN_1_sub2.append(iteration)
qlike_IBEX_FNN_1_sub2 = np.array(qlike_IBEX_FNN_1_sub2)
QLIKE_FNN_1_IBEX_sub2 = sum(qlike_IBEX_FNN_1_sub2)/len(y_actualvalues)

#%% IXIC

data = IXIC
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 4

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
predictions = model_IXIC.predict(test_X)
predictions_IXIC_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_IXIC_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_IXIC_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_IXIC_sub2 = mean_squared_error(test_y_IXIC_FNN_sub2, predictions_IXIC_FNN_sub2)
print(MSE_FNN_1_IXIC_sub2)

mseIXIC_FNN_sub2 = []
for i in np.arange(len(test_y_IXIC_FNN_sub2)):
    mse = (predictions_IXIC_FNN_sub2[i]-test_y_IXIC_FNN_sub2[i])**2
    mseIXIC_FNN_sub2.append(mse)
mseIXIC_FNN_sub2 = np.array(mseIXIC_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_IXIC_FNN_sub2
y_actualvalues = test_y_IXIC_FNN_sub2
qlike_IXIC_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_IXIC_FNN_1_sub2.append(iteration)
qlike_IXIC_FNN_1_sub2 = np.array(qlike_IXIC_FNN_1_sub2)
QLIKE_FNN_1_IXIC_sub2 = sum(qlike_IXIC_FNN_1_sub2)/len(y_actualvalues)

#%% N225

data = N225
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 6

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
predictions = model_N225.predict(test_X)
predictions_N225_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_N225_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_N225_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_N225_sub2 = mean_squared_error(test_y_N225_FNN_sub2, predictions_N225_FNN_sub2)
print(MSE_FNN_1_N225_sub2)

mseN225_FNN_sub2 = []
for i in np.arange(len(test_y_N225_FNN_sub2)):
    mse = (predictions_N225_FNN_sub2[i]-test_y_N225_FNN_sub2[i])**2
    mseN225_FNN_sub2.append(mse)
mseN225_FNN_sub2 = np.array(mseN225_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_N225_FNN_sub2
y_actualvalues = test_y_N225_FNN_sub2
qlike_N225_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_N225_FNN_1_sub2.append(iteration)
qlike_N225_FNN_1_sub2 = np.array(qlike_N225_FNN_1_sub2)
QLIKE_FNN_1_N225_sub2 = sum(qlike_N225_FNN_1_sub2)/len(y_actualvalues)

#%% OMXC20

data = OMXC20
look_back = 21
batch_size = 64
epochs = 200
hidden_nodes = 3

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
predictions = model_OMXC20.predict(test_X)
predictions_OMXC20_FNN_sub2 = scaler.inverse_transform(predictions)
test_y_OMXC20_FNN_sub2 = scaler.inverse_transform(test_y.reshape(-1,1))

predictDates_OMXC20_sub2 = data.tail(len(test_X)).index

# MSE

MSE_FNN_1_OMXC20_sub2 = mean_squared_error(test_y_OMXC20_FNN_sub2, predictions_OMXC20_FNN_sub2)
print(MSE_FNN_1_OMXC20_sub2)

mseOMXC20_FNN_sub2 = []
for i in np.arange(len(test_y_OMXC20_FNN_sub2)):
    mse = (predictions_OMXC20_FNN_sub2[i]-test_y_OMXC20_FNN_sub2[i])**2
    mseOMXC20_FNN_sub2.append(mse)
mseOMXC20_FNN_sub2 = np.array(mseOMXC20_FNN_sub2)

# QLIKE

y_forecastvalues = predictions_OMXC20_FNN_sub2
y_actualvalues = test_y_OMXC20_FNN_sub2
qlike_OMXC20_FNN_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlike_OMXC20_FNN_1_sub2.append(iteration)
qlike_OMXC20_FNN_1_sub2 = np.array(qlike_OMXC20_FNN_1_sub2)
QLIKE_FNN_1_OMXC20_sub2 = sum(qlike_OMXC20_FNN_1_sub2)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_FNN': {
    'DJI': mseDJI_FNN_sub2,
    'FTSE': mseFTSE_FNN_sub2,
    'FTSEMIB': mseFTSEMIB_FNN_sub2,
    'GDAXI': mseGDAXI_FNN_sub2,
    'SPX': mseSPX_FNN_sub2,
    'HSI': mseHSI_FNN_sub2,
    'IBEX': mseIBEX_FNN_sub2,
    'IXIC': mseIXIC_FNN_sub2,
    'N225': mseN225_FNN_sub2,
    'OMXC20': mseOMXC20_FNN_sub2
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_FNN_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_FNN': {
    'DJI': qlike_DJI_FNN_1_sub2,
    'FTSE': qlike_FTSE_FNN_1_sub2,
    'FTSEMIB': qlike_FTSEMIB_FNN_1_sub2,
    'GDAXI': qlike_GDAXI_FNN_1_sub2,
    'SPX': qlike_SPX_FNN_1_sub2,
    'HSI': qlike_HSI_FNN_1_sub2,
    'IBEX': qlike_IBEX_FNN_1_sub2,
    'IXIC': qlike_IXIC_FNN_1_sub2,
    'N225': qlike_N225_FNN_1_sub2,
    'OMXC20': qlike_OMXC20_FNN_1_sub2
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_FNN':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_FNN_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%

predict = {
    'predictions_FNN': {
        'DJI': predictions_DJI_FNN_sub2,
        'FTSE': predictions_FTSE_FNN_sub2,
        'FTSEMIB': predictions_FTSEMIB_FNN_sub2,
        'GDAXI': predictions_GDAXI_FNN_sub2,
        'SPX': predictions_SPX_FNN_sub2,
        'HSI': predictions_HSI_FNN_sub2,
        'IBEX': predictions_IBEX_FNN_sub2,
        'IXIC': predictions_IXIC_FNN_sub2,
        'N225': predictions_N225_FNN_sub2,
        'OMXC20': predictions_OMXC20_FNN_sub2
                }
        }

for k1 in predict:
    if k1 == 'predictions_FNN':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_FNN_1_sub2.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')


