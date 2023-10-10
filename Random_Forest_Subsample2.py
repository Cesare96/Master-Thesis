# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:21:34 2023

@author: cesar
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#%% Random Forest Subsample1

# Optimal parameters from cross validation

lag_DJI = 17
param_DJI = {'max_depth':None, 'n_estimators':100}

lag_FTSE = 1
param_FTSE = {'max_depth':5, 'n_estimators':200}

lag_FTSEMIB = 1
param_FTSEMIB = {'max_depth':5, 'n_estimators':100}

lag_GDAXI = 1
param_GDAXI = {'max_depth':5, 'n_estimators':10}

lag_SPX = 22
param_SPX = {'max_depth':20, 'n_estimators':100}

lag_HSI = 1
param_HSI = {'max_depth':5, 'n_estimators':100}

lag_IBEX = 1
param_IBEX = {'max_depth':5, 'n_estimators':200}

lag_IXIC = 21
param_IXIC = {'max_depth':5, 'n_estimators':200}

lag_N225 = 1
param_N225 = {'max_depth':5, 'n_estimators':200}

lag_OMXC20 = 1
param_OMXC20 = {'max_depth':5, 'n_estimators':50}

#%%

start_date2_DJI = '2020-08-17'
start_date2_DJI = pd.to_datetime(start_date2_DJI, format = '%Y-%m-%d')
end_date2_DJI = '2022-05-05'
end_date2_DJI = pd.to_datetime(end_date2_DJI, format = '%Y-%m-%d')

start_date2_FTSE = '2020-08-11'
start_date2_FTSE = pd.to_datetime(start_date2_FTSE, format = '%Y-%m-%d')
end_date2_FTSE = '2022-05-05'
end_date2_FTSE = pd.to_datetime(end_date2_FTSE, format = '%Y-%m-%d')

start_date2_FTSEMIB = '2020-08-02'
start_date2_FTSEMIB = pd.to_datetime(start_date2_FTSEMIB, format = '%Y-%m-%d')
end_date2_FTSEMIB = '2022-05-05'
end_date2_FTSEMIB = pd.to_datetime(end_date2_FTSEMIB, format = '%Y-%m-%d')

start_date2_GDAXI = '2020-07-19'
start_date2_GDAXI = pd.to_datetime(start_date2_GDAXI, format = '%Y-%m-%d')
end_date2_GDAXI = '2022-05-05'
end_date2_GDAXI = pd.to_datetime(end_date2_GDAXI, format = '%Y-%m-%d')

start_date2_SPX = '2020-11-02'
start_date2_SPX = pd.to_datetime(start_date2_SPX, format = '%Y-%m-%d')
end_date2_SPX = '2022-05-05'
end_date2_SPX = pd.to_datetime(end_date2_SPX, format = '%Y-%m-%d')

start_date2_HSI = '2020-04-26'
start_date2_HSI = pd.to_datetime(start_date2_HSI, format = '%Y-%m-%d')
end_date2_HSI = '2022-05-05'
end_date2_HSI = pd.to_datetime(end_date2_HSI, format = '%Y-%m-%d')

start_date2_IBEX = '2020-05-15'
start_date2_IBEX = pd.to_datetime(start_date2_IBEX, format = '%Y-%m-%d')
end_date2_IBEX = '2022-05-05'
end_date2_IBEX = pd.to_datetime(end_date2_IBEX, format = '%Y-%m-%d')

start_date2_IXIC = '2020-07-23'
start_date2_IXIC = pd.to_datetime(start_date2_IXIC, format = '%Y-%m-%d')
end_date2_IXIC = '2022-05-05'
end_date2_IXIC = pd.to_datetime(end_date2_IXIC, format = '%Y-%m-%d')

start_date2_N225 = '2020-05-06'
start_date2_N225 = pd.to_datetime(start_date2_N225, format = '%Y-%m-%d')
end_date2_N225 = '2022-05-05'
end_date2_N225 = pd.to_datetime(end_date2_N225, format = '%Y-%m-%d')

start_date2_OMXC20 = '2020-05-02'
start_date2_OMXC20 = pd.to_datetime(start_date2_OMXC20, format = '%Y-%m-%d')
end_date2_OMXC20 = '2022-05-05'
end_date2_OMXC20 = pd.to_datetime(end_date2_OMXC20, format = '%Y-%m-%d')


X_test_DJI_rf_sub2 = X_test_DJI_rf[start_date2_DJI:end_date2_DJI]
X_test_FTSE_rf_sub2 = X_test_FTSE_rf[start_date2_FTSE:end_date2_FTSE]
X_test_FTSEMIB_rf_sub2 = X_test_FTSEMIB_rf[start_date2_FTSEMIB:end_date2_FTSEMIB]
X_test_GDAXI_rf_sub2 = X_test_GDAXI_rf[start_date2_GDAXI:end_date2_GDAXI]
X_test_SPX_rf_sub2 = X_test_SPX_rf[start_date2_SPX:end_date2_SPX]
X_test_HSI_rf_sub2 = X_test_HSI_rf[start_date2_HSI:end_date2_HSI]
X_test_IBEX_rf_sub2 = X_test_IBEX_rf[start_date2_IBEX:end_date2_IBEX]
X_test_IXIC_rf_sub2 = X_test_IXIC_rf[start_date2_IXIC:end_date2_IXIC]
X_test_N225_rf_sub2 = X_test_N225_rf[start_date2_N225:end_date2_N225]
X_test_OMXC20_rf_sub2 = X_test_OMXC20_rf[start_date2_OMXC20:end_date2_OMXC20]

y_test_DJI_rf_sub2 = y_test_DJI_rf[start_date2_DJI:end_date2_DJI]
y_test_FTSE_rf_sub2 = y_test_FTSE_rf[start_date2_FTSE:end_date2_FTSE]
y_test_FTSEMIB_rf_sub2 = y_test_FTSEMIB_rf[start_date2_FTSEMIB:end_date2_FTSEMIB]
y_test_GDAXI_rf_sub2 = y_test_GDAXI_rf[start_date2_GDAXI:end_date2_GDAXI]
y_test_SPX_rf_sub2 = y_test_SPX_rf[start_date2_SPX:end_date2_SPX]
y_test_HSI_rf_sub2 = y_test_HSI_rf[start_date2_HSI:end_date2_HSI]
y_test_IBEX_rf_sub2 = y_test_IBEX_rf[start_date2_IBEX:end_date2_IBEX]
y_test_IXIC_rf_sub2 = y_test_IXIC_rf[start_date2_IXIC:end_date2_IXIC]
y_test_N225_rf_sub2 = y_test_N225_rf[start_date2_N225:end_date2_N225]
y_test_OMXC20_rf_sub2 = y_test_OMXC20_rf[start_date2_OMXC20:end_date2_OMXC20]

y_pred_1_DJI_rf_sub2 = best_regressor_DJI.predict(X_test_DJI_rf_sub2)
y_pred_1_FTSE_rf_sub2 = best_regressor_FTSE.predict(X_test_FTSE_rf_sub2)
y_pred_1_FTSEMIB_rf_sub2 = best_regressor_FTSEMIB.predict(X_test_FTSEMIB_rf_sub2)
y_pred_1_GDAXI_rf_sub2 = best_regressor_GDAXI.predict(X_test_GDAXI_rf_sub2)
y_pred_1_SPX_rf_sub2 = best_regressor_SPX.predict(X_test_SPX_rf_sub2)
y_pred_1_HSI_rf_sub2 = best_regressor_HSI.predict(X_test_HSI_rf_sub2)
y_pred_1_IBEX_rf_sub2 = best_regressor_IBEX.predict(X_test_IBEX_rf_sub2)
y_pred_1_IXIC_rf_sub2 = best_regressor_IXIC.predict(X_test_IXIC_rf_sub2)
y_pred_1_N225_rf_sub2 = best_regressor_N225.predict(X_test_N225_rf_sub2)
y_pred_1_OMXC20_rf_sub2 = best_regressor_OMXC20.predict(X_test_OMXC20_rf_sub2)

# MSE 

MSE_DJI_1_rf_sub2 = mean_squared_error(y_test_DJI_rf_sub2, y_pred_1_DJI_rf_sub2)
MSE_FTSE_1_rf_sub2 = mean_squared_error(y_test_FTSE_rf_sub2, y_pred_1_FTSE_rf_sub2)
MSE_FTSEMIB_1_rf_sub2 = mean_squared_error(y_test_FTSEMIB_rf_sub2, y_pred_1_FTSEMIB_rf_sub2)
MSE_GDAXI_1_rf_sub2 = mean_squared_error(y_test_GDAXI_rf_sub2, y_pred_1_GDAXI_rf_sub2)
MSE_SPX_1_rf_sub2 = mean_squared_error(y_test_SPX_rf_sub2, y_pred_1_SPX_rf_sub2)
MSE_HSI_1_rf_sub2 = mean_squared_error(y_test_HSI_rf_sub2, y_pred_1_HSI_rf_sub2)
MSE_IBEX_1_rf_sub2 = mean_squared_error(y_test_IBEX_rf_sub2, y_pred_1_IBEX_rf_sub2)
MSE_IXIC_1_rf_sub2 = mean_squared_error(y_test_IXIC_rf_sub2, y_pred_1_IXIC_rf_sub2)
MSE_N225_1_rf_sub2 = mean_squared_error(y_test_N225_rf_sub2, y_pred_1_N225_rf_sub2)
MSE_OMXC20_1_rf_sub2 = mean_squared_error(y_test_OMXC20_rf_sub2, y_pred_1_OMXC20_rf_sub2)

mseDJI_rf_sub2 = []
for i in np.arange(len(y_test_DJI_rf_sub2)):
    mse = (y_pred_1_DJI_rf_sub2[i]-y_test_DJI_rf_sub2[i])**2
    mseDJI_rf_sub2.append(mse)
mseDJI_rf_sub2 = np.array(mseDJI_rf_sub2)

mseFTSE_rf_sub2 = []
for i in np.arange(len(y_test_FTSE_rf_sub2)):
    mse = (y_pred_1_FTSE_rf_sub2[i]-y_test_FTSE_rf_sub2[i])**2
    mseFTSE_rf_sub2.append(mse)
mseFTSE_rf_sub2 = np.array(mseFTSE_rf_sub2)

mseFTSEMIB_rf_sub2 = []
for i in np.arange(len(y_test_FTSEMIB_rf_sub2)):
    mse = (y_pred_1_FTSEMIB_rf_sub2[i]-y_test_FTSEMIB_rf_sub2[i])**2
    mseFTSEMIB_rf_sub2.append(mse)
mseFTSEMIB_rf_sub2 = np.array(mseFTSEMIB_rf_sub2)

mseGDAXI_rf_sub2 = []
for i in np.arange(len(y_test_GDAXI_rf_sub2)):
    mse = (y_pred_1_GDAXI_rf_sub2[i]-y_test_GDAXI_rf_sub2[i])**2
    mseGDAXI_rf_sub2.append(mse)
mseGDAXI_rf_sub2 = np.array(mseGDAXI_rf_sub2)

mseSPX_rf_sub2 = []
for i in np.arange(len(y_test_SPX_rf_sub2)):
    mse = (y_pred_1_SPX_rf_sub2[i]-y_test_SPX_rf_sub2[i])**2
    mseSPX_rf_sub2.append(mse)
mseSPX_rf_sub2 = np.array(mseSPX_rf_sub2)

mseHSI_rf_sub2 = []
for i in np.arange(len(y_test_HSI_rf_sub2)):
    mse = (y_pred_1_HSI_rf_sub2[i]-y_test_HSI_rf_sub2[i])**2
    mseHSI_rf_sub2.append(mse)
mseHSI_rf_sub2 = np.array(mseHSI_rf_sub2)

mseIBEX_rf_sub2 = []
for i in np.arange(len(y_test_IBEX_rf_sub2)):
    mse = (y_pred_1_IBEX_rf_sub2[i]-y_test_IBEX_rf_sub2[i])**2
    mseIBEX_rf_sub2.append(mse)
mseIBEX_rf_sub2 = np.array(mseIBEX_rf_sub2)

mseIXIC_rf_sub2 = []
for i in np.arange(len(y_test_IXIC_rf_sub2)):
    mse = (y_pred_1_IXIC_rf_sub2[i]-y_test_IXIC_rf_sub2[i])**2
    mseIXIC_rf_sub2.append(mse)
mseIXIC_rf_sub2 = np.array(mseIXIC_rf_sub2)

mseN225_rf_sub2 = []
for i in np.arange(len(y_test_N225_rf_sub2)):
    mse = (y_pred_1_N225_rf_sub2[i]-y_test_N225_rf_sub2[i])**2
    mseN225_rf_sub2.append(mse)
mseN225_rf_sub2 = np.array(mseN225_rf_sub2)

mseOMXC20_rf_sub2 = []
for i in np.arange(len(y_test_OMXC20_rf_sub2)):
    mse = (y_pred_1_OMXC20_rf_sub2[i]-y_test_OMXC20_rf_sub2[i])**2
    mseOMXC20_rf_sub2.append(mse)
mseOMXC20_rf_sub2 = np.array(mseOMXC20_rf_sub2)

# QLIKE

# DJI

y_forecastvalues = np.array(y_pred_1_DJI_rf_sub2)
y_actualvalues = np.array(y_test_DJI_rf_sub2)
qlikeDJI_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_rf_1_sub2.append(iteration)
qlikeDJI_rf_1_sub2 = np.array(qlikeDJI_rf_1_sub2)
QLIKE_rf_1_DJI_sub2 = sum(qlikeDJI_rf_1_sub2)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(y_pred_1_FTSE_rf_sub2)
y_actualvalues = np.array(y_test_FTSE_rf_sub2)
qlikeFTSE_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_rf_1_sub2.append(iteration)
qlikeFTSE_rf_1_sub2 = np.array(qlikeFTSE_rf_1_sub2)
QLIKE_rf_1_FTSE_sub2 = sum(qlikeFTSE_rf_1_sub2)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(y_pred_1_FTSEMIB_rf_sub2)
y_actualvalues = np.array(y_test_FTSEMIB_rf_sub2)
qlikeFTSEMIB_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_rf_1_sub2.append(iteration)
qlikeFTSEMIB_rf_1_sub2 = np.array(qlikeFTSEMIB_rf_1_sub2)
QLIKE_rf_1_FTSEMIB_sub2 = sum(qlikeFTSEMIB_rf_1_sub2)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(y_pred_1_GDAXI_rf_sub2)
y_actualvalues = np.array(y_test_GDAXI_rf_sub2)
qlikeGDAXI_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_rf_1_sub2.append(iteration)
qlikeGDAXI_rf_1_sub2 = np.array(qlikeGDAXI_rf_1_sub2)
QLIKE_rf_1_GDAXI_sub2 = sum(qlikeGDAXI_rf_1_sub2)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(y_pred_1_SPX_rf_sub2)
y_actualvalues = np.array(y_test_SPX_rf_sub2)
qlikeSPX_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_rf_1_sub2.append(iteration)
qlikeSPX_rf_1_sub2 = np.array(qlikeSPX_rf_1_sub2)
QLIKE_rf_1_SPX_sub2 = sum(qlikeSPX_rf_1_sub2)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(y_pred_1_HSI_rf_sub2)
y_actualvalues = np.array(y_test_HSI_rf_sub2)
qlikeHSI_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_rf_1_sub2.append(iteration)
qlikeHSI_rf_1_sub2 = np.array(qlikeHSI_rf_1_sub2)
QLIKE_rf_1_HSI_sub2 = sum(qlikeHSI_rf_1_sub2)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(y_pred_1_IBEX_rf_sub2)
y_actualvalues = np.array(y_test_IBEX_rf_sub2)
qlikeIBEX_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_rf_1_sub2.append(iteration)
qlikeIBEX_rf_1_sub2 = np.array(qlikeIBEX_rf_1_sub2)
QLIKE_rf_1_IBEX_sub2 = sum(qlikeIBEX_rf_1_sub2)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(y_pred_1_IXIC_rf_sub2)
y_actualvalues = np.array(y_test_IXIC_rf_sub2)
qlikeIXIC_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_rf_1_sub2.append(iteration)
qlikeIXIC_rf_1_sub2 = np.array(qlikeIXIC_rf_1_sub2)
QLIKE_rf_1_IXIC_sub2 = sum(qlikeIXIC_rf_1_sub2)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(y_pred_1_N225_rf_sub2)
y_actualvalues = np.array(y_test_N225_rf_sub2)
qlikeN225_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_rf_1_sub2.append(iteration)
qlikeN225_rf_1_sub2 = np.array(qlikeN225_rf_1_sub2)
QLIKE_rf_1_N225_sub2 = sum(qlikeN225_rf_1_sub2)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(y_pred_1_OMXC20_rf_sub2)
y_actualvalues = np.array(y_test_OMXC20_rf_sub2)
qlikeOMXC20_rf_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_rf_1_sub2.append(iteration)
qlikeOMXC20_rf_1_sub2 = np.array(qlikeOMXC20_rf_1_sub2)
QLIKE_rf_1_OMXC20_sub2 = sum(qlikeOMXC20_rf_1_sub2)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_rf': {
        'DJI': mseDJI_rf_sub2,
        'FTSE': mseFTSE_rf_sub2,
        'FTSEMIB': mseFTSEMIB_rf_sub2,
        'GDAXI': mseGDAXI_rf_sub2,
        'SPX': mseSPX_rf_sub2,
        'HSI': mseHSI_rf_sub2,
        'IBEX': mseIBEX_rf_sub2,
        'IXIC': mseIXIC_rf_sub2,
        'N225': mseN225_rf_sub2,
        'OMXC20': mseOMXC20_rf_sub2
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_rf_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_rf': {
        'DJI': qlikeDJI_rf_1_sub2,
        'FTSE': qlikeFTSE_rf_1_sub2,
        'FTSEMIB': qlikeFTSEMIB_rf_1_sub2,
        'GDAXI': qlikeGDAXI_rf_1_sub2,
        'SPX': qlikeSPX_rf_1_sub2,
        'HSI': qlikeHSI_rf_1_sub2,
        'IBEX': qlikeIBEX_rf_1_sub2,
        'IXIC': qlikeIXIC_rf_1_sub2,
        'N225': qlikeN225_rf_1_sub2,
        'OMXC20': qlikeOMXC20_rf_1_sub2
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_rf_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%

predict = {
    'predictions_rf': {
        'DJI': y_pred_1_DJI_rf_sub2,
        'FTSE': y_pred_1_FTSE_rf_sub2,
        'FTSEMIB': y_pred_1_FTSEMIB_rf_sub2,
        'GDAXI': y_pred_1_GDAXI_rf_sub2,
        'SPX': y_pred_1_SPX_rf_sub2,
        'HSI': y_pred_1_HSI_rf_sub2,
        'IBEX': y_pred_1_IBEX_rf_sub2,
        'IXIC': y_pred_1_IXIC_rf_sub2,
        'N225': y_pred_1_N225_rf_sub2,
        'OMXC20': y_pred_1_OMXC20_rf_sub2
                }
        }

for k1 in predict:
    if k1 == 'predictions_rf':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_rf_1_sub2.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')
