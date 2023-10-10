# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:49:06 2023

@author: cesar
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#%% Random Forest Subsample1

# Optimal parameters form cross validation

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

start_date_DJI = '2015-09-14'
end_date_DJI = '2020-02-21' 
start_date_DJI = pd.to_datetime(start_date_DJI)
end_date_DJI = pd.to_datetime(end_date_DJI)
y_test_AR1_DJI_sub1 = DJI.loc[start_date_DJI:end_date_DJI]

start_date_FTSE = '2015-09-24'
end_date_FTSE = '2020-02-24' 
start_date_FTSE = pd.to_datetime(start_date_FTSE)
end_date_FTSE = pd.to_datetime(end_date_FTSE)
y_test_AR1_FTSE_sub1 = FTSE.loc[start_date_FTSE:end_date_FTSE]

start_date_FTSEMIB = '2018-07-09'
end_date_FTSEMIB = '2020-02-21' 
start_date_FTSEMIB = pd.to_datetime(start_date_FTSEMIB)
end_date_FTSEMIB = pd.to_datetime(end_date_FTSEMIB)
y_test_AR1_FTSEMIB_sub1 = FTSEMIB.loc[start_date_FTSEMIB:end_date_FTSEMIB]

start_date_GDAXI = '2015-09-09'
end_date_GDAXI = '2020-02-25' 
start_date_GDAXI = pd.to_datetime(start_date_GDAXI)
end_date_GDAXI = pd.to_datetime(end_date_GDAXI)
y_test_AR1_GDAXI_sub1 = GDAXI.loc[start_date_GDAXI:end_date_GDAXI]

start_date_SPX = '2015-09-17'
end_date_SPX = '2020-02-25' 
start_date_SPX = pd.to_datetime(start_date_SPX)
end_date_SPX = pd.to_datetime(end_date_SPX)
y_test_AR1_SPX_sub1 = SPX.loc[start_date_SPX:end_date_SPX]

start_date_HSI = '2015-09-16'
end_date_HSI = '2020-03-10' 
start_date_HSI = pd.to_datetime(start_date_HSI)
end_date_HSI = pd.to_datetime(end_date_HSI)
y_test_AR1_HSI_sub1 = HSI.loc[start_date_HSI:end_date_HSI]

start_date_IBEX = '2015-10-05'
end_date_IBEX = '2020-02-26' 
start_date_IBEX = pd.to_datetime(start_date_IBEX)
end_date_IBEX = pd.to_datetime(end_date_IBEX)
y_test_AR1_IBEX_sub1 = IBEX.loc[start_date_IBEX:end_date_IBEX]

start_date_IXIC = '2015-09-23'
end_date_IXIC = '2020-02-21' 
start_date_IXIC = pd.to_datetime(start_date_IXIC)
end_date_IXIC = pd.to_datetime(end_date_IXIC)
y_test_AR1_IXIC_sub1 = IXIC.loc[start_date_IXIC:end_date_IXIC]

start_date_N225 = '2015-09-10'
end_date_N225 = '2020-02-21' 
start_date_N225 = pd.to_datetime(start_date_N225)
end_date_N225 = pd.to_datetime(end_date_N225)
y_test_AR1_N225_sub1 = N225.loc[start_date_N225:end_date_N225]

start_date_OMXC20 = '2017-06-11'
end_date_OMXC20 = '2020-02-21' 
start_date_OMXC20 = pd.to_datetime(start_date_OMXC20)
end_date_OMXC20 = pd.to_datetime(end_date_OMXC20)
y_test_AR1_OMXC20_sub1 = OMXC20.loc[start_date_OMXC20:end_date_OMXC20]

#%% 

X_test_DJI_rf_sub1 = X_test_DJI_rf[start_date_DJI:end_date_DJI]
X_test_FTSE_rf_sub1 = X_test_FTSE_rf[start_date_FTSE:end_date_FTSE]
X_test_FTSEMIB_rf_sub1 = X_test_FTSEMIB_rf[start_date_FTSEMIB:end_date_FTSEMIB]
X_test_GDAXI_rf_sub1 = X_test_GDAXI_rf[start_date_GDAXI:end_date_GDAXI]
X_test_SPX_rf_sub1 = X_test_SPX_rf[start_date_SPX:end_date_SPX]
X_test_HSI_rf_sub1 = X_test_HSI_rf[start_date_HSI:end_date_HSI]
X_test_IBEX_rf_sub1 = X_test_IBEX_rf[start_date_IBEX:end_date_IBEX]
X_test_IXIC_rf_sub1 = X_test_IXIC_rf[start_date_IXIC:end_date_IXIC]
X_test_N225_rf_sub1 = X_test_N225_rf[start_date_N225:end_date_N225]
X_test_OMXC20_rf_sub1 = X_test_OMXC20_rf[start_date_OMXC20:end_date_OMXC20]

y_test_DJI_rf_sub1 = y_test_DJI_rf[start_date_DJI:end_date_DJI]
y_test_FTSE_rf_sub1 = y_test_FTSE_rf[start_date_FTSE:end_date_FTSE]
y_test_FTSEMIB_rf_sub1 = y_test_FTSEMIB_rf[start_date_FTSEMIB:end_date_FTSEMIB]
y_test_GDAXI_rf_sub1 = y_test_GDAXI_rf[start_date_GDAXI:end_date_GDAXI]
y_test_SPX_rf_sub1 = y_test_SPX_rf[start_date_SPX:end_date_SPX]
y_test_HSI_rf_sub1 = y_test_HSI_rf[start_date_HSI:end_date_HSI]
y_test_IBEX_rf_sub1 = y_test_IBEX_rf[start_date_IBEX:end_date_IBEX]
y_test_IXIC_rf_sub1 = y_test_IXIC_rf[start_date_IXIC:end_date_IXIC]
y_test_N225_rf_sub1 = y_test_N225_rf[start_date_N225:end_date_N225]
y_test_OMXC20_rf_sub1 = y_test_OMXC20_rf[start_date_OMXC20:end_date_OMXC20]

y_pred_1_DJI_rf_sub1 = best_regressor_DJI.predict(X_test_DJI_rf_sub1)
y_pred_1_FTSE_rf_sub1 = best_regressor_FTSE.predict(X_test_FTSE_rf_sub1)
y_pred_1_FTSEMIB_rf_sub1 = best_regressor_FTSEMIB.predict(X_test_FTSEMIB_rf_sub1)
y_pred_1_GDAXI_rf_sub1 = best_regressor_GDAXI.predict(X_test_GDAXI_rf_sub1)
y_pred_1_SPX_rf_sub1 = best_regressor_SPX.predict(X_test_SPX_rf_sub1)
y_pred_1_HSI_rf_sub1 = best_regressor_HSI.predict(X_test_HSI_rf_sub1)
y_pred_1_IBEX_rf_sub1 = best_regressor_IBEX.predict(X_test_IBEX_rf_sub1)
y_pred_1_IXIC_rf_sub1 = best_regressor_IXIC.predict(X_test_IXIC_rf_sub1)
y_pred_1_N225_rf_sub1 = best_regressor_N225.predict(X_test_N225_rf_sub1)
y_pred_1_OMXC20_rf_sub1 = best_regressor_OMXC20.predict(X_test_OMXC20_rf_sub1)

# MSE 

MSE_DJI_1_rf_sub1 = mean_squared_error(y_test_DJI_rf_sub1, y_pred_1_DJI_rf_sub1)
MSE_FTSE_1_rf_sub1 = mean_squared_error(y_test_FTSE_rf_sub1, y_pred_1_FTSE_rf_sub1)
MSE_FTSEMIB_1_rf_sub1 = mean_squared_error(y_test_FTSEMIB_rf_sub1, y_pred_1_FTSEMIB_rf_sub1)
MSE_GDAXI_1_rf_sub1 = mean_squared_error(y_test_GDAXI_rf_sub1, y_pred_1_GDAXI_rf_sub1)
MSE_SPX_1_rf_sub1 = mean_squared_error(y_test_SPX_rf_sub1, y_pred_1_SPX_rf_sub1)
MSE_HSI_1_rf_sub1 = mean_squared_error(y_test_HSI_rf_sub1, y_pred_1_HSI_rf_sub1)
MSE_IBEX_1_rf_sub1 = mean_squared_error(y_test_IBEX_rf_sub1, y_pred_1_IBEX_rf_sub1)
MSE_IXIC_1_rf_sub1 = mean_squared_error(y_test_IXIC_rf_sub1, y_pred_1_IXIC_rf_sub1)
MSE_N225_1_rf_sub1 = mean_squared_error(y_test_N225_rf_sub1, y_pred_1_N225_rf_sub1)
MSE_OMXC20_1_rf_sub1 = mean_squared_error(y_test_OMXC20_rf_sub1, y_pred_1_OMXC20_rf_sub1)

mseDJI_rf_sub1 = []
for i in np.arange(len(y_test_DJI_rf_sub1)):
    mse = (y_pred_1_DJI_rf_sub1[i]-y_test_DJI_rf_sub1[i])**2
    mseDJI_rf_sub1.append(mse)
mseDJI_rf_sub1 = np.array(mseDJI_rf_sub1)

mseFTSE_rf_sub1 = []
for i in np.arange(len(y_test_FTSE_rf_sub1)):
    mse = (y_pred_1_FTSE_rf_sub1[i]-y_test_FTSE_rf_sub1[i])**2
    mseFTSE_rf_sub1.append(mse)
mseFTSE_rf_sub1 = np.array(mseFTSE_rf_sub1)

mseFTSEMIB_rf_sub1 = []
for i in np.arange(len(y_test_FTSEMIB_rf_sub1)):
    mse = (y_pred_1_FTSEMIB_rf_sub1[i]-y_test_FTSEMIB_rf_sub1[i])**2
    mseFTSEMIB_rf_sub1.append(mse)
mseFTSEMIB_rf_sub1 = np.array(mseFTSEMIB_rf_sub1)

mseGDAXI_rf_sub1 = []
for i in np.arange(len(y_test_GDAXI_rf_sub1)):
    mse = (y_pred_1_GDAXI_rf_sub1[i]-y_test_GDAXI_rf_sub1[i])**2
    mseGDAXI_rf_sub1.append(mse)
mseGDAXI_rf_sub1 = np.array(mseGDAXI_rf_sub1)

mseSPX_rf_sub1 = []
for i in np.arange(len(y_test_SPX_rf_sub1)):
    mse = (y_pred_1_SPX_rf_sub1[i]-y_test_SPX_rf_sub1[i])**2
    mseSPX_rf_sub1.append(mse)
mseSPX_rf_sub1 = np.array(mseSPX_rf_sub1)

mseHSI_rf_sub1 = []
for i in np.arange(len(y_test_HSI_rf_sub1)):
    mse = (y_pred_1_HSI_rf_sub1[i]-y_test_HSI_rf_sub1[i])**2
    mseHSI_rf_sub1.append(mse)
mseHSI_rf_sub1 = np.array(mseHSI_rf_sub1)

mseIBEX_rf_sub1 = []
for i in np.arange(len(y_test_IBEX_rf_sub1)):
    mse = (y_pred_1_IBEX_rf_sub1[i]-y_test_IBEX_rf_sub1[i])**2
    mseIBEX_rf_sub1.append(mse)
mseIBEX_rf_sub1 = np.array(mseIBEX_rf_sub1)

mseIXIC_rf_sub1 = []
for i in np.arange(len(y_test_IXIC_rf_sub1)):
    mse = (y_pred_1_IXIC_rf_sub1[i]-y_test_IXIC_rf_sub1[i])**2
    mseIXIC_rf_sub1.append(mse)
mseIXIC_rf_sub1 = np.array(mseIXIC_rf_sub1)

mseN225_rf_sub1 = []
for i in np.arange(len(y_test_N225_rf_sub1)):
    mse = (y_pred_1_N225_rf_sub1[i]-y_test_N225_rf_sub1[i])**2
    mseN225_rf_sub1.append(mse)
mseN225_rf_sub1 = np.array(mseN225_rf_sub1)

mseOMXC20_rf_sub1 = []
for i in np.arange(len(y_test_OMXC20_rf_sub1)):
    mse = (y_pred_1_OMXC20_rf_sub1[i]-y_test_OMXC20_rf_sub1[i])**2
    mseOMXC20_rf_sub1.append(mse)
mseOMXC20_rf_sub1 = np.array(mseOMXC20_rf_sub1)

# QLIKE

# DJI

y_forecastvalues = np.array(y_pred_1_DJI_rf_sub1)
y_actualvalues = np.array(y_test_DJI_rf_sub1)
qlikeDJI_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_rf_1_sub1.append(iteration)
qlikeDJI_rf_1_sub1 = np.array(qlikeDJI_rf_1_sub1)
QLIKE_rf_1_DJI_sub1 = sum(qlikeDJI_rf_1_sub1)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(y_pred_1_FTSE_rf_sub1)
y_actualvalues = np.array(y_test_FTSE_rf_sub1)
qlikeFTSE_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_rf_1_sub1.append(iteration)
qlikeFTSE_rf_1_sub1 = np.array(qlikeFTSE_rf_1_sub1)
QLIKE_rf_1_FTSE_sub1 = sum(qlikeFTSE_rf_1_sub1)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(y_pred_1_FTSEMIB_rf_sub1)
y_actualvalues = np.array(y_test_FTSEMIB_rf_sub1)
qlikeFTSEMIB_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_rf_1_sub1.append(iteration)
qlikeFTSEMIB_rf_1_sub1 = np.array(qlikeFTSEMIB_rf_1_sub1)
QLIKE_rf_1_FTSEMIB_sub1 = sum(qlikeFTSEMIB_rf_1_sub1)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(y_pred_1_GDAXI_rf_sub1)
y_actualvalues = np.array(y_test_GDAXI_rf_sub1)
qlikeGDAXI_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_rf_1_sub1.append(iteration)
qlikeGDAXI_rf_1_sub1 = np.array(qlikeGDAXI_rf_1_sub1)
QLIKE_rf_1_GDAXI_sub1 = sum(qlikeGDAXI_rf_1_sub1)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(y_pred_1_SPX_rf_sub1)
y_actualvalues = np.array(y_test_SPX_rf_sub1)
qlikeSPX_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_rf_1_sub1.append(iteration)
qlikeSPX_rf_1_sub1 = np.array(qlikeSPX_rf_1_sub1)
QLIKE_rf_1_SPX_sub1 = sum(qlikeSPX_rf_1_sub1)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(y_pred_1_HSI_rf_sub1)
y_actualvalues = np.array(y_test_HSI_rf_sub1)
qlikeHSI_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_rf_1_sub1.append(iteration)
qlikeHSI_rf_1_sub1 = np.array(qlikeHSI_rf_1_sub1)
QLIKE_rf_1_HSI_sub1 = sum(qlikeHSI_rf_1_sub1)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(y_pred_1_IBEX_rf_sub1)
y_actualvalues = np.array(y_test_IBEX_rf_sub1)
qlikeIBEX_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_rf_1_sub1.append(iteration)
qlikeIBEX_rf_1_sub1 = np.array(qlikeIBEX_rf_1_sub1)
QLIKE_rf_1_IBEX_sub1 = sum(qlikeIBEX_rf_1_sub1)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(y_pred_1_IXIC_rf_sub1)
y_actualvalues = np.array(y_test_IXIC_rf_sub1)
qlikeIXIC_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_rf_1_sub1.append(iteration)
qlikeIXIC_rf_1_sub1 = np.array(qlikeIXIC_rf_1_sub1)
QLIKE_rf_1_IXIC_sub1 = sum(qlikeIXIC_rf_1_sub1)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(y_pred_1_N225_rf_sub1)
y_actualvalues = np.array(y_test_N225_rf_sub1)
qlikeN225_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_rf_1_sub1.append(iteration)
qlikeN225_rf_1_sub1 = np.array(qlikeN225_rf_1_sub1)
QLIKE_rf_1_N225_sub1 = sum(qlikeN225_rf_1_sub1)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(y_pred_1_OMXC20_rf_sub1)
y_actualvalues = np.array(y_test_OMXC20_rf_sub1)
qlikeOMXC20_rf_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_rf_1_sub1.append(iteration)
qlikeOMXC20_rf_1_sub1 = np.array(qlikeOMXC20_rf_1_sub1)
QLIKE_rf_1_OMXC20_sub1 = sum(qlikeOMXC20_rf_1_sub1)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_rf': {
        'DJI': mseDJI_rf_sub1,
        'FTSE': mseFTSE_rf_sub1,
        'FTSEMIB': mseFTSEMIB_rf_sub1,
        'GDAXI': mseGDAXI_rf_sub1,
        'SPX': mseSPX_rf_sub1,
        'HSI': mseHSI_rf_sub1,
        'IBEX': mseIBEX_rf_sub1,
        'IXIC': mseIXIC_rf_sub1,
        'N225': mseN225_rf_sub1,
        'OMXC20': mseOMXC20_rf_sub1
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_rf_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%%

arrays_Out = {
    'qlike_rf': {
        'DJI': qlikeDJI_rf_1_sub1,
        'FTSE': qlikeFTSE_rf_1_sub1,
        'FTSEMIB': qlikeFTSEMIB_rf_1_sub1,
        'GDAXI': qlikeGDAXI_rf_1_sub1,
        'SPX': qlikeSPX_rf_1_sub1,
        'HSI': qlikeHSI_rf_1_sub1,
        'IBEX': qlikeIBEX_rf_1_sub1,
        'IXIC': qlikeIXIC_rf_1_sub1,
        'N225': qlikeN225_rf_1_sub1,
        'OMXC20': qlikeOMXC20_rf_1_sub1
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_rf_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%

predict = {
    'predictions_rf': {
        'DJI': y_pred_1_DJI_rf_sub1,
        'FTSE': y_pred_1_FTSE_rf_sub1,
        'FTSEMIB': y_pred_1_FTSEMIB_rf_sub1,
        'GDAXI': y_pred_1_GDAXI_rf_sub1,
        'SPX': y_pred_1_SPX_rf_sub1,
        'HSI': y_pred_1_HSI_rf_sub1,
        'IBEX': y_pred_1_IBEX_rf_sub1,
        'IXIC': y_pred_1_IXIC_rf_sub1,
        'N225': y_pred_1_N225_rf_sub1,
        'OMXC20': y_pred_1_OMXC20_rf_sub1
                }
        }

for k1 in predict:
    if k1 == 'predictions_rf':    
        for k2 in predict[k1]:
            nome_file = 'predictions{}_rf_1_sub1.csv'.format(k2)
            np.savetxt(nome_file, predict[k1][k2], delimiter=',')