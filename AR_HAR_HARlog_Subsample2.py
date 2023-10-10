# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:50:24 2023

@author: cesar
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

#%%

start_date2_DJI = '2020-08-17'
start_date2_DJI = pd.to_datetime(start_date2_DJI, format = '%Y-%m-%d')
end_date2_DJI = '2022-05-05'
end_date2_DJI = pd.to_datetime(end_date2_DJI, format = '%Y-%m-%d')
start_date2_DJI_X = '2020-08-16'
start_date2_DJI_X = pd.to_datetime(start_date2_DJI_X, format = '%Y-%m-%d')
end_date2_DJI_X = '2022-05-04'
end_date2_DJI_X = pd.to_datetime(end_date2_DJI_X, format = '%Y-%m-%d')

start_date2_FTSE = '2020-08-11'
start_date2_FTSE = pd.to_datetime(start_date2_FTSE, format = '%Y-%m-%d')
end_date2_FTSE = '2022-05-05'
end_date2_FTSE = pd.to_datetime(end_date2_FTSE, format = '%Y-%m-%d')
start_date2_FTSE_X = '2020-08-10'
start_date2_FTSE_X = pd.to_datetime(start_date2_FTSE_X, format = '%Y-%m-%d')
end_date2_FTSE_X = '2022-05-04'
end_date2_FTSE_X = pd.to_datetime(end_date2_FTSE_X, format = '%Y-%m-%d')

start_date2_FTSEMIB = '2020-08-02'
start_date2_FTSEMIB = pd.to_datetime(start_date2_FTSEMIB, format = '%Y-%m-%d')
end_date2_FTSEMIB = '2022-05-05'
end_date2_FTSEMIB = pd.to_datetime(end_date2_FTSEMIB, format = '%Y-%m-%d')
start_date2_FTSEMIB_X = '2020-07-30'
start_date2_FTSEMIB_X = pd.to_datetime(start_date2_FTSEMIB_X, format = '%Y-%m-%d')
end_date2_FTSEMIB_X = '2022-05-04'
end_date2_FTSEMIB_X = pd.to_datetime(end_date2_FTSEMIB_X, format = '%Y-%m-%d')

start_date2_GDAXI = '2020-07-19'
start_date2_GDAXI = pd.to_datetime(start_date2_GDAXI, format = '%Y-%m-%d')
end_date2_GDAXI = '2022-05-05'
end_date2_GDAXI = pd.to_datetime(end_date2_GDAXI, format = '%Y-%m-%d')
start_date2_GDAXI_X = '2020-07-16'
start_date2_GDAXI_X = pd.to_datetime(start_date2_GDAXI_X, format = '%Y-%m-%d')
end_date2_GDAXI_X = '2022-05-04'
end_date2_GDAXI_X = pd.to_datetime(end_date2_GDAXI_X, format = '%Y-%m-%d')

start_date2_SPX = '2020-11-02'
start_date2_SPX = pd.to_datetime(start_date2_SPX, format = '%Y-%m-%d')
end_date2_SPX = '2022-05-05'
end_date2_SPX = pd.to_datetime(end_date2_SPX, format = '%Y-%m-%d')
start_date2_SPX_X = '2020-10-30'
start_date2_SPX_X = pd.to_datetime(start_date2_SPX_X, format = '%Y-%m-%d')
end_date2_SPX_X = '2022-05-04'
end_date2_SPX_X = pd.to_datetime(end_date2_SPX_X, format = '%Y-%m-%d')

start_date2_HSI = '2020-04-26'
start_date2_HSI = pd.to_datetime(start_date2_HSI, format = '%Y-%m-%d')
end_date2_HSI = '2022-05-05'
end_date2_HSI = pd.to_datetime(end_date2_HSI, format = '%Y-%m-%d')
start_date2_HSI_X = '2020-04-23'
start_date2_HSI_X = pd.to_datetime(start_date2_HSI_X, format = '%Y-%m-%d')
end_date2_HSI_X = '2022-05-04'
end_date2_HSI_X = pd.to_datetime(end_date2_HSI_X, format = '%Y-%m-%d')

start_date2_IBEX = '2020-05-15'
start_date2_IBEX = pd.to_datetime(start_date2_IBEX, format = '%Y-%m-%d')
end_date2_IBEX = '2022-05-05'
end_date2_IBEX = pd.to_datetime(end_date2_IBEX, format = '%Y-%m-%d')
start_date2_IBEX_X = '2020-05-14'
start_date2_IBEX_X = pd.to_datetime(start_date2_IBEX_X, format = '%Y-%m-%d')
end_date2_IBEX_X = '2022-05-04'
end_date2_IBEX_X = pd.to_datetime(end_date2_IBEX_X, format = '%Y-%m-%d')

start_date2_IXIC = '2020-07-23'
start_date2_IXIC = pd.to_datetime(start_date2_IXIC, format = '%Y-%m-%d')
end_date2_IXIC = '2022-05-05'
end_date2_IXIC = pd.to_datetime(end_date2_IXIC, format = '%Y-%m-%d')
start_date2_IXIC_X = '2020-07-22'
start_date2_IXIC_X = pd.to_datetime(start_date2_IXIC_X, format = '%Y-%m-%d')
end_date2_IXIC_X = '2022-05-04'
end_date2_IXIC_X = pd.to_datetime(end_date2_IXIC_X, format = '%Y-%m-%d')

start_date2_N225 = '2020-05-06'
start_date2_N225 = pd.to_datetime(start_date2_N225, format = '%Y-%m-%d')
end_date2_N225 = '2022-05-05'
end_date2_N225 = pd.to_datetime(end_date2_N225, format = '%Y-%m-%d')
start_date2_N225_X = '2020-04-30'
start_date2_N225_X = pd.to_datetime(start_date2_N225_X, format = '%Y-%m-%d')
end_date2_N225_X = '2022-05-04'
end_date2_N225_X = pd.to_datetime(end_date2_N225_X, format = '%Y-%m-%d')

start_date2_OMXC20 = '2020-05-03'
start_date2_OMXC20 = pd.to_datetime(start_date2_OMXC20, format = '%Y-%m-%d')
end_date2_OMXC20 = '2022-05-05'
end_date2_OMXC20 = pd.to_datetime(end_date2_OMXC20, format = '%Y-%m-%d')
start_date2_OMXC20_X = '2020-04-30'
start_date2_OMXC20_X = pd.to_datetime(start_date2_OMXC20_X, format = '%Y-%m-%d')
end_date2_OMXC20_X = '2022-05-04'
end_date2_OMXC20_X = pd.to_datetime(end_date2_OMXC20_X, format = '%Y-%m-%d')

# AR(1) 

y_test_AR1_sub2_DJI = y_test_AR1_DJI[start_date2_DJI:end_date2_DJI]
X_test_AR1_sub2_DJI = X_test_AR1_DJI[start_date2_DJI_X:end_date2_DJI_X]

y_test_AR1_sub2_FTSE = y_test_AR1_FTSE[start_date2_FTSE:end_date2_FTSE]
X_test_AR1_sub2_FTSE = X_test_AR1_FTSE[start_date2_FTSE_X:end_date2_FTSE_X]

y_test_AR1_sub2_FTSEMIB = y_test_AR1_FTSEMIB[start_date2_FTSEMIB:end_date2_FTSEMIB]
X_test_AR1_sub2_FTSEMIB = X_test_AR1_FTSEMIB[start_date2_FTSEMIB_X:end_date2_FTSEMIB_X]

y_test_AR1_sub2_GDAXI = y_test_AR1_GDAXI[start_date2_GDAXI:end_date2_GDAXI]
X_test_AR1_sub2_GDAXI = X_test_AR1_GDAXI[start_date2_GDAXI_X:end_date2_GDAXI_X]

y_test_AR1_sub2_SPX = y_test_AR1_SPX[start_date2_SPX:end_date2_SPX]
X_test_AR1_sub2_SPX = X_test_AR1_SPX[start_date2_SPX_X:end_date2_SPX_X]

y_test_AR1_sub2_HSI = y_test_AR1_HSI[start_date2_HSI:end_date2_HSI]
X_test_AR1_sub2_HSI = X_test_AR1_HSI[start_date2_HSI_X:end_date2_HSI_X]

y_test_AR1_sub2_IBEX = y_test_AR1_IBEX[start_date2_IBEX:end_date2_IBEX]
X_test_AR1_sub2_IBEX = X_test_AR1_IBEX[start_date2_IBEX_X:end_date2_IBEX_X]

y_test_AR1_sub2_IXIC = y_test_AR1_IXIC[start_date2_IXIC:end_date2_IXIC]
X_test_AR1_sub2_IXIC = X_test_AR1_IXIC[start_date2_IXIC_X:end_date2_IXIC_X]

y_test_AR1_sub2_N225 = y_test_AR1_N225[start_date2_N225:end_date2_N225]
X_test_AR1_sub2_N225 = X_test_AR1_N225[start_date2_N225_X:end_date2_N225_X]

y_test_AR1_sub2_OMXC20 = y_test_AR1_OMXC20[start_date2_OMXC20:end_date2_OMXC20]
X_test_AR1_sub2_OMXC20 = X_test_AR1_OMXC20[start_date2_OMXC20_X:end_date2_OMXC20_X]

X_test_AR1_sub2_DJI_c = sm.add_constant(X_test_AR1_sub2_DJI)
X_test_AR1_sub2_FTSE_c = sm.add_constant(X_test_AR1_sub2_FTSE)
X_test_AR1_sub2_FTSEMIB_c = sm.add_constant(X_test_AR1_sub2_FTSEMIB)
X_test_AR1_sub2_GDAXI_c = sm.add_constant(X_test_AR1_sub2_GDAXI)
X_test_AR1_sub2_SPX_c = sm.add_constant(X_test_AR1_sub2_SPX)
X_test_AR1_sub2_HSI_c = sm.add_constant(X_test_AR1_sub2_HSI)
X_test_AR1_sub2_IBEX_c = sm.add_constant(X_test_AR1_sub2_IBEX)
X_test_AR1_sub2_IXIC_c = sm.add_constant(X_test_AR1_sub2_IXIC)
X_test_AR1_sub2_N225_c = sm.add_constant(X_test_AR1_sub2_N225)
X_test_AR1_sub2_OMXC20_c = sm.add_constant(X_test_AR1_sub2_OMXC20)

y_pred_AR1_1_DJI_sub2 = model_AR1_DJI.predict(X_test_AR1_sub2_DJI_c)
y_pred_AR1_1_FTSE_sub2 = model_AR1_FTSE.predict(X_test_AR1_sub2_FTSE_c)
y_pred_AR1_1_FTSEMIB_sub2 = model_AR1_FTSEMIB.predict(X_test_AR1_sub2_FTSEMIB_c)
y_pred_AR1_1_GDAXI_sub2 = model_AR1_GDAXI.predict(X_test_AR1_sub2_GDAXI_c)
y_pred_AR1_1_SPX_sub2 = model_AR1_SPX.predict(X_test_AR1_sub2_SPX_c)
y_pred_AR1_1_HSI_sub2 = model_AR1_HSI.predict(X_test_AR1_sub2_HSI_c)
y_pred_AR1_1_IBEX_sub2 = model_AR1_IBEX.predict(X_test_AR1_sub2_IBEX_c)
y_pred_AR1_1_IXIC_sub2 = model_AR1_IXIC.predict(X_test_AR1_sub2_IXIC_c)
y_pred_AR1_1_N225_sub2 = model_AR1_N225.predict(X_test_AR1_sub2_N225_c)
y_pred_AR1_1_OMXC20_sub2 = model_AR1_OMXC20.predict(X_test_AR1_sub2_OMXC20_c)

# AR(1) MSE

MSE_AR1_1_DJI_sub2 = mean_squared_error(y_test_AR1_sub2_DJI, y_pred_AR1_1_DJI_sub2)
MSE_AR1_1_FTSE_sub2 = mean_squared_error(y_test_AR1_sub2_FTSE, y_pred_AR1_1_FTSE_sub2)
MSE_AR1_1_FTSEMIB_sub2 = mean_squared_error(y_test_AR1_sub2_FTSEMIB, y_pred_AR1_1_FTSEMIB_sub2)
MSE_AR1_1_GDAXI_sub2 = mean_squared_error(y_test_AR1_sub2_GDAXI, y_pred_AR1_1_GDAXI_sub2)
MSE_AR1_1_SPX_sub2 = mean_squared_error(y_test_AR1_sub2_SPX, y_pred_AR1_1_SPX_sub2)
MSE_AR1_1_HSI_sub2 = mean_squared_error(y_test_AR1_sub2_HSI, y_pred_AR1_1_HSI_sub2)
MSE_AR1_1_IBEX_sub2 = mean_squared_error(y_test_AR1_sub2_IBEX, y_pred_AR1_1_IBEX_sub2)
MSE_AR1_1_IXIC_sub2 = mean_squared_error(y_test_AR1_sub2_IXIC, y_pred_AR1_1_IXIC_sub2)
MSE_AR1_1_N225_sub2 = mean_squared_error(y_test_AR1_sub2_N225, y_pred_AR1_1_N225_sub2)
MSE_AR1_1_OMXC20_sub2 = mean_squared_error(y_test_AR1_sub2_OMXC20, y_pred_AR1_1_OMXC20_sub2)

mseDJI_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_DJI)):
    mse = (y_pred_AR1_1_DJI_sub2[i]-y_test_AR1_sub2_DJI[i])**2
    mseDJI_AR1_sub2.append(mse)
mseDJI_AR1_sub2 = np.array(mseDJI_AR1_sub2)

mseFTSE_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_FTSE)):
    mse = (y_pred_AR1_1_FTSE_sub2[i]-y_test_AR1_sub2_FTSE[i])**2
    mseFTSE_AR1_sub2.append(mse)
mseFTSE_AR1_sub2 = np.array(mseFTSE_AR1_sub2)

mseFTSEMIB_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_FTSEMIB)):
    mse = (y_pred_AR1_1_FTSEMIB_sub2[i]-y_test_AR1_sub2_FTSEMIB[i])**2
    mseFTSEMIB_AR1_sub2.append(mse)
mseFTSEMIB_AR1_sub2 = np.array(mseFTSEMIB_AR1_sub2)

mseGDAXI_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_GDAXI)):
    mse = (y_pred_AR1_1_GDAXI_sub2[i]-y_test_AR1_sub2_GDAXI[i])**2
    mseGDAXI_AR1_sub2.append(mse)
mseGDAXI_AR1_sub2 = np.array(mseGDAXI_AR1_sub2)

mseSPX_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_SPX)):
    mse = (y_pred_AR1_1_SPX_sub2[i]-y_test_AR1_sub2_SPX[i])**2
    mseSPX_AR1_sub2.append(mse)
mseSPX_AR1_sub2 = np.array(mseSPX_AR1_sub2)

mseHSI_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_HSI)):
    mse = (y_pred_AR1_1_HSI_sub2[i]-y_test_AR1_sub2_HSI[i])**2
    mseHSI_AR1_sub2.append(mse)
mseHSI_AR1_sub2 = np.array(mseHSI_AR1_sub2)

mseIBEX_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_IBEX)):
    mse = (y_pred_AR1_1_IBEX_sub2[i]-y_test_AR1_sub2_IBEX[i])**2
    mseIBEX_AR1_sub2.append(mse)
mseIBEX_AR1_sub2 = np.array(mseIBEX_AR1_sub2)

mseIXIC_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_IXIC)):
    mse = (y_pred_AR1_1_IXIC_sub2[i]-y_test_AR1_sub2_IXIC[i])**2
    mseIXIC_AR1_sub2.append(mse)
mseIXIC_AR1_sub2 = np.array(mseIXIC_AR1_sub2)

mseN225_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_N225)):
    mse = (y_pred_AR1_1_N225_sub2[i]-y_test_AR1_sub2_N225[i])**2
    mseN225_AR1_sub2.append(mse)
mseN225_AR1_sub2 = np.array(mseN225_AR1_sub2)

mseOMXC20_AR1_sub2 = []
for i in np.arange(len(y_test_AR1_sub2_OMXC20)):
    mse = (y_pred_AR1_1_OMXC20_sub2[i]-y_test_AR1_sub2_OMXC20[i])**2
    mseOMXC20_AR1_sub2.append(mse)
mseOMXC20_AR1_sub2 = np.array(mseOMXC20_AR1_sub2)



# AR(1) QLIKE 

# DJI 

y_forecastvalues = np.array(y_pred_AR1_1_DJI_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_DJI)
qlikeDJI_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_DJI_sub2 = sum(qlikeDJI_AR1_1_sub2)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(y_pred_AR1_1_FTSE_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_FTSE)
qlikeFTSE_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_FTSE_sub2 = sum(qlikeFTSE_AR1_1_sub2)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(y_pred_AR1_1_FTSEMIB_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_FTSEMIB)
qlikeFTSEMIB_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_FTSEMIB_sub2 = sum(qlikeFTSEMIB_AR1_1_sub2)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(y_pred_AR1_1_GDAXI_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_GDAXI)
qlikeGDAXI_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_GDAXI_sub2 = sum(qlikeGDAXI_AR1_1_sub2)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(y_pred_AR1_1_SPX_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_SPX)
qlikeSPX_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_SPX_sub2 = sum(qlikeSPX_AR1_1_sub2)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(y_pred_AR1_1_HSI_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_HSI)
qlikeHSI_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_HSI_sub2 = sum(qlikeHSI_AR1_1_sub2)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(y_pred_AR1_1_IBEX_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_IBEX)
qlikeIBEX_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_IBEX_sub2 = sum(qlikeIBEX_AR1_1_sub2)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(y_pred_AR1_1_IXIC_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_IXIC)
qlikeIXIC_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_IXIC_sub2 = sum(qlikeIXIC_AR1_1_sub2)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(y_pred_AR1_1_N225_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_N225)
qlikeN225_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_N225_sub2 = sum(qlikeN225_AR1_1_sub2)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(y_pred_AR1_1_OMXC20_sub2)
y_actualvalues = np.array(y_test_AR1_sub2_OMXC20)
qlikeOMXC20_AR1_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_AR1_1_sub2.append(iteration)
QLIKE_AR1_1_OMXC20_sub2 = sum(qlikeOMXC20_AR1_1_sub2)/len(y_actualvalues)

# HAR 

y_test_DJI_sub2 = y_test_DJI[start_date2_DJI:end_date2_DJI]
y_test_FTSE_sub2 = y_test_FTSE[start_date2_FTSE:end_date2_FTSE]
y_test_FTSEMIB_sub2 = y_test_FTSEMIB[start_date2_FTSEMIB:end_date2_FTSEMIB]
y_test_GDAXI_sub2 = y_test_GDAXI[start_date2_GDAXI:end_date2_GDAXI]
y_test_SPX_sub2 = y_test_SPX[start_date2_SPX:end_date2_SPX]
y_test_HSI_sub2 = y_test_HSI[start_date2_HSI:end_date2_HSI]
y_test_IBEX_sub2 = y_test_IBEX[start_date2_IBEX:end_date2_IBEX]
y_test_IXIC_sub2 = y_test_IXIC[start_date2_IXIC:end_date2_IXIC]
y_test_N225_sub2 = y_test_N225[start_date2_N225:end_date2_N225]
y_test_OMXC20_sub2 = y_test_OMXC20[start_date2_OMXC20:end_date2_OMXC20]

X_test_DJI_c_sub2 = X_test_DJI_c[start_date2_DJI:end_date2_DJI]
X_test_FTSE_c_sub2 = X_test_FTSE_c[start_date2_FTSE:end_date2_FTSE]
X_test_FTSEMIB_c_sub2 = X_test_FTSEMIB_c[start_date2_FTSEMIB:end_date2_FTSEMIB]
X_test_GDAXI_c_sub2 = X_test_GDAXI_c[start_date2_GDAXI:end_date2_GDAXI]
X_test_SPX_c_sub2 = X_test_SPX_c[start_date2_SPX:end_date2_SPX]
X_test_HSI_c_sub2 = X_test_HSI_c[start_date2_HSI:end_date2_HSI]
X_test_IBEX_c_sub2 = X_test_IBEX_c[start_date2_IBEX:end_date2_IBEX]
X_test_IXIC_c_sub2 = X_test_IXIC_c[start_date2_IXIC:end_date2_IXIC]
X_test_N225_c_sub2 = X_test_N225_c[start_date2_N225:end_date2_N225]
X_test_OMXC20_c_sub2 = X_test_OMXC20_c[start_date2_OMXC20:end_date2_OMXC20]

fcHAR_DJI_1_sub2 = regHAR_DJI.predict(X_test_DJI_c_sub2)
fcHAR_FTSE_1_sub2 = regHAR_FTSE.predict(X_test_FTSE_c_sub2)
fcHAR_FTSEMIB_1_sub2 = regHAR_FTSEMIB.predict(X_test_FTSEMIB_c_sub2)
fcHAR_GDAXI_1_sub2 = regHAR_GDAXI.predict(X_test_GDAXI_c_sub2)
fcHAR_SPX_1_sub2 = regHAR_SPX.predict(X_test_SPX_c_sub2)
fcHAR_HSI_1_sub2 = regHAR_HSI.predict(X_test_HSI_c_sub2)
fcHAR_IBEX_1_sub2 = regHAR_IBEX.predict(X_test_IBEX_c_sub2)
fcHAR_IXIC_1_sub2 = regHAR_IXIC.predict(X_test_IXIC_c_sub2)
fcHAR_N225_1_sub2 = regHAR_N225.predict(X_test_N225_c_sub2)
fcHAR_OMXC20_1_sub2 = regHAR_OMXC20.predict(X_test_OMXC20_c_sub2)

# HAR MSE

MSE_HAR_1_DJI_sub2 = mean_squared_error(y_test_DJI_sub2, fcHAR_DJI_1_sub2)
MSE_HAR_1_FTSE_sub2 = mean_squared_error(y_test_FTSE_sub2, fcHAR_FTSE_1_sub2)
MSE_HAR_1_FTSEMIB_sub2 = mean_squared_error(y_test_FTSEMIB_sub2, fcHAR_FTSEMIB_1_sub2)
MSE_HAR_1_GDAXI_sub2 = mean_squared_error(y_test_GDAXI_sub2, fcHAR_GDAXI_1_sub2)
MSE_HAR_1_SPX_sub2 = mean_squared_error(y_test_SPX_sub2, fcHAR_SPX_1_sub2)
MSE_HAR_1_HSI_sub2 = mean_squared_error(y_test_HSI_sub2, fcHAR_HSI_1_sub2)
MSE_HAR_1_IBEX_sub2 = mean_squared_error(y_test_IBEX_sub2, fcHAR_IBEX_1_sub2)
MSE_HAR_1_IXIC_sub2 = mean_squared_error(y_test_IXIC_sub2, fcHAR_IXIC_1_sub2)
MSE_HAR_1_N225_sub2 = mean_squared_error(y_test_N225_sub2, fcHAR_N225_1_sub2)
MSE_HAR_1_OMXC20_sub2 = mean_squared_error(y_test_OMXC20_sub2, fcHAR_OMXC20_1_sub2)

mseDJI_HAR_sub2 = []
for i in np.arange(len(y_test_DJI_sub2)):
    mse = (fcHAR_DJI_1_sub2[i]-y_test_DJI_sub2[i])**2
    mseDJI_HAR_sub2.append(mse)
mseDJI_HAR_sub2 = np.array(mseDJI_HAR_sub2)

mseFTSE_HAR_sub2 = []
for i in np.arange(len(y_test_FTSE_sub2)):
    mse = (fcHAR_FTSE_1_sub2[i]-y_test_FTSE_sub2[i])**2
    mseFTSE_HAR_sub2.append(mse)
mseFTSE_HAR_sub2 = np.array(mseFTSE_HAR_sub2)

mseFTSEMIB_HAR_sub2 = []
for i in np.arange(len(y_test_FTSEMIB_sub2)):
    mse = (fcHAR_FTSEMIB_1_sub2[i]-y_test_FTSEMIB_sub2[i])**2
    mseFTSEMIB_HAR_sub2.append(mse)
mseFTSEMIB_HAR_sub2 = np.array(mseFTSEMIB_HAR_sub2)

mseGDAXI_HAR_sub2 = []
for i in np.arange(len(y_test_GDAXI_sub2)):
    mse = (fcHAR_GDAXI_1_sub2[i]-y_test_GDAXI_sub2[i])**2
    mseGDAXI_HAR_sub2.append(mse)
mseGDAXI_HAR_sub2 = np.array(mseGDAXI_HAR_sub2)

mseSPX_HAR_sub2 = []
for i in np.arange(len(y_test_SPX_sub2)):
    mse = (fcHAR_SPX_1_sub2[i]-y_test_SPX_sub2[i])**2
    mseSPX_HAR_sub2.append(mse)
mseSPX_HAR_sub2 = np.array(mseSPX_HAR_sub2)

mseHSI_HAR_sub2 = []
for i in np.arange(len(y_test_HSI_sub2)):
    mse = (fcHAR_HSI_1_sub2[i]-y_test_HSI_sub2[i])**2
    mseHSI_HAR_sub2.append(mse)
mseHSI_HAR_sub2 = np.array(mseHSI_HAR_sub2)

mseIBEX_HAR_sub2 = []
for i in np.arange(len(y_test_IBEX_sub2)):
    mse = (fcHAR_IBEX_1_sub2[i]-y_test_IBEX_sub2[i])**2
    mseIBEX_HAR_sub2.append(mse)
mseIBEX_HAR_sub2 = np.array(mseIBEX_HAR_sub2)

mseIXIC_HAR_sub2 = []
for i in np.arange(len(y_test_IXIC_sub2)):
    mse = (fcHAR_IXIC_1_sub2[i]-y_test_IXIC_sub2[i])**2
    mseIXIC_HAR_sub2.append(mse)
mseIXIC_HAR_sub2 = np.array(mseIXIC_HAR_sub2)

mseN225_HAR_sub2 = []
for i in np.arange(len(y_test_N225_sub2)):
    mse = (fcHAR_N225_1_sub2[i]-y_test_N225_sub2[i])**2
    mseN225_HAR_sub2.append(mse)
mseN225_HAR_sub2 = np.array(mseN225_HAR_sub2)

mseOMXC20_HAR_sub2 = []
for i in np.arange(len(y_test_OMXC20_sub2)):
    mse = (fcHAR_OMXC20_1_sub2[i]-y_test_OMXC20_sub2[i])**2
    mseOMXC20_HAR_sub2.append(mse)
mseOMXC20_HAR_sub2 = np.array(mseOMXC20_HAR_sub2)

# HAR QLIKE

# DJI

y_forecastvalues = np.array(fcHAR_DJI_1_sub2)
y_actualvalues = np.array(y_test_DJI_sub2)
qlikeDJI_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_DJI_sub2 = sum(qlikeDJI_HAR_1_sub2)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(fcHAR_FTSE_1_sub2)
y_actualvalues = np.array(y_test_FTSE_sub2)
qlikeFTSE_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_FTSE_sub2 = sum(qlikeFTSE_HAR_1_sub2)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(fcHAR_FTSEMIB_1_sub2)
y_actualvalues = np.array(y_test_FTSEMIB_sub2)
qlikeFTSEMIB_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_FTSEMIB_sub2 = sum(qlikeFTSEMIB_HAR_1_sub2)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(fcHAR_GDAXI_1_sub2)
y_actualvalues = np.array(y_test_GDAXI_sub2)
qlikeGDAXI_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_GDAXI_sub2 = sum(qlikeGDAXI_HAR_1_sub2)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(fcHAR_SPX_1_sub2)
y_actualvalues = np.array(y_test_SPX_sub2)
qlikeSPX_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_SPX_sub2 = sum(qlikeSPX_HAR_1_sub2)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(fcHAR_HSI_1_sub2)
y_actualvalues = np.array(y_test_HSI_sub2)
qlikeHSI_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_HSI_sub2 = sum(qlikeHSI_HAR_1_sub2)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(fcHAR_IBEX_1_sub2)
y_actualvalues = np.array(y_test_IBEX_sub2)
qlikeIBEX_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_IBEX_sub2 = sum(qlikeIBEX_HAR_1_sub2)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(fcHAR_IXIC_1_sub2)
y_actualvalues = np.array(y_test_IXIC_sub2)
qlikeIXIC_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_IXIC_sub2 = sum(qlikeIXIC_HAR_1_sub2)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(fcHAR_N225_1_sub2)
y_actualvalues = np.array(y_test_N225_sub2)
qlikeN225_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_N225_sub2 = sum(qlikeN225_HAR_1_sub2)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(fcHAR_OMXC20_1_sub2)
y_actualvalues = np.array(y_test_OMXC20_sub2)
qlikeOMXC20_HAR_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HAR_1_sub2.append(iteration)
QLIKE_HAR_1_OMXC20_sub2 = sum(qlikeOMXC20_HAR_1_sub2)/len(y_actualvalues)

# HARlog

X_test_DJIlog_c_sub2 = X_test_DJIlog_c[start_date2_DJI:end_date2_DJI]
X_test_FTSElog_c_sub2 = X_test_FTSElog_c[start_date2_FTSE:end_date2_FTSE]
X_test_FTSEMIBlog_c_sub2 = X_test_FTSEMIBlog_c[start_date2_FTSEMIB:end_date2_FTSEMIB]
X_test_GDAXIlog_c_sub2 = X_test_GDAXIlog_c[start_date2_GDAXI:end_date2_GDAXI]
X_test_SPXlog_c_sub2 = X_test_SPXlog_c[start_date2_SPX:end_date2_SPX]
X_test_HSIlog_c_sub2 = X_test_HSIlog_c[start_date2_HSI:end_date2_HSI]
X_test_IBEXlog_c_sub2 = X_test_IBEXlog_c[start_date2_IBEX:end_date2_IBEX]
X_test_IXIClog_c_sub2 = X_test_IXIClog_c[start_date2_IXIC:end_date2_IXIC]
X_test_N225log_c_sub2 = X_test_N225log_c[start_date2_N225:end_date2_N225]
X_test_OMXC20log_c_sub2 = X_test_OMXC20log_c[start_date2_OMXC20:end_date2_OMXC20]

fcHARlog_DJI_1_sub2 = regHARlog_DJI.predict(X_test_DJIlog_c_sub2)
fcHARlog_FTSE_1_sub2 = regHARlog_FTSE.predict(X_test_FTSElog_c_sub2)
fcHARlog_FTSEMIB_1_sub2 = regHARlog_FTSEMIB.predict(X_test_FTSEMIBlog_c_sub2)
fcHARlog_GDAXI_1_sub2 = regHARlog_GDAXI.predict(X_test_GDAXIlog_c_sub2)
fcHARlog_SPX_1_sub2 = regHARlog_SPX.predict(X_test_SPXlog_c_sub2)
fcHARlog_HSI_1_sub2 = regHARlog_HSI.predict(X_test_HSIlog_c_sub2)
fcHARlog_IBEX_1_sub2 = regHARlog_IBEX.predict(X_test_IBEXlog_c_sub2)
fcHARlog_IXIC_1_sub2 = regHARlog_IXIC.predict(X_test_IXIClog_c_sub2)
fcHARlog_N225_1_sub2 = regHARlog_N225.predict(X_test_N225log_c_sub2)
fcHARlog_OMXC20_1_sub2 = regHARlog_OMXC20.predict(X_test_OMXC20log_c_sub2)

fcHARlog_DJI_1_sub2_adj = np.exp(fcHARlog_DJI_1_sub2)*np.exp(np.var(reslogDJI)/2)
fcHARlog_FTSE_1_sub2_adj = np.exp(fcHARlog_FTSE_1_sub2)*np.exp(np.var(reslogFTSE)/2)
fcHARlog_FTSEMIB_1_sub2_adj = np.exp(fcHARlog_FTSEMIB_1_sub2)*np.exp(np.var(reslogFTSEMIB)/2)
fcHARlog_GDAXI_1_sub2_adj = np.exp(fcHARlog_GDAXI_1_sub2)*np.exp(np.var(reslogGDAXI)/2)
fcHARlog_SPX_1_sub2_adj = np.exp(fcHARlog_SPX_1_sub2)*np.exp(np.var(reslogSPX)/2)
fcHARlog_HSI_1_sub2_adj = np.exp(fcHARlog_HSI_1_sub2)*np.exp(np.var(reslogHSI)/2)
fcHARlog_IBEX_1_sub2_adj = np.exp(fcHARlog_IBEX_1_sub2)*np.exp(np.var(reslogIBEX)/2)
fcHARlog_IXIC_1_sub2_adj = np.exp(fcHARlog_IXIC_1_sub2)*np.exp(np.var(reslogIXIC)/2)
fcHARlog_N225_1_sub2_adj = np.exp(fcHARlog_N225_1_sub2)*np.exp(np.var(reslogN225)/2)
fcHARlog_OMXC20_1_sub2_adj = np.exp(fcHARlog_OMXC20_1_sub2)*np.exp(np.var(reslogOMXC20)/2)

# MSE HAR log

MSE_HARlog_1_DJI_sub2 = mean_squared_error(y_test_DJI_sub2, fcHARlog_DJI_1_sub2_adj)
MSE_HARlog_1_FTSE_sub2 = mean_squared_error(y_test_FTSE_sub2, fcHARlog_FTSE_1_sub2_adj)
MSE_HARlog_1_FTSEMIB_sub2 = mean_squared_error(y_test_FTSEMIB_sub2, fcHARlog_FTSEMIB_1_sub2_adj)
MSE_HARlog_1_GDAXI_sub2 = mean_squared_error(y_test_GDAXI_sub2, fcHARlog_GDAXI_1_sub2_adj)
MSE_HARlog_1_SPX_sub2 = mean_squared_error(y_test_SPX_sub2, fcHARlog_SPX_1_sub2_adj)
MSE_HARlog_1_HSI_sub2 = mean_squared_error(y_test_HSI_sub2, fcHARlog_HSI_1_sub2_adj)
MSE_HARlog_1_IBEX_sub2 = mean_squared_error(y_test_IBEX_sub2, fcHARlog_IBEX_1_sub2_adj)
MSE_HARlog_1_IXIC_sub2 = mean_squared_error(y_test_IXIC_sub2, fcHARlog_IXIC_1_sub2_adj)
MSE_HARlog_1_N225_sub2 = mean_squared_error(y_test_N225_sub2, fcHARlog_N225_1_sub2_adj)
MSE_HARlog_1_OMXC20_sub2 = mean_squared_error(y_test_OMXC20_sub2, fcHARlog_OMXC20_1_sub2_adj)

mseDJI_HARlog_sub2 = []
for i in np.arange(len(y_test_DJI_sub2)):
    mse = (fcHARlog_DJI_1_sub2_adj[i]-y_test_DJI_sub2[i])**2
    mseDJI_HARlog_sub2.append(mse)
mseDJI_HARlog_sub2 = np.array(mseDJI_HARlog_sub2)

mseFTSE_HARlog_sub2 = []
for i in np.arange(len(y_test_FTSE_sub2)):
    mse = (fcHARlog_FTSE_1_sub2_adj[i]-y_test_FTSE_sub2[i])**2
    mseFTSE_HARlog_sub2.append(mse)
mseFTSE_HARlog_sub2 = np.array(mseFTSE_HARlog_sub2)

mseFTSEMIB_HARlog_sub2 = []
for i in np.arange(len(y_test_FTSEMIB_sub2)):
    mse = (fcHARlog_FTSEMIB_1_sub2_adj[i]-y_test_FTSEMIB_sub2[i])**2
    mseFTSEMIB_HARlog_sub2.append(mse)
mseFTSEMIB_HARlog_sub2 = np.array(mseFTSEMIB_HARlog_sub2)

mseGDAXI_HARlog_sub2 = []
for i in np.arange(len(y_test_GDAXI_sub2)):
    mse = (fcHARlog_GDAXI_1_sub2_adj[i]-y_test_GDAXI_sub2[i])**2
    mseGDAXI_HARlog_sub2.append(mse)
mseGDAXI_HARlog_sub2 = np.array(mseGDAXI_HARlog_sub2)

mseSPX_HARlog_sub2 = []
for i in np.arange(len(y_test_SPX_sub2)):
    mse = (fcHARlog_SPX_1_sub2_adj[i]-y_test_SPX_sub2[i])**2
    mseSPX_HARlog_sub2.append(mse)
mseSPX_HARlog_sub2 = np.array(mseSPX_HARlog_sub2)

mseHSI_HARlog_sub2 = []
for i in np.arange(len(y_test_HSI_sub2)):
    mse = (fcHARlog_HSI_1_sub2_adj[i]-y_test_HSI_sub2[i])**2
    mseHSI_HARlog_sub2.append(mse)
mseHSI_HARlog_sub2 = np.array(mseHSI_HARlog_sub2)

mseIBEX_HARlog_sub2 = []
for i in np.arange(len(y_test_IBEX_sub2)):
    mse = (fcHARlog_IBEX_1_sub2_adj[i]-y_test_IBEX_sub2[i])**2
    mseIBEX_HARlog_sub2.append(mse)
mseIBEX_HARlog_sub2 = np.array(mseIBEX_HARlog_sub2)

mseIXIC_HARlog_sub2 = []
for i in np.arange(len(y_test_IXIC_sub2)):
    mse = (fcHARlog_IXIC_1_sub2_adj[i]-y_test_IXIC_sub2[i])**2
    mseIXIC_HARlog_sub2.append(mse)
mseIXIC_HARlog_sub2 = np.array(mseIXIC_HARlog_sub2)

mseN225_HARlog_sub2 = []
for i in np.arange(len(y_test_N225_sub2)):
    mse = (fcHARlog_N225_1_sub2_adj[i]-y_test_N225_sub2[i])**2
    mseN225_HARlog_sub2.append(mse)
mseN225_HARlog_sub2 = np.array(mseN225_HARlog_sub2)

mseOMXC20_HARlog_sub2 = []
for i in np.arange(len(y_test_OMXC20_sub2)):
    mse = (fcHARlog_OMXC20_1_sub2_adj[i]-y_test_OMXC20_sub2[i])**2
    mseOMXC20_HARlog_sub2.append(mse)
mseOMXC20_HARlog_sub2 = np.array(mseOMXC20_HARlog_sub2)

# QLIKE HARlog

# DJI

y_forecastvalues = np.array(fcHARlog_DJI_1_sub2_adj)
y_actualvalues = np.array(y_test_DJI_sub2)
qlikeDJI_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_DJI_sub2 = sum(qlikeDJI_HARlog_1_sub2)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(fcHARlog_FTSE_1_sub2_adj)
y_actualvalues = np.array(y_test_FTSE_sub2)
qlikeFTSE_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_FTSE_sub2 = sum(qlikeFTSE_HARlog_1_sub2)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(fcHARlog_FTSEMIB_1_sub2_adj)
y_actualvalues = np.array(y_test_FTSEMIB_sub2)
qlikeFTSEMIB_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_FTSEMIB_sub2 = sum(qlikeFTSEMIB_HARlog_1_sub2)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(fcHARlog_GDAXI_1_sub2_adj)
y_actualvalues = np.array(y_test_GDAXI_sub2)
qlikeGDAXI_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_GDAXI_sub2 = sum(qlikeGDAXI_HARlog_1_sub2)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(fcHARlog_SPX_1_sub2_adj)
y_actualvalues = np.array(y_test_SPX_sub2)
qlikeSPX_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_SPX_sub2 = sum(qlikeSPX_HARlog_1_sub2)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(fcHARlog_HSI_1_sub2_adj)
y_actualvalues = np.array(y_test_HSI_sub2)
qlikeHSI_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_HSI_sub2 = sum(qlikeHSI_HARlog_1_sub2)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(fcHARlog_IBEX_1_sub2_adj)
y_actualvalues = np.array(y_test_IBEX_sub2)
qlikeIBEX_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_IBEX_sub2 = sum(qlikeIBEX_HARlog_1_sub2)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(fcHARlog_IXIC_1_sub2_adj)
y_actualvalues = np.array(y_test_IXIC_sub2)
qlikeIXIC_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_IXIC_sub2 = sum(qlikeIXIC_HARlog_1_sub2)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(fcHARlog_N225_1_sub2_adj)
y_actualvalues = np.array(y_test_N225_sub2)
qlikeN225_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_N225_sub2 = sum(qlikeN225_HARlog_1_sub2)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(fcHARlog_OMXC20_1_sub2_adj)
y_actualvalues = np.array(y_test_OMXC20_sub2)
qlikeOMXC20_HARlog_1_sub2 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HARlog_1_sub2.append(iteration)
QLIKE_HARlog_1_OMXC20_sub2 = sum(qlikeOMXC20_HARlog_1_sub2)/len(y_actualvalues)

#%% 

arrays_Out = {
    'mse_AR1': {
        'DJI': mseDJI_AR1_sub2,
        'FTSE': mseFTSE_AR1_sub2,
        'FTSEMIB': mseFTSEMIB_AR1_sub2,
        'GDAXI': mseGDAXI_AR1_sub2,
        'SPX': mseSPX_AR1_sub2,
        'HSI': mseHSI_AR1_sub2,
        'IBEX': mseIBEX_AR1_sub2,
        'IXIC': mseIXIC_AR1_sub2,
        'N225': mseN225_AR1_sub2,
        'OMXC20': mseOMXC20_AR1_sub2
    },
    'mse_HAR': {
        'DJI': mseDJI_HAR_sub2,
        'FTSE': mseFTSE_HAR_sub2,
        'FTSEMIB': mseFTSEMIB_HAR_sub2,
        'GDAXI': mseGDAXI_HAR_sub2,
        'SPX': mseSPX_HAR_sub2,
        'HSI': mseHSI_HAR_sub2,
        'IBEX': mseIBEX_HAR_sub2,
        'IXIC': mseIXIC_HAR_sub2,
        'N225': mseN225_HAR_sub2,
        'OMXC20': mseOMXC20_HAR_sub2
    },
    'mse_HARlog': {
        'DJI': mseDJI_HARlog_sub2,
        'FTSE': mseFTSE_HARlog_sub2,
        'FTSEMIB': mseFTSEMIB_HARlog_sub2,
        'GDAXI': mseGDAXI_HARlog_sub2,
        'SPX': mseSPX_HARlog_sub2,
        'HSI': mseHSI_HARlog_sub2,
        'IBEX': mseIBEX_HARlog_sub2,
        'IXIC': mseIXIC_HARlog_sub2,
        'N225': mseN225_HARlog_sub2,
        'OMXC20': mseOMXC20_HARlog_sub2
                    }
    }


for k1 in arrays_Out:
    if k1 == 'mse_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_AR1_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HAR_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HARlog_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%%

arrays_Out = {
    'qlike_AR1': {
        'DJI': qlikeDJI_AR1_1_sub2,
        'FTSE': qlikeFTSE_AR1_1_sub2,
        'FTSEMIB': qlikeFTSEMIB_AR1_1_sub2,
        'GDAXI': qlikeGDAXI_AR1_1_sub2,
        'SPX': qlikeSPX_AR1_1_sub2,
        'HSI': qlikeHSI_AR1_1_sub2,
        'IBEX': qlikeIBEX_AR1_1_sub2,
        'IXIC': qlikeIXIC_AR1_1_sub2,
        'N225': qlikeN225_AR1_1_sub2,
        'OMXC20': qlikeOMXC20_AR1_1_sub2
    },
    'qlike_HAR': {
        'DJI': qlikeDJI_HAR_1_sub2,
        'FTSE': qlikeFTSE_HAR_1_sub2,
        'FTSEMIB': qlikeFTSEMIB_HAR_1_sub2,
        'GDAXI': qlikeGDAXI_HAR_1_sub2,
        'SPX': qlikeSPX_HAR_1_sub2,
        'HSI': qlikeHSI_HAR_1_sub2,
        'IBEX': qlikeIBEX_HAR_1_sub2,
        'IXIC': qlikeIXIC_HAR_1_sub2,
        'N225': qlikeN225_HAR_1_sub2,
        'OMXC20': qlikeOMXC20_HAR_1_sub2
    },
    'qlike_HARlog': {
        'DJI': qlikeDJI_HARlog_1_sub2, 
        'FTSE': qlikeFTSE_HARlog_1_sub2,
        'FTSEMIB': qlikeFTSEMIB_HARlog_1_sub2,
        'GDAXI': qlikeGDAXI_HARlog_1_sub2,
        'SPX': qlikeSPX_HARlog_1_sub2,
        'HSI': qlikeHSI_HARlog_1_sub2,
        'IBEX': qlikeIBEX_HARlog_1_sub2,
        'IXIC': qlikeIXIC_HARlog_1_sub2,
        'N225': qlikeN225_HARlog_1_sub2,
        'OMXC20': qlikeOMXC20_HARlog_1_sub2
                    }
    }


for k1 in arrays_Out:
    if k1 == 'qlike_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_AR1_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HAR_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HARlog_sub2.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')    