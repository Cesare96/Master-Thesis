# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:33:44 2023

@author: cesar
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

#%% Subsample for AR1

start_date_DJI = '2015-09-14'
end_date_DJI = '2020-02-21' 
start_date_DJI = pd.to_datetime(start_date_DJI)
end_date_DJI = pd.to_datetime(end_date_DJI)
start_date_DJI_X = '2015-09-13'
end_date_DJI_X = '2020-02-20'
start_date_DJI_X = pd.to_datetime(start_date_DJI_X)
end_date_DJI_X = pd.to_datetime(end_date_DJI_X)
y_test_AR1_DJI_sub1 = DJI.loc[start_date_DJI:end_date_DJI]

start_date_FTSE = '2015-09-24'
end_date_FTSE = '2020-02-24' 
start_date_FTSE = pd.to_datetime(start_date_FTSE)
end_date_FTSE = pd.to_datetime(end_date_FTSE)
start_date_FTSE_X = '2015-09-23'
end_date_FTSE_X = '2020-02-21'
start_date_FTSE_X = pd.to_datetime(start_date_FTSE_X)
end_date_FTSE_X = pd.to_datetime(end_date_FTSE_X)
y_test_AR1_FTSE_sub1 = FTSE.loc[start_date_FTSE:end_date_FTSE]

start_date_FTSEMIB = '2018-07-09'
end_date_FTSEMIB = '2020-02-21' 
start_date_FTSEMIB = pd.to_datetime(start_date_FTSEMIB)
end_date_FTSEMIB = pd.to_datetime(end_date_FTSEMIB)
start_date_FTSEMIB_X = '2018-07-08'
end_date_FTSEMIB_X = '2020-02-20'
start_date_FTSEMIB_X = pd.to_datetime(start_date_FTSEMIB_X)
end_date_FTSEMIB_X = pd.to_datetime(end_date_FTSEMIB_X)
y_test_AR1_FTSEMIB_sub1 = FTSEMIB.loc[start_date_FTSEMIB:end_date_FTSEMIB]

start_date_GDAXI = '2015-09-09'
end_date_GDAXI = '2020-02-25' 
start_date_GDAXI = pd.to_datetime(start_date_GDAXI)
end_date_GDAXI = pd.to_datetime(end_date_GDAXI)
start_date_GDAXI_X = '2015-09-08'
end_date_GDAXI_X = '2020-02-24'
start_date_GDAXI_X = pd.to_datetime(start_date_GDAXI_X)
end_date_GDAXI_X = pd.to_datetime(end_date_GDAXI_X)
y_test_AR1_GDAXI_sub1 = GDAXI.loc[start_date_GDAXI:end_date_GDAXI]

start_date_SPX = '2015-09-17'
end_date_SPX = '2020-02-25' 
start_date_SPX = pd.to_datetime(start_date_SPX)
end_date_SPX = pd.to_datetime(end_date_SPX)
start_date_SPX_X = '2015-09-16'
end_date_SPX_X = '2020-02-24'
start_date_SPX_X = pd.to_datetime(start_date_SPX_X)
end_date_SPX_X = pd.to_datetime(end_date_SPX_X)
y_test_AR1_SPX_sub1 = SPX.loc[start_date_SPX:end_date_SPX]

start_date_HSI = '2015-09-15'
end_date_HSI = '2020-03-09' 
start_date_HSI = pd.to_datetime(start_date_HSI)
end_date_HSI = pd.to_datetime(end_date_HSI)
start_date_HSI_X = '2015-09-14'
end_date_HSI_X = '2020-03-06'
start_date_HSI_X = pd.to_datetime(start_date_HSI_X)
end_date_HSI_X = pd.to_datetime(end_date_HSI_X)
y_test_AR1_HSI_sub1 = HSI.loc[start_date_HSI:end_date_HSI]

start_date_IBEX = '2015-10-05'
end_date_IBEX = '2020-02-26' 
start_date_IBEX = pd.to_datetime(start_date_IBEX)
end_date_IBEX = pd.to_datetime(end_date_IBEX)
start_date_IBEX_X = '2015-10-04'
end_date_IBEX_X = '2020-02-25'
start_date_IBEX_X = pd.to_datetime(start_date_IBEX_X)
end_date_IBEX_X = pd.to_datetime(end_date_IBEX_X)
y_test_AR1_IBEX_sub1 = IBEX.loc[start_date_IBEX:end_date_IBEX]

start_date_IXIC = '2015-09-23'
end_date_IXIC = '2020-02-21' 
start_date_IXIC = pd.to_datetime(start_date_IXIC)
end_date_IXIC = pd.to_datetime(end_date_IXIC)
start_date_IXIC_X = '2015-09-22'
end_date_IXIC_X = '2020-02-20'
start_date_IXIC_X = pd.to_datetime(start_date_IXIC_X)
end_date_IXIC_X = pd.to_datetime(end_date_IXIC_X)
y_test_AR1_IXIC_sub1 = IXIC.loc[start_date_IXIC:end_date_IXIC]

start_date_N225 = '2015-09-10'
end_date_N225 = '2020-02-21' 
start_date_N225 = pd.to_datetime(start_date_N225)
end_date_N225 = pd.to_datetime(end_date_N225)
start_date_N225_X = '2015-09-09'
end_date_N225_X = '2020-02-20'
start_date_N225_X = pd.to_datetime(start_date_N225_X)
end_date_N225_X = pd.to_datetime(end_date_N225_X)
y_test_AR1_N225_sub1 = N225.loc[start_date_N225:end_date_N225]

start_date_OMXC20 = '2017-06-11'
end_date_OMXC20 = '2020-02-21' 
start_date_OMXC20 = pd.to_datetime(start_date_OMXC20)
end_date_OMXC20 = pd.to_datetime(end_date_OMXC20)
start_date_OMXC20_X = '2017-06-08'
end_date_OMXC20_X = '2020-02-20'
start_date_OMXC20_X = pd.to_datetime(start_date_OMXC20_X)
end_date_OMXC20_X = pd.to_datetime(end_date_OMXC20_X)
y_test_AR1_OMXC20_sub1 = OMXC20.loc[start_date_OMXC20:end_date_OMXC20]

#%% AR(1), HAR and HARlog with the subsample1

# AR(1) forecast

X_test_AR1_sub1_DJI = DJI.loc[start_date_DJI_X:end_date_DJI_X]
X_test_AR1_sub1_FTSE = FTSE.loc[start_date_FTSE_X:end_date_FTSE_X]
X_test_AR1_sub1_FTSEMIB = FTSEMIB.loc[start_date_FTSEMIB_X:end_date_FTSEMIB_X]
X_test_AR1_sub1_GDAXI = GDAXI.loc[start_date_GDAXI_X:end_date_GDAXI_X]
X_test_AR1_sub1_SPX = SPX.loc[start_date_SPX_X:end_date_SPX_X]
X_test_AR1_sub1_HSI = HSI.loc[start_date_HSI_X:end_date_HSI_X]
X_test_AR1_sub1_IBEX = IBEX.loc[start_date_IBEX_X:end_date_IBEX_X]
X_test_AR1_sub1_IXIC = IXIC.loc[start_date_IXIC_X:end_date_IXIC_X]
X_test_AR1_sub1_N225 = N225.loc[start_date_N225_X:end_date_N225_X]
X_test_AR1_sub1_OMXC20 = OMXC20.loc[start_date_OMXC20_X:end_date_OMXC20_X]

X_test_AR1_sub1_DJI_c = sm.add_constant(X_test_AR1_sub1_DJI)
X_test_AR1_sub1_FTSE_c = sm.add_constant(X_test_AR1_sub1_FTSE)
X_test_AR1_sub1_FTSEMIB_c = sm.add_constant(X_test_AR1_sub1_FTSEMIB)
X_test_AR1_sub1_GDAXI_c = sm.add_constant(X_test_AR1_sub1_GDAXI)
X_test_AR1_sub1_SPX_c = sm.add_constant(X_test_AR1_sub1_SPX)
X_test_AR1_sub1_HSI_c = sm.add_constant(X_test_AR1_sub1_HSI)
X_test_AR1_sub1_IBEX_c = sm.add_constant(X_test_AR1_sub1_IBEX)
X_test_AR1_sub1_IXIC_c = sm.add_constant(X_test_AR1_sub1_IXIC)
X_test_AR1_sub1_N225_c = sm.add_constant(X_test_AR1_sub1_N225)
X_test_AR1_sub1_OMXC20_c = sm.add_constant(X_test_AR1_sub1_OMXC20)

y_pred_AR1_1_DJI_sub1 = model_AR1_DJI.predict(X_test_AR1_sub1_DJI_c)
y_pred_AR1_1_FTSE_sub1 = model_AR1_FTSE.predict(X_test_AR1_sub1_FTSE_c)
y_pred_AR1_1_FTSEMIB_sub1 = model_AR1_FTSEMIB.predict(X_test_AR1_sub1_FTSEMIB_c)
y_pred_AR1_1_GDAXI_sub1 = model_AR1_GDAXI.predict(X_test_AR1_sub1_GDAXI_c)
y_pred_AR1_1_SPX_sub1 = model_AR1_SPX.predict(X_test_AR1_sub1_SPX_c)
y_pred_AR1_1_HSI_sub1 = model_AR1_HSI.predict(X_test_AR1_sub1_HSI_c)
y_pred_AR1_1_IBEX_sub1 = model_AR1_IBEX.predict(X_test_AR1_sub1_IBEX_c)
y_pred_AR1_1_IXIC_sub1 = model_AR1_IXIC.predict(X_test_AR1_sub1_IXIC_c)
y_pred_AR1_1_N225_sub1 = model_AR1_N225.predict(X_test_AR1_sub1_N225_c)
y_pred_AR1_1_OMXC20_sub1 = model_AR1_OMXC20.predict(X_test_AR1_sub1_OMXC20_c)

# AR(1) MSE

MSE_AR1_1_DJI_sub1 = mean_squared_error(y_test_AR1_DJI_sub1, y_pred_AR1_1_DJI_sub1)
MSE_AR1_1_FTSE_sub1 = mean_squared_error(y_test_AR1_FTSE_sub1, y_pred_AR1_1_FTSE_sub1)
MSE_AR1_1_FTSEMIB_sub1 = mean_squared_error(y_test_AR1_FTSEMIB_sub1, y_pred_AR1_1_FTSEMIB_sub1)
MSE_AR1_1_GDAXI_sub1 = mean_squared_error(y_test_AR1_GDAXI_sub1, y_pred_AR1_1_GDAXI_sub1)
MSE_AR1_1_SPX_sub1 = mean_squared_error(y_test_AR1_SPX_sub1, y_pred_AR1_1_SPX_sub1)
MSE_AR1_1_HSI_sub1 = mean_squared_error(y_test_AR1_HSI_sub1, y_pred_AR1_1_HSI_sub1)
MSE_AR1_1_IBEX_sub1 = mean_squared_error(y_test_AR1_IBEX_sub1, y_pred_AR1_1_IBEX_sub1)
MSE_AR1_1_IXIC_sub1 = mean_squared_error(y_test_AR1_IXIC_sub1, y_pred_AR1_1_IXIC_sub1)
MSE_AR1_1_N225_sub1 = mean_squared_error(y_test_AR1_N225_sub1, y_pred_AR1_1_N225_sub1)
MSE_AR1_1_OMXC20_sub1 = mean_squared_error(y_test_AR1_OMXC20_sub1, y_pred_AR1_1_OMXC20_sub1)

mseDJI_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_DJI_sub1)):
    mse = (y_pred_AR1_1_DJI_sub1[i]-y_test_AR1_DJI_sub1[i])**2
    mseDJI_AR1_sub1.append(mse)
mseDJI_AR1_sub1 = np.array(mseDJI_AR1_sub1)

mseFTSE_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_FTSE_sub1)):
    mse = (y_pred_AR1_1_FTSE_sub1[i]-y_test_AR1_FTSE_sub1[i])**2
    mseFTSE_AR1_sub1.append(mse)
mseFTSE_AR1_sub1 = np.array(mseFTSE_AR1_sub1)

mseFTSEMIB_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_FTSEMIB_sub1)):
    mse = (y_pred_AR1_1_FTSEMIB_sub1[i]-y_test_AR1_FTSEMIB_sub1[i])**2
    mseFTSEMIB_AR1_sub1.append(mse)
mseFTSEMIB_AR1_sub1 = np.array(mseFTSEMIB_AR1_sub1)

mseGDAXI_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_GDAXI_sub1)):
    mse = (y_pred_AR1_1_GDAXI_sub1[i]-y_test_AR1_GDAXI_sub1[i])**2
    mseGDAXI_AR1_sub1.append(mse)
mseGDAXI_AR1_sub1 = np.array(mseGDAXI_AR1_sub1)

mseSPX_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_SPX_sub1)):
    mse = (y_pred_AR1_1_SPX_sub1[i]-y_test_AR1_SPX_sub1[i])**2
    mseSPX_AR1_sub1.append(mse)
mseSPX_AR1_sub1 = np.array(mseSPX_AR1_sub1)

mseHSI_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_HSI_sub1)):
    mse = (y_pred_AR1_1_HSI_sub1[i]-y_test_AR1_HSI_sub1[i])**2
    mseHSI_AR1_sub1.append(mse)
mseHSI_AR1_sub1 = np.array(mseHSI_AR1_sub1)

mseIBEX_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_IBEX_sub1)):
    mse = (y_pred_AR1_1_IBEX_sub1[i]-y_test_AR1_IBEX_sub1[i])**2
    mseIBEX_AR1_sub1.append(mse)
mseIBEX_AR1_sub1 = np.array(mseIBEX_AR1_sub1)

mseIXIC_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_IXIC_sub1)):
    mse = (y_pred_AR1_1_IXIC_sub1[i]-y_test_AR1_IXIC_sub1[i])**2
    mseIXIC_AR1_sub1.append(mse)
mseIXIC_AR1_sub1 = np.array(mseIXIC_AR1_sub1)

mseN225_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_N225_sub1)):
    mse = (y_pred_AR1_1_N225_sub1[i]-y_test_AR1_N225_sub1[i])**2
    mseN225_AR1_sub1.append(mse)
mseN225_AR1_sub1 = np.array(mseN225_AR1_sub1)

mseOMXC20_AR1_sub1 = []
for i in np.arange(len(y_test_AR1_OMXC20_sub1)):
    mse = (y_pred_AR1_1_OMXC20_sub1[i]-y_test_AR1_OMXC20_sub1[i])**2
    mseOMXC20_AR1_sub1.append(mse)
mseOMXC20_AR1_sub1 = np.array(mseOMXC20_AR1_sub1)
    

# AR(1) QLIKE 

# DJI 

y_forecastvalues = np.array(y_pred_AR1_1_DJI_sub1)
y_actualvalues = np.array(y_test_AR1_DJI_sub1)
qlikeDJI_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_DJI_sub1 = sum(qlikeDJI_AR1_1_sub1)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(y_pred_AR1_1_FTSE_sub1)
y_actualvalues = np.array(y_test_AR1_FTSE_sub1)
qlikeFTSE_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_FTSE_sub1 = sum(qlikeFTSE_AR1_1_sub1)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(y_pred_AR1_1_FTSEMIB_sub1)
y_actualvalues = np.array(y_test_AR1_FTSEMIB_sub1)
qlikeFTSEMIB_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_FTSEMIB_sub1 = sum(qlikeFTSEMIB_AR1_1_sub1)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(y_pred_AR1_1_GDAXI_sub1)
y_actualvalues = np.array(y_test_AR1_GDAXI_sub1)
qlikeGDAXI_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_GDAXI_sub1 = sum(qlikeGDAXI_AR1_1_sub1)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(y_pred_AR1_1_SPX_sub1)
y_actualvalues = np.array(y_test_AR1_SPX_sub1)
qlikeSPX_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_SPX_sub1 = sum(qlikeSPX_AR1_1_sub1)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(y_pred_AR1_1_HSI_sub1)
y_actualvalues = np.array(y_test_AR1_HSI_sub1)
qlikeHSI_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_HSI_sub1 = sum(qlikeHSI_AR1_1_sub1)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(y_pred_AR1_1_IBEX_sub1)
y_actualvalues = np.array(y_test_AR1_IBEX_sub1)
qlikeIBEX_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_IBEX_sub1 = sum(qlikeIBEX_AR1_1_sub1)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(y_pred_AR1_1_IXIC_sub1)
y_actualvalues = np.array(y_test_AR1_IXIC_sub1)
qlikeIXIC_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_IXIC_sub1 = sum(qlikeIXIC_AR1_1_sub1)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(y_pred_AR1_1_N225_sub1)
y_actualvalues = np.array(y_test_AR1_N225_sub1)
qlikeN225_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_N225_sub1 = sum(qlikeN225_AR1_1_sub1)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(y_pred_AR1_1_OMXC20_sub1)
y_actualvalues = np.array(y_test_AR1_OMXC20_sub1)
qlikeOMXC20_AR1_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_AR1_1_sub1.append(iteration)
QLIKE_AR1_1_OMXC20_sub1 = sum(qlikeOMXC20_AR1_1_sub1)/len(y_actualvalues)

# HAR 

X_test_DJI_c_sub1 = X_test_DJI_c.loc[start_date_DJI:end_date_DJI]
y_test_sub1_DJI_HAR = DJI.loc[start_date_DJI:end_date_DJI]

X_test_FTSE_c_sub1 = X_test_FTSE_c.loc[start_date_FTSE:end_date_FTSE]
y_test_sub1_FTSE_HAR = FTSE.loc[start_date_FTSE:end_date_FTSE]

X_test_FTSEMIB_c_sub1 = X_test_FTSEMIB_c.loc[start_date_FTSEMIB:end_date_FTSEMIB]
y_test_sub1_FTSEMIB_HAR = FTSEMIB.loc[start_date_FTSEMIB:end_date_FTSEMIB]

X_test_GDAXI_c_sub1 = X_test_GDAXI_c.loc[start_date_GDAXI:end_date_GDAXI]
y_test_sub1_GDAXI_HAR = GDAXI.loc[start_date_GDAXI:end_date_GDAXI]

X_test_SPX_c_sub1 = X_test_SPX_c.loc[start_date_SPX:end_date_SPX]
y_test_sub1_SPX_HAR = SPX.loc[start_date_SPX:end_date_SPX]

X_test_HSI_c_sub1 = X_test_HSI_c.loc[start_date_HSI:end_date_HSI]
y_test_sub1_HSI_HAR = HSI.loc[start_date_HSI:end_date_HSI]

X_test_IBEX_c_sub1 = X_test_IBEX_c.loc[start_date_IBEX:end_date_IBEX]
y_test_sub1_IBEX_HAR = IBEX.loc[start_date_IBEX:end_date_IBEX]

X_test_IXIC_c_sub1 = X_test_IXIC_c.loc[start_date_IXIC:end_date_IXIC]
y_test_sub1_IXIC_HAR = IXIC.loc[start_date_IXIC:end_date_IXIC]

X_test_N225_c_sub1 = X_test_N225_c.loc[start_date_N225:end_date_N225]
y_test_sub1_N225_HAR = N225.loc[start_date_N225:end_date_N225]

X_test_OMXC20_c_sub1 = X_test_OMXC20_c.loc[start_date_OMXC20:end_date_OMXC20]
y_test_sub1_OMXC20_HAR = OMXC20.loc[start_date_OMXC20:end_date_OMXC20]

fcHAR_DJI_1_sub1 = regHAR_DJI.predict(X_test_DJI_c_sub1)
fcHAR_FTSE_1_sub1 = regHAR_FTSE.predict(X_test_FTSE_c_sub1)
fcHAR_FTSEMIB_1_sub1 = regHAR_FTSEMIB.predict(X_test_FTSEMIB_c_sub1)
fcHAR_GDAXI_1_sub1 = regHAR_GDAXI.predict(X_test_GDAXI_c_sub1)
fcHAR_SPX_1_sub1 = regHAR_SPX.predict(X_test_SPX_c_sub1)
fcHAR_HSI_1_sub1 = regHAR_HSI.predict(X_test_HSI_c_sub1)
fcHAR_IBEX_1_sub1 = regHAR_IBEX.predict(X_test_IBEX_c_sub1)
fcHAR_IXIC_1_sub1 = regHAR_IXIC.predict(X_test_IXIC_c_sub1)
fcHAR_N225_1_sub1 = regHAR_N225.predict(X_test_N225_c_sub1)
fcHAR_OMXC20_1_sub1 = regHAR_OMXC20.predict(X_test_OMXC20_c_sub1)

# HAR MSE

MSE_HAR_1_DJI_sub1 = mean_squared_error(y_test_sub1_DJI_HAR, fcHAR_DJI_1_sub1)
MSE_HAR_1_FTSE_sub1 = mean_squared_error(y_test_sub1_FTSE_HAR, fcHAR_FTSE_1_sub1)
MSE_HAR_1_FTSEMIB_sub1 = mean_squared_error(y_test_sub1_FTSEMIB_HAR, fcHAR_FTSEMIB_1_sub1)
MSE_HAR_1_GDAXI_sub1 = mean_squared_error(y_test_sub1_GDAXI_HAR, fcHAR_GDAXI_1_sub1)
MSE_HAR_1_SPX_sub1 = mean_squared_error(y_test_sub1_SPX_HAR, fcHAR_SPX_1_sub1)
MSE_HAR_1_HSI_sub1 = mean_squared_error(y_test_sub1_HSI_HAR, fcHAR_HSI_1_sub1)
MSE_HAR_1_IBEX_sub1 = mean_squared_error(y_test_sub1_IBEX_HAR, fcHAR_IBEX_1_sub1)
MSE_HAR_1_IXIC_sub1 = mean_squared_error(y_test_sub1_IXIC_HAR, fcHAR_IXIC_1_sub1)
MSE_HAR_1_N225_sub1 = mean_squared_error(y_test_sub1_N225_HAR, fcHAR_N225_1_sub1)
MSE_HAR_1_OMXC20_sub1 = mean_squared_error(y_test_sub1_OMXC20_HAR, fcHAR_OMXC20_1_sub1)

mseDJI_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_DJI_HAR)):
    mse = (fcHAR_DJI_1_sub1[i]-y_test_sub1_DJI_HAR[i])**2
    mseDJI_HAR_sub1.append(mse)
mseDJI_HAR_sub1 = np.array(mseDJI_HAR_sub1)

mseFTSE_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_FTSE_HAR)):
    mse = (fcHAR_FTSE_1_sub1[i]-y_test_sub1_FTSE_HAR[i])**2
    mseFTSE_HAR_sub1.append(mse)
mseFTSE_HAR_sub1 = np.array(mseFTSE_HAR_sub1)

mseFTSEMIB_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_FTSEMIB_HAR)):
    mse = (fcHAR_FTSEMIB_1_sub1[i]-y_test_sub1_FTSEMIB_HAR[i])**2
    mseFTSEMIB_HAR_sub1.append(mse)
mseFTSEMIB_HAR_sub1 = np.array(mseFTSEMIB_HAR_sub1)

mseGDAXI_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_GDAXI_HAR)):
    mse = (fcHAR_GDAXI_1_sub1[i]-y_test_sub1_GDAXI_HAR[i])**2
    mseGDAXI_HAR_sub1.append(mse)
mseGDAXI_HAR_sub1 = np.array(mseGDAXI_HAR_sub1)

mseSPX_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_SPX_HAR)):
    mse = (fcHAR_SPX_1_sub1[i]-y_test_sub1_SPX_HAR[i])**2
    mseSPX_HAR_sub1.append(mse)
mseSPX_HAR_sub1 = np.array(mseSPX_HAR_sub1)

mseHSI_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_HSI_HAR)):
    mse = (fcHAR_HSI_1_sub1[i]-y_test_sub1_HSI_HAR[i])**2
    mseHSI_HAR_sub1.append(mse)
mseHSI_HAR_sub1 = np.array(mseHSI_HAR_sub1)

mseIBEX_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_IBEX_HAR)):
    mse = (fcHAR_IBEX_1_sub1[i]-y_test_sub1_IBEX_HAR[i])**2
    mseIBEX_HAR_sub1.append(mse)
mseIBEX_HAR_sub1 = np.array(mseIBEX_HAR_sub1)

mseIXIC_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_IXIC_HAR)):
    mse = (fcHAR_IXIC_1_sub1[i]-y_test_sub1_IXIC_HAR[i])**2
    mseIXIC_HAR_sub1.append(mse)
mseIXIC_HAR_sub1 = np.array(mseIXIC_HAR_sub1)

mseN225_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_N225_HAR)):
    mse = (fcHAR_N225_1_sub1[i]-y_test_sub1_N225_HAR[i])**2
    mseN225_HAR_sub1.append(mse)
mseN225_HAR_sub1 = np.array(mseN225_HAR_sub1)

mseOMXC20_HAR_sub1 = []
for i in np.arange(len(y_test_sub1_OMXC20_HAR)):
    mse = (fcHAR_OMXC20_1_sub1[i]-y_test_sub1_OMXC20_HAR[i])**2
    mseOMXC20_HAR_sub1.append(mse)
mseOMXC20_HAR_sub1 = np.array(mseOMXC20_HAR_sub1)

# HAR QLIKE

# DJI

y_forecastvalues = np.array(fcHAR_DJI_1_sub1)
y_actualvalues = np.array(y_test_sub1_DJI_HAR)
qlikeDJI_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeDJI_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_DJI_sub1 = sum(qlikeDJI_HAR_1_sub1) / len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(fcHAR_FTSE_1_sub1)
y_actualvalues = np.array(y_test_sub1_FTSE_HAR)
qlikeFTSE_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeFTSE_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_FTSE_sub1 = sum(qlikeFTSE_HAR_1_sub1) / len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(fcHAR_FTSEMIB_1_sub1)
y_actualvalues = np.array(y_test_sub1_FTSEMIB_HAR)
qlikeFTSEMIB_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_FTSEMIB_sub1 = sum(qlikeFTSEMIB_HAR_1_sub1) / len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(fcHAR_GDAXI_1_sub1)
y_actualvalues = np.array(y_test_sub1_GDAXI_HAR)
qlikeGDAXI_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeGDAXI_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_GDAXI_sub1 = sum(qlikeGDAXI_HAR_1_sub1) / len(y_actualvalues)

# SPX

y_forecastvalues = np.array(fcHAR_SPX_1_sub1)
y_actualvalues = np.array(y_test_sub1_SPX_HAR)
qlikeSPX_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeSPX_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_SPX_sub1 = sum(qlikeSPX_HAR_1_sub1) / len(y_actualvalues)

# HSI

y_forecastvalues = np.array(fcHAR_HSI_1_sub1)
y_actualvalues = np.array(y_test_sub1_HSI_HAR)
qlikeHSI_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeHSI_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_HSI_sub1 = sum(qlikeHSI_HAR_1_sub1) / len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(fcHAR_IBEX_1_sub1)
y_actualvalues = np.array(y_test_sub1_IBEX_HAR)
qlikeIBEX_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeIBEX_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_IBEX_sub1 = sum(qlikeIBEX_HAR_1_sub1) / len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(fcHAR_IXIC_1_sub1)
y_actualvalues = np.array(y_test_sub1_IXIC_HAR)
qlikeIXIC_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeIXIC_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_IXIC_sub1 = sum(qlikeIXIC_HAR_1_sub1) / len(y_actualvalues)

# N225

y_forecastvalues = np.array(fcHAR_N225_1_sub1)
y_actualvalues = np.array(y_test_sub1_N225_HAR)
qlikeN225_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeN225_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_N225_sub1 = sum(qlikeN225_HAR_1_sub1) / len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(fcHAR_OMXC20_1_sub1)
y_actualvalues = np.array(y_test_sub1_OMXC20_HAR)
qlikeOMXC20_HAR_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i] / y_forecastvalues[i] - np.log(y_actualvalues[i] / y_forecastvalues[i]) - 1
    qlikeOMXC20_HAR_1_sub1.append(iteration)
QLIKE_HAR_1_OMXC20_sub1 = sum(qlikeOMXC20_HAR_1_sub1) / len(y_actualvalues)


# HARlog

X_test_DJIlog_c_sub1 = X_test_DJIlog_c.loc[start_date_DJI:end_date_DJI]
X_test_FTSElog_c_sub1 = X_test_FTSElog_c.loc[start_date_FTSE:end_date_FTSE]
X_test_FTSEMIBlog_c_sub1 = X_test_FTSEMIBlog_c.loc[start_date_FTSEMIB:end_date_FTSEMIB]
X_test_GDAXIlog_c_sub1 = X_test_GDAXIlog_c.loc[start_date_GDAXI:end_date_GDAXI]
X_test_SPXlog_c_sub1 = X_test_SPXlog_c.loc[start_date_SPX:end_date_SPX]
X_test_HSIlog_c_sub1 = X_test_HSIlog_c.loc[start_date_HSI:end_date_HSI]
X_test_IBEXlog_c_sub1 = X_test_IBEXlog_c.loc[start_date_IBEX:end_date_IBEX]
X_test_IXIClog_c_sub1 = X_test_IXIClog_c.loc[start_date_IXIC:end_date_IXIC]
X_test_N225log_c_sub1 = X_test_N225log_c.loc[start_date_N225:end_date_N225]
X_test_OMXC20log_c_sub1 = X_test_OMXC20log_c.loc[start_date_OMXC20:end_date_OMXC20]

fcHARlog_DJI_1_sub1 = regHARlog_DJI.predict(X_test_DJIlog_c_sub1)
fcHARlog_FTSE_1_sub1 = regHARlog_FTSE.predict(X_test_FTSElog_c_sub1)
fcHARlog_FTSEMIB_1_sub1 = regHARlog_FTSEMIB.predict(X_test_FTSEMIBlog_c_sub1)
fcHARlog_GDAXI_1_sub1 = regHARlog_GDAXI.predict(X_test_GDAXIlog_c_sub1)
fcHARlog_SPX_1_sub1 = regHARlog_SPX.predict(X_test_SPXlog_c_sub1)
fcHARlog_HSI_1_sub1 = regHARlog_HSI.predict(X_test_HSIlog_c_sub1)
fcHARlog_IBEX_1_sub1 = regHARlog_IBEX.predict(X_test_IBEXlog_c_sub1)
fcHARlog_IXIC_1_sub1 = regHARlog_IXIC.predict(X_test_IXIClog_c_sub1)
fcHARlog_N225_1_sub1 = regHARlog_N225.predict(X_test_N225log_c_sub1)
fcHARlog_OMXC20_1_sub1 = regHARlog_OMXC20.predict(X_test_OMXC20log_c_sub1)

fcHARlog_DJI_1_sub1_adj = np.exp(fcHARlog_DJI_1_sub1)*np.exp(np.var(reslogDJI)/2)
fcHARlog_FTSE_1_sub1_adj = np.exp(fcHARlog_FTSE_1_sub1)*np.exp(np.var(reslogFTSE)/2)
fcHARlog_FTSEMIB_1_sub1_adj = np.exp(fcHARlog_FTSEMIB_1_sub1)*np.exp(np.var(reslogFTSEMIB)/2)
fcHARlog_GDAXI_1_sub1_adj = np.exp(fcHARlog_GDAXI_1_sub1)*np.exp(np.var(reslogGDAXI)/2)
fcHARlog_SPX_1_sub1_adj = np.exp(fcHARlog_SPX_1_sub1)*np.exp(np.var(reslogSPX)/2)
fcHARlog_HSI_1_sub1_adj = np.exp(fcHARlog_HSI_1_sub1)*np.exp(np.var(reslogHSI)/2)
fcHARlog_IBEX_1_sub1_adj = np.exp(fcHARlog_IBEX_1_sub1)*np.exp(np.var(reslogIBEX)/2)
fcHARlog_IXIC_1_sub1_adj = np.exp(fcHARlog_IXIC_1_sub1)*np.exp(np.var(reslogIXIC)/2)
fcHARlog_N225_1_sub1_adj = np.exp(fcHARlog_N225_1_sub1)*np.exp(np.var(reslogN225)/2)
fcHARlog_OMXC20_1_sub1_adj = np.exp(fcHARlog_OMXC20_1_sub1)*np.exp(np.var(reslogOMXC20)/2)

# MSE HAR log

MSE_HARlog_1_DJI_sub1 = mean_squared_error(y_test_sub1_DJI_HAR, fcHARlog_DJI_1_sub1_adj)
MSE_HARlog_1_FTSE_sub1 = mean_squared_error(y_test_sub1_FTSE_HAR, fcHARlog_FTSE_1_sub1_adj)
MSE_HARlog_1_FTSEMIB_sub1 = mean_squared_error(y_test_sub1_FTSEMIB_HAR, fcHARlog_FTSEMIB_1_sub1_adj)
MSE_HARlog_1_GDAXI_sub1 = mean_squared_error(y_test_sub1_GDAXI_HAR, fcHARlog_GDAXI_1_sub1_adj)
MSE_HARlog_1_SPX_sub1 = mean_squared_error(y_test_sub1_SPX_HAR, fcHARlog_SPX_1_sub1_adj)
MSE_HARlog_1_HSI_sub1 = mean_squared_error(y_test_sub1_HSI_HAR, fcHARlog_HSI_1_sub1_adj)
MSE_HARlog_1_IBEX_sub1 = mean_squared_error(y_test_sub1_IBEX_HAR, fcHARlog_IBEX_1_sub1_adj)
MSE_HARlog_1_IXIC_sub1 = mean_squared_error(y_test_sub1_IXIC_HAR, fcHARlog_IXIC_1_sub1_adj)
MSE_HARlog_1_N225_sub1 = mean_squared_error(y_test_sub1_N225_HAR, fcHARlog_N225_1_sub1_adj)
MSE_HARlog_1_OMXC20_sub1 = mean_squared_error(y_test_sub1_OMXC20_HAR, fcHARlog_OMXC20_1_sub1_adj)

mseDJI_HARlog_sub1 = []
for i in np.arange(len(y_test_sub1_DJI_HAR)):
    mse = (fcHARlog_DJI_1_sub1_adj[i]-y_test_sub1_DJI_HAR[i])**2
    mseDJI_HARlog_sub1.append(mse)
mseDJI_HARlog_sub1 = np.array(mseDJI_HARlog_sub1)

mseFTSE_HARlog_sub1 = []
for i in range(len(y_test_sub1_FTSE_HAR)):
    mse = (fcHARlog_FTSE_1_sub1_adj[i] - y_test_sub1_FTSE_HAR[i]) ** 2
    mseFTSE_HARlog_sub1.append(mse)
mseFTSE_HARlog_sub1 = np.array(mseFTSE_HARlog_sub1)

mseFTSEMIB_HARlog_sub1 = []
for i in range(len(y_test_sub1_FTSEMIB_HAR)):
    mse = (fcHARlog_FTSEMIB_1_sub1_adj[i] - y_test_sub1_FTSEMIB_HAR[i]) ** 2
    mseFTSEMIB_HARlog_sub1.append(mse)
mseFTSEMIB_HARlog_sub1 = np.array(mseFTSEMIB_HARlog_sub1)

mseGDAXI_HARlog_sub1 = []
for i in range(len(y_test_sub1_GDAXI_HAR)):
    mse = (fcHARlog_GDAXI_1_sub1_adj[i] - y_test_sub1_GDAXI_HAR[i]) ** 2
    mseGDAXI_HARlog_sub1.append(mse)
mseGDAXI_HARlog_sub1 = np.array(mseGDAXI_HARlog_sub1)

mseSPX_HARlog_sub1 = []
for i in range(len(y_test_sub1_SPX_HAR)):
    mse = (fcHARlog_SPX_1_sub1_adj[i] - y_test_sub1_SPX_HAR[i]) ** 2
    mseSPX_HARlog_sub1.append(mse)
mseSPX_HARlog_sub1 = np.array(mseSPX_HARlog_sub1)

mseHSI_HARlog_sub1 = []
for i in range(len(y_test_sub1_HSI_HAR)):
    mse = (fcHARlog_HSI_1_sub1_adj[i] - y_test_sub1_HSI_HAR[i]) ** 2
    mseHSI_HARlog_sub1.append(mse)
mseHSI_HARlog_sub1 = np.array(mseHSI_HARlog_sub1)

mseIBEX_HARlog_sub1 = []
for i in range(len(y_test_sub1_IBEX_HAR)):
    mse = (fcHARlog_IBEX_1_sub1_adj[i] - y_test_sub1_IBEX_HAR[i]) ** 2
    mseIBEX_HARlog_sub1.append(mse)
mseIBEX_HARlog_sub1 = np.array(mseIBEX_HARlog_sub1)

mseIXIC_HARlog_sub1 = []
for i in range(len(y_test_sub1_IXIC_HAR)):
    mse = (fcHARlog_IXIC_1_sub1_adj[i] - y_test_sub1_IXIC_HAR[i]) ** 2
    mseIXIC_HARlog_sub1.append(mse)
mseIXIC_HARlog_sub1 = np.array(mseIXIC_HARlog_sub1)

mseN225_HARlog_sub1 = []
for i in range(len(y_test_sub1_N225_HAR)):
    mse = (fcHARlog_N225_1_sub1_adj[i] - y_test_sub1_N225_HAR[i]) ** 2
    mseN225_HARlog_sub1.append(mse)
mseN225_HARlog_sub1 = np.array(mseN225_HARlog_sub1)

mseOMXC20_HARlog_sub1 = []
for i in range(len(y_test_sub1_OMXC20_HAR)):
    mse = (fcHARlog_OMXC20_1_sub1_adj[i] - y_test_sub1_OMXC20_HAR[i]) ** 2
    mseOMXC20_HARlog_sub1.append(mse)
mseOMXC20_HARlog_sub1 = np.array(mseOMXC20_HARlog_sub1)

# QLIKE HARlog

# DJI 

y_forecastvalues = np.array(fcHARlog_DJI_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_DJI_HAR)
qlikeDJI_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_DJI_sub1 = sum(qlikeDJI_HARlog_1_sub1)/len(y_actualvalues)

# FTSE

y_forecastvalues = np.array(fcHARlog_FTSE_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_FTSE_HAR)
qlikeFTSE_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_FTSE_sub1 = sum(qlikeFTSE_HARlog_1_sub1)/len(y_actualvalues)

# FTSEMIB

y_forecastvalues = np.array(fcHARlog_FTSEMIB_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_FTSEMIB_HAR)
qlikeFTSEMIB_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_FTSEMIB_sub1 = sum(qlikeFTSEMIB_HARlog_1_sub1)/len(y_actualvalues)

# GDAXI

y_forecastvalues = np.array(fcHARlog_GDAXI_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_GDAXI_HAR)
qlikeGDAXI_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_GDAXI_sub1 = sum(qlikeGDAXI_HARlog_1_sub1)/len(y_actualvalues)

# SPX

y_forecastvalues = np.array(fcHARlog_SPX_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_SPX_HAR)
qlikeSPX_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_SPX_sub1 = sum(qlikeSPX_HARlog_1_sub1)/len(y_actualvalues)

# HSI

y_forecastvalues = np.array(fcHARlog_HSI_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_HSI_HAR)
qlikeHSI_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_HSI_sub1 = sum(qlikeHSI_HARlog_1_sub1)/len(y_actualvalues)

# IBEX

y_forecastvalues = np.array(fcHARlog_IBEX_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_IBEX_HAR)
qlikeIBEX_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_IBEX_sub1 = sum(qlikeIBEX_HARlog_1_sub1)/len(y_actualvalues)

# IXIC

y_forecastvalues = np.array(fcHARlog_IXIC_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_IXIC_HAR)
qlikeIXIC_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_IXIC_sub1 = sum(qlikeIXIC_HARlog_1_sub1)/len(y_actualvalues)

# N225

y_forecastvalues = np.array(fcHARlog_N225_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_N225_HAR)
qlikeN225_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_N225_sub1 = sum(qlikeN225_HARlog_1_sub1)/len(y_actualvalues)

# OMXC20

y_forecastvalues = np.array(fcHARlog_OMXC20_1_sub1_adj)
y_actualvalues = np.array(y_test_sub1_OMXC20_HAR)
qlikeOMXC20_HARlog_1_sub1 = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HARlog_1_sub1.append(iteration)
QLIKE_HARlog_1_OMXC20_sub1 = sum(qlikeOMXC20_HARlog_1_sub1)/len(y_actualvalues)

#%%

arrays_Out = {
    'mse_AR1': {
        'DJI': mseDJI_AR1_sub1,
        'FTSE': mseFTSE_AR1_sub1,
        'FTSEMIB': mseFTSEMIB_AR1_sub1,
        'GDAXI': mseGDAXI_AR1_sub1,
        'SPX': mseSPX_AR1_sub1,
        'HSI': mseHSI_AR1_sub1,
        'IBEX': mseIBEX_AR1_sub1,
        'IXIC': mseIXIC_AR1_sub1,
        'N225': mseN225_AR1_sub1,
        'OMXC20': mseOMXC20_AR1_sub1
    },
    'mse_HAR': {
        'DJI': mseDJI_HAR_sub1,
        'FTSE': mseFTSE_HAR_sub1,
        'FTSEMIB': mseFTSEMIB_HAR_sub1,
        'GDAXI': mseGDAXI_HAR_sub1,
        'SPX': mseSPX_HAR_sub1,
        'HSI': mseHSI_HAR_sub1,
        'IBEX': mseIBEX_HAR_sub1,
        'IXIC': mseIXIC_HAR_sub1,
        'N225': mseN225_HAR_sub1,
        'OMXC20': mseOMXC20_HAR_sub1
    },
    'mse_HARlog': {
        'DJI': mseDJI_HARlog_sub1,
        'FTSE': mseFTSE_HARlog_sub1,
        'FTSEMIB': mseFTSEMIB_HARlog_sub1,
        'GDAXI': mseGDAXI_HARlog_sub1,
        'SPX': mseSPX_HARlog_sub1,
        'HSI': mseHSI_HARlog_sub1,
        'IBEX': mseIBEX_HARlog_sub1,
        'IXIC': mseIXIC_HARlog_sub1,
        'N225': mseN225_HARlog_sub1,
        'OMXC20': mseOMXC20_HARlog_sub1
                    }
    }


for k1 in arrays_Out:
    if k1 == 'mse_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_AR1_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HAR_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HARlog_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%% 

arrays_Out = {
    'qlike_AR1': {
        'DJI': qlikeDJI_AR1_1_sub1,
        'FTSE': qlikeFTSE_AR1_1_sub1,
        'FTSEMIB': qlikeFTSEMIB_AR1_1_sub1,
        'GDAXI': qlikeGDAXI_AR1_1_sub1,
        'SPX': qlikeSPX_AR1_1_sub1,
        'HSI': qlikeHSI_AR1_1_sub1,
        'IBEX': qlikeIBEX_AR1_1_sub1,
        'IXIC': qlikeIXIC_AR1_1_sub1,
        'N225': qlikeN225_AR1_1_sub1,
        'OMXC20': qlikeOMXC20_AR1_1_sub1
    },
    'qlike_HAR': {
        'DJI': qlikeDJI_HAR_1_sub1,
        'FTSE': qlikeFTSE_HAR_1_sub1,
        'FTSEMIB': qlikeFTSEMIB_HAR_1_sub1,
        'GDAXI': qlikeGDAXI_HAR_1_sub1,
        'SPX': qlikeSPX_HAR_1_sub1,
        'HSI': qlikeHSI_HAR_1_sub1,
        'IBEX': qlikeIBEX_HAR_1_sub1,
        'IXIC': qlikeIXIC_HAR_1_sub1,
        'N225': qlikeN225_HAR_1_sub1,
        'OMXC20': qlikeOMXC20_HAR_1_sub1
    },
    'qlike_HARlog': {
        'DJI': qlikeDJI_HARlog_1_sub1, 
        'FTSE': qlikeFTSE_HARlog_1_sub1,
        'FTSEMIB': qlikeFTSEMIB_HARlog_1_sub1,
        'GDAXI': qlikeGDAXI_HARlog_1_sub1,
        'SPX': qlikeSPX_HARlog_1_sub1,
        'HSI': qlikeHSI_HARlog_1_sub1,
        'IBEX': qlikeIBEX_HARlog_1_sub1,
        'IXIC': qlikeIXIC_HARlog_1_sub1,
        'N225': qlikeN225_HARlog_1_sub1,
        'OMXC20': qlikeOMXC20_HARlog_1_sub1
                    }
    }


for k1 in arrays_Out:
    if k1 == 'qlike_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_AR1_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HAR_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HARlog_sub1.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')            

