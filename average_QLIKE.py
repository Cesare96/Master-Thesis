# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:10:07 2023

@author: cesar
"""

import numpy as np

#%% Out

averageMSE_AR1 = np.mean([MSE_AR1_1_DJI, MSE_AR1_1_FTSE, MSE_AR1_1_FTSEMIB, MSE_AR1_1_GDAXI, MSE_AR1_1_SPX, MSE_AR1_1_HSI, MSE_AR1_1_IBEX, MSE_AR1_1_IXIC, MSE_AR1_1_N225, MSE_AR1_1_OMXC20])
print(round(averageMSE_AR1, 11))

averageMSE_HAR = np.mean([MSE_HAR_1_DJI, MSE_HAR_1_FTSE, MSE_HAR_1_FTSEMIB, MSE_HAR_1_GDAXI, MSE_HAR_1_SPX, MSE_HAR_1_HSI, MSE_HAR_1_IBEX, MSE_HAR_1_IXIC, MSE_HAR_1_N225, MSE_HAR_1_OMXC20])
print(round(averageMSE_HAR, 11))

averageMSE_HARlog = np.mean([MSE_HARlog_1_DJI, MSE_HARlog_1_FTSE, MSE_HARlog_1_FTSEMIB, MSE_HARlog_1_GDAXI, MSE_HARlog_1_SPX, MSE_HARlog_1_HSI, MSE_HARlog_1_IBEX, MSE_HARlog_1_IXIC, MSE_HARlog_1_N225, MSE_HARlog_1_OMXC20])
print(round(averageMSE_HARlog, 11))

averageMSE_rf = np.mean([MSE_DJI_1_rf, MSE_FTSE_1_rf, MSE_FTSEMIB_1_rf, MSE_GDAXI_1_rf, MSE_SPX_1_rf, MSE_HSI_1_rf, MSE_IBEX_1_rf, MSE_IXIC_1_rf, MSE_N225_1_rf, MSE_OMXC20_1_rf])
print(round(averageMSE_rf, 11))

averageMSE_LSTM = np.mean([MSE_LSTM_1_DJI, MSE_LSTM_1_FTSE, MSE_LSTM_1_FTSEMIB, MSE_LSTM_1_GDAXI, MSE_LSTM_1_SPX, MSE_LSTM_1_HSI, MSE_LSTM_1_IBEX, MSE_LSTM_1_IXIC, MSE_LSTM_1_N225, MSE_LSTM_1_OMXC20])
print(round(averageMSE_LSTM, 11))

averageMSE_FNN = np.mean([MSE_FNN_1_DJI, MSE_FNN_1_FTSE, MSE_FNN_1_FTSEMIB, MSE_FNN_1_GDAXI, MSE_FNN_1_SPX, MSE_FNN_1_HSI, MSE_FNN_1_IBEX, MSE_FNN_1_IXIC, MSE_FNN_1_N225, MSE_FNN_1_OMXC20])
print(round(averageMSE_FNN, 11))

numero1 = round(averageMSE_AR1/averageMSE_HAR,3)
print(numero1)

numero2 = round(averageMSE_HARlog/averageMSE_HAR,3)
print(numero2)

numero3 = round(averageMSE_rf/averageMSE_HAR,3)
print(numero3)

numero4 = round(averageMSE_LSTM/averageMSE_HAR,3)
print(numero4)

numero5 = round(averageMSE_FNN/averageMSE_HAR,3)
print(numero5)
