# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:37:51 2023

@author: cesar
"""

#%%

import numpy as np

#%% Out

averageMSE_AR1 = np.mean([MSE_AR1_1_DJI, MSE_AR1_1_FTSE, MSE_AR1_1_FTSEMIB, MSE_AR1_1_GDAXI, MSE_AR1_1_SPX, MSE_AR1_1_HSI, MSE_AR1_1_IBEX, MSE_AR1_1_IXIC, MSE_AR1_1_N225, MSE_AR1_1_OMXC20])
print(f'AR(1): {round(averageMSE_AR1, 11)}')

averageMSE_HAR = np.mean([MSE_HAR_1_DJI, MSE_HAR_1_FTSE, MSE_HAR_1_FTSEMIB, MSE_HAR_1_GDAXI, MSE_HAR_1_SPX, MSE_HAR_1_HSI, MSE_HAR_1_IBEX, MSE_HAR_1_IXIC, MSE_HAR_1_N225, MSE_HAR_1_OMXC20])
print(f'HAR: {round(averageMSE_HAR, 11)}')

averageMSE_HARlog = np.mean([MSE_HARlog_1_DJI, MSE_HARlog_1_FTSE, MSE_HARlog_1_FTSEMIB, MSE_HARlog_1_GDAXI, MSE_HARlog_1_SPX, MSE_HARlog_1_HSI, MSE_HARlog_1_IBEX, MSE_HARlog_1_IXIC, MSE_HARlog_1_N225, MSE_HARlog_1_OMXC20])
print(f'HARlog: {round(averageMSE_HARlog, 11)}')

averageMSE_rf = np.mean([MSE_DJI_1_rf, MSE_FTSE_1_rf, MSE_FTSEMIB_1_rf, MSE_GDAXI_1_rf, MSE_SPX_1_rf, MSE_HSI_1_rf, MSE_IBEX_1_rf, MSE_IXIC_1_rf, MSE_N225_1_rf, MSE_OMXC20_1_rf])
print(f'RF: {round(averageMSE_rf, 11)}')

averageMSE_LSTM = np.mean([MSE_LSTM_1_DJI, MSE_LSTM_1_FTSE, MSE_LSTM_1_FTSEMIB, MSE_LSTM_1_GDAXI, MSE_LSTM_1_SPX, MSE_LSTM_1_HSI, MSE_LSTM_1_IBEX, MSE_LSTM_1_IXIC, MSE_LSTM_1_N225, MSE_LSTM_1_OMXC20])
print(f'LSTM: {round(averageMSE_LSTM, 11)}')

averageMSE_FNN = np.mean([MSE_FNN_1_DJI, MSE_FNN_1_FTSE, MSE_FNN_1_FTSEMIB, MSE_FNN_1_GDAXI, MSE_FNN_1_SPX, MSE_FNN_1_HSI, MSE_FNN_1_IBEX, MSE_FNN_1_IXIC, MSE_FNN_1_N225, MSE_FNN_1_OMXC20])
print(f'FFN: {round(averageMSE_FNN, 11)}')

numero1 = round(averageMSE_AR1/averageMSE_HAR, 3)
print(f'AR1: {numero1}')

numeroBench = round(averageMSE_HAR/averageMSE_HAR,3)
print(f'HAR: {numeroBench}')

numero2 = round(averageMSE_HARlog/averageMSE_HAR, 3)
print(f'HARlog: {numero2}')

numero3 = round(averageMSE_rf/averageMSE_HAR, 3)
print(f'RF: {numero3}')

numero4 = round(averageMSE_LSTM/averageMSE_HAR, 3)
print(f'LSTM: {numero4}')

numero5 = round(averageMSE_FNN/averageMSE_HAR, 3)
print(f'FFN: {numero5}')


#%% In

averageMSE_AR1_In = np.mean([MSE_AR1_1_DJI_In, MSE_AR1_1_FTSE_In, MSE_AR1_1_FTSEMIB_In, MSE_AR1_1_GDAXI_In, MSE_AR1_1_SPX_In, MSE_AR1_1_HSI_In, MSE_AR1_1_IBEX_In, MSE_AR1_1_IXIC_In, MSE_AR1_1_N225_In, MSE_AR1_1_OMXC20_In])
print(f'AR(1): {round(averageMSE_AR1_In, 11)}')

averageMSE_HAR_In = np.mean([MSE_HAR_1_DJI_In, MSE_HAR_1_FTSE_In, MSE_HAR_1_FTSEMIB_In, MSE_HAR_1_GDAXI_In, MSE_HAR_1_SPX_In, MSE_HAR_1_HSI_In, MSE_HAR_1_IBEX_In, MSE_HAR_1_IXIC_In, MSE_HAR_1_N225_In, MSE_HAR_1_OMXC20_In])
print(f'HAR: {round(averageMSE_HAR_In, 11)}')

averageMSE_HARlog_In = np.mean([MSE_HARlog_1_DJI_In, MSE_HARlog_1_FTSE_In, MSE_HARlog_1_FTSEMIB_In, MSE_HARlog_1_GDAXI_In, MSE_HARlog_1_SPX_In, MSE_HARlog_1_HSI_In, MSE_HARlog_1_IBEX_In, MSE_HARlog_1_IXIC_In, MSE_HARlog_1_N225_In, MSE_HARlog_1_OMXC20_In])
print(f'HARlog: {round(averageMSE_HARlog_In, 11)}')

averageMSE_rf_In = np.mean([MSE_DJI_1_rf, MSE_FTSE_1_rf_In, MSE_FTSEMIB_1_rf_In, MSE_GDAXI_1_rf_In, MSE_SPX_1_rf_In, MSE_HSI_1_rf_In, MSE_IBEX_1_rf_In, MSE_IXIC_1_rf_In, MSE_N225_1_rf_In, MSE_OMXC20_1_rf_In])
print(f'RF: {round(averageMSE_rf_In, 11)}')

averageMSE_LSTM_In = np.mean([MSE_LSTM_1_DJI_In, MSE_LSTM_1_FTSE_In, MSE_LSTM_1_FTSEMIB_In, MSE_LSTM_1_GDAXI_In, MSE_LSTM_1_SPX_In, MSE_LSTM_1_HSI_In, MSE_LSTM_1_IBEX_In, MSE_LSTM_1_IXIC_In, MSE_LSTM_1_N225_In, MSE_LSTM_1_OMXC20_In])
print(f'LSTM: {round(averageMSE_LSTM_In, 11)}')

averageMSE_FNN_In = np.mean([MSE_FNN_1_DJI_In, MSE_FNN_1_FTSE_In, MSE_FNN_1_FTSEMIB_In, MSE_FNN_1_GDAXI_In, MSE_FNN_1_SPX_In, MSE_FNN_1_HSI_In, MSE_FNN_1_IBEX_In, MSE_FNN_1_IXIC_In, MSE_FNN_1_N225_In, MSE_FNN_1_OMXC20_In])
print(f'FFN: {round(averageMSE_FNN_In, 11)}')

numero1 = round(averageMSE_AR1_In/averageMSE_HAR_In, 3)
print(f'AR1: {numero1}')

numeroBench = round(averageMSE_HAR_In/averageMSE_HAR_In,3)
print(f'HAR: {numeroBench}')

numero2 = round(averageMSE_HARlog_In/averageMSE_HAR_In, 3)
print(f'HARlog: {numero2}')

numero3 = round(averageMSE_rf_In/averageMSE_HAR_In, 3)
print(f'RF: {numero3}')

numero4 = round(averageMSE_LSTM_In/averageMSE_HAR_In, 3)
print(f'LSTM: {numero4}')

numero5 = round(averageMSE_FNN_In/averageMSE_HAR_In, 3)
print(f'FFN: {numero5}')



