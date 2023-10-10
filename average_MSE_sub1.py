# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:46:01 2023

@author: cesar
"""

#%%

import numpy as np

#%% 

averageMSE_AR1_sub1 = np.mean([MSE_AR1_1_DJI_sub1, MSE_AR1_1_FTSE_sub1, MSE_AR1_1_FTSEMIB_sub1, MSE_AR1_1_GDAXI_sub1, MSE_AR1_1_SPX_sub1, MSE_AR1_1_HSI_sub1, MSE_AR1_1_IBEX_sub1, MSE_AR1_1_IXIC_sub1, MSE_AR1_1_N225_sub1, MSE_AR1_1_OMXC20_sub1])
print(f'AR(1): {round(averageMSE_AR1_sub1, 11)}')

averageMSE_HAR_sub1 = np.mean([MSE_HAR_1_DJI_sub1, MSE_HAR_1_FTSE_sub1, MSE_HAR_1_FTSEMIB_sub1, MSE_HAR_1_GDAXI_sub1, MSE_HAR_1_SPX_sub1, MSE_HAR_1_HSI_sub1, MSE_HAR_1_IBEX_sub1, MSE_HAR_1_IXIC_sub1, MSE_HAR_1_N225_sub1, MSE_HAR_1_OMXC20_sub1])
print(f'HAR: {round(averageMSE_HAR_sub1, 11)}')

averageMSE_HARlog_sub1 = np.mean([MSE_HARlog_1_DJI_sub1, MSE_HARlog_1_FTSE_sub1, MSE_HARlog_1_FTSEMIB_sub1, MSE_HARlog_1_GDAXI_sub1, MSE_HARlog_1_SPX_sub1, MSE_HARlog_1_HSI_sub1, MSE_HARlog_1_IBEX_sub1, MSE_HARlog_1_IXIC_sub1, MSE_HARlog_1_N225_sub1, MSE_HARlog_1_OMXC20_sub1])
print(f'HARlog: {round(averageMSE_HARlog_sub1, 11)}')

averageMSE_rf_sub1 = np.mean([MSE_DJI_1_rf_sub1, MSE_FTSE_1_rf_sub1, MSE_FTSEMIB_1_rf_sub1, MSE_GDAXI_1_rf_sub1, MSE_SPX_1_rf_sub1, MSE_HSI_1_rf_sub1, MSE_IBEX_1_rf_sub1, MSE_IXIC_1_rf_sub1, MSE_N225_1_rf_sub1, MSE_OMXC20_1_rf_sub1])
print(f'RF: {round(averageMSE_rf_sub1, 11)}')

averageMSE_LSTM_sub1 = np.mean([MSE_LSTM_1_DJI_sub1, MSE_LSTM_1_FTSE_sub1, MSE_LSTM_1_FTSEMIB_sub1, MSE_LSTM_1_GDAXI_sub1, MSE_LSTM_1_SPX_sub1, MSE_LSTM_1_HSI_sub1, MSE_LSTM_1_IBEX_sub1, MSE_LSTM_1_IXIC_sub1, MSE_LSTM_1_N225_sub1, MSE_LSTM_1_OMXC20_sub1])
print(f'LSTM: {round(averageMSE_LSTM_sub1, 11)}')

averageMSE_FNN_sub1 = np.mean([MSE_FNN_1_DJI_sub1, MSE_FNN_1_FTSE_sub1, MSE_FNN_1_FTSEMIB_sub1, MSE_FNN_1_GDAXI_sub1, MSE_FNN_1_SPX_sub1, MSE_FNN_1_HSI_sub1, MSE_FNN_1_IBEX_sub1, MSE_FNN_1_IXIC_sub1, MSE_FNN_1_N225_sub1, MSE_FNN_1_OMXC20_sub1])
print(f'FFN: {round(averageMSE_FNN_sub1, 11)}')

numero1_sub1 = round(averageMSE_AR1_sub1/averageMSE_HAR_sub1, 3)
print(f'AR1: {numero1_sub1}')

numeroBench_sub1 = round(averageMSE_HAR_sub1/averageMSE_HAR_sub1,3)
print(f'HAR: {numeroBench_sub1}')

numero2_sub1 = round(averageMSE_HARlog_sub1/averageMSE_HAR_sub1, 3)
print(f'HARlog: {numero2_sub1}')

numero3_sub1 = round(averageMSE_rf_sub1/averageMSE_HAR_sub1, 3)
print(f'RF: {numero3_sub1}')

numero4_sub1 = round(averageMSE_LSTM_sub1/averageMSE_HAR_sub1, 3)
print(f'LSTM: {numero4_sub1}')

numero5_sub1 = round(averageMSE_FNN_sub1/averageMSE_HAR_sub1, 3)
print(f'FFN: {numero5_sub1}')


