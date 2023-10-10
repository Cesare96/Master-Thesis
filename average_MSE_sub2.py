# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:47:18 2023

@author: cesar
"""

#%%

import numpy as np

#%%

averageMSE_AR1_sub2 = np.mean([MSE_AR1_1_DJI_sub2, MSE_AR1_1_FTSE_sub2, MSE_AR1_1_FTSEMIB_sub2, MSE_AR1_1_GDAXI_sub2, MSE_AR1_1_SPX_sub2, MSE_AR1_1_HSI_sub2, MSE_AR1_1_IBEX_sub2, MSE_AR1_1_IXIC_sub2, MSE_AR1_1_N225_sub2, MSE_AR1_1_OMXC20_sub2])
print(f'AR(1): {round(averageMSE_AR1_sub2, 11)}')

averageMSE_HAR_sub2 = np.mean([MSE_HAR_1_DJI_sub2, MSE_HAR_1_FTSE_sub2, MSE_HAR_1_FTSEMIB_sub2, MSE_HAR_1_GDAXI_sub2, MSE_HAR_1_SPX_sub2, MSE_HAR_1_HSI_sub2, MSE_HAR_1_IBEX_sub2, MSE_HAR_1_IXIC_sub2, MSE_HAR_1_N225_sub2, MSE_HAR_1_OMXC20_sub2])
print(f'HAR: {round(averageMSE_HAR_sub2, 11)}')

averageMSE_HARlog_sub2 = np.mean([MSE_HARlog_1_DJI_sub2, MSE_HARlog_1_FTSE_sub2, MSE_HARlog_1_FTSEMIB_sub2, MSE_HARlog_1_GDAXI_sub2, MSE_HARlog_1_SPX_sub2, MSE_HARlog_1_HSI_sub2, MSE_HARlog_1_IBEX_sub2, MSE_HARlog_1_IXIC_sub2, MSE_HARlog_1_N225_sub2, MSE_HARlog_1_OMXC20_sub2])
print(f'HARlog: {round(averageMSE_HARlog_sub2, 11)}')

averageMSE_rf_sub2 = np.mean([MSE_DJI_1_rf_sub2, MSE_FTSE_1_rf_sub2, MSE_FTSEMIB_1_rf_sub2, MSE_GDAXI_1_rf_sub2, MSE_SPX_1_rf_sub2, MSE_HSI_1_rf_sub2, MSE_IBEX_1_rf_sub2, MSE_IXIC_1_rf_sub2, MSE_N225_1_rf_sub2, MSE_OMXC20_1_rf_sub2])
print(f'RF: {round(averageMSE_rf_sub2, 11)}')

averageMSE_LSTM_sub2 = np.mean([MSE_LSTM_1_DJI_sub2, MSE_LSTM_1_FTSE_sub2, MSE_LSTM_1_FTSEMIB_sub2, MSE_LSTM_1_GDAXI_sub2, MSE_LSTM_1_SPX_sub2, MSE_LSTM_1_HSI_sub2, MSE_LSTM_1_IBEX_sub2, MSE_LSTM_1_IXIC_sub2, MSE_LSTM_1_N225_sub2, MSE_LSTM_1_OMXC20_sub2])
print(f'LSTM: {round(averageMSE_LSTM_sub2, 11)}')

averageMSE_FNN_sub2 = np.mean([MSE_FNN_1_DJI_sub2, MSE_FNN_1_FTSE_sub2, MSE_FNN_1_FTSEMIB_sub2, MSE_FNN_1_GDAXI_sub2, MSE_FNN_1_SPX_sub2, MSE_FNN_1_HSI_sub2, MSE_FNN_1_IBEX_sub2, MSE_FNN_1_IXIC_sub2, MSE_FNN_1_N225_sub2, MSE_FNN_1_OMXC20_sub2])
print(f'FFN: {round(averageMSE_FNN_sub2, 11)}')

numero1_sub2 = round(averageMSE_AR1_sub2/averageMSE_HAR_sub2, 3)
print(f'AR1: {numero1_sub2}')

numeroBench_sub2 = round(averageMSE_HAR_sub2/averageMSE_HAR_sub2,3)
print(f'HAR: {numeroBench_sub2}')

numero2_sub2 = round(averageMSE_HARlog_sub2/averageMSE_HAR_sub2, 3)
print(f'HARlog: {numero2_sub2}')

numero3_sub2 = round(averageMSE_rf_sub2/averageMSE_HAR_sub2, 3)
print(f'RF: {numero3_sub2}')

numero4_sub2 = round(averageMSE_LSTM_sub2/averageMSE_HAR_sub2, 3)
print(f'LSTM: {numero4_sub2}')

numero5_sub2 = round(averageMSE_FNN_sub2/averageMSE_HAR_sub2, 3)
print(f'FFN: {numero5_sub2}')