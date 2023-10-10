# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:46:53 2023

@author: cesar
"""

#%%

import numpy as np

#%% Out

averageQLIKE_AR1_sub1 = np.mean([QLIKE_AR1_1_DJI_sub1, QLIKE_AR1_1_FTSE_sub1, QLIKE_AR1_1_FTSEMIB_sub1, QLIKE_AR1_1_GDAXI_sub1, QLIKE_AR1_1_SPX_sub1, QLIKE_AR1_1_HSI_sub1, QLIKE_AR1_1_IBEX_sub1, QLIKE_AR1_1_IXIC_sub1, QLIKE_AR1_1_N225_sub1, QLIKE_AR1_1_OMXC20_sub1])
print(f'AR(1): {round(averageQLIKE_AR1_sub1, 11)}')

averageQLIKE_HAR_sub1 = np.mean([QLIKE_HAR_1_DJI_sub1, QLIKE_HAR_1_FTSE_sub1, QLIKE_HAR_1_FTSEMIB_sub1, QLIKE_HAR_1_GDAXI_sub1, QLIKE_HAR_1_SPX_sub1, QLIKE_HAR_1_HSI_sub1, QLIKE_HAR_1_IBEX_sub1, QLIKE_HAR_1_IXIC_sub1, QLIKE_HAR_1_N225_sub1, QLIKE_HAR_1_OMXC20_sub1])
print(f'HAR: {round(averageQLIKE_HAR_sub1, 11)}')

averageQLIKE_HARlog_sub1 = np.mean([QLIKE_HARlog_1_DJI_sub1, QLIKE_HARlog_1_FTSE_sub1, QLIKE_HARlog_1_FTSEMIB_sub1, QLIKE_HARlog_1_GDAXI_sub1, QLIKE_HARlog_1_SPX_sub1, QLIKE_HARlog_1_HSI_sub1, QLIKE_HARlog_1_IBEX_sub1, QLIKE_HARlog_1_IXIC_sub1, QLIKE_HARlog_1_N225_sub1, QLIKE_HARlog_1_OMXC20_sub1])
print(f'HARlog: {round(averageQLIKE_HARlog_sub1, 11)}')

averageQLIKE_rf_sub1 = np.mean([QLIKE_rf_1_DJI_sub1, QLIKE_rf_1_FTSE_sub1, QLIKE_rf_1_FTSEMIB_sub1, QLIKE_rf_1_GDAXI_sub1, QLIKE_rf_1_SPX_sub1, QLIKE_rf_1_HSI_sub1, QLIKE_rf_1_IBEX_sub1, QLIKE_rf_1_IXIC_sub1, QLIKE_rf_1_N225_sub1, QLIKE_rf_1_OMXC20_sub1])
print(f'RF: {round(averageQLIKE_rf_sub1, 11)}')

averageQLIKE_LSTM_sub1 = np.mean([QLIKE_LSTM_1_DJI_sub1, QLIKE_LSTM_1_FTSE_sub1, QLIKE_LSTM_1_FTSEMIB_sub1, QLIKE_LSTM_1_GDAXI_sub1, QLIKE_LSTM_1_SPX_sub1, QLIKE_LSTM_1_HSI_sub1, QLIKE_LSTM_1_IBEX_sub1, QLIKE_LSTM_1_IXIC_sub1, QLIKE_LSTM_1_N225_sub1, QLIKE_LSTM_1_OMXC20_sub1])
print(f'LSTM: {round(averageQLIKE_LSTM_sub1, 11)}')

averageQLIKE_FNN_sub1 = np.mean([QLIKE_FNN_1_DJI_sub1, QLIKE_FNN_1_FTSE_sub1, QLIKE_FNN_1_FTSEMIB_sub1, QLIKE_FNN_1_GDAXI_sub1, QLIKE_FNN_1_SPX_sub1, QLIKE_FNN_1_HSI_sub1, QLIKE_FNN_1_IBEX_sub1, QLIKE_FNN_1_IXIC_sub1, QLIKE_FNN_1_N225_sub1, QLIKE_FNN_1_OMXC20_sub1])
print(f'FFN: {round(averageQLIKE_FNN_sub1, 11)}')

numero1_sub1 = round(averageQLIKE_AR1_sub1/averageQLIKE_HAR_sub1,3)
print(f'AR1: {numero1_sub1}')

numeroBench_sub1 = round(averageQLIKE_HAR_sub1/averageQLIKE_HAR_sub1,3)
print(f'HAR: {numeroBench_sub1}')

numero2_sub1 = round(averageQLIKE_HARlog_sub1/averageQLIKE_HAR_sub1,3)
print(f'HARlog: {numero2_sub1}')

numero3_sub1 = round(averageQLIKE_rf_sub1/averageQLIKE_HAR_sub1,3)
print(f'RF: {numero3_sub1}')

numero4_sub1 = round(averageQLIKE_LSTM_sub1/averageQLIKE_HAR_sub1,3)
print(f'LSTM: {numero4_sub1}')

numero5_sub1 = round(averageQLIKE_FNN_sub1/averageQLIKE_HAR_sub1,3)
print(f'FFN: {numero5_sub1}')
