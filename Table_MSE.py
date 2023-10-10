# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:19:19 2023

@author: cesar
"""

from tabulate import tabulate

#%% Table MSE

print('Table of MSE between, AR(1), HAR and HARlog models, Random Forests, LSTM and FNN\n')

table = [['Index', 'MSE AR(1)', 'MSE HAR', 'MSE HARlog', 'MSE_RF', 'MSE LSTM', 'MSE FNN'], 
        ['DJI', round(MSE_AR1_1_DJI, 11), round(MSE_HAR_1_DJI, 11), round(MSE_HARlog_1_DJI, 11), round(MSE_DJI_1_rf, 11), round(MSE_LSTM_1_DJI, 11), round(MSE_FNN_1_DJI, 11)], 
        ['FTSE', round(MSE_AR1_1_FTSE, 11), round(MSE_HAR_1_FTSE, 11), round(MSE_HARlog_1_FTSE, 11), round(MSE_FTSE_1_rf, 11), round(MSE_LSTM_1_FTSE, 11), round(MSE_FNN_1_FTSE, 11)], 
        ['FTSEMIB', round(MSE_AR1_1_FTSEMIB, 11), round(MSE_HAR_1_FTSEMIB, 11), round(MSE_HARlog_1_FTSEMIB, 11), round(MSE_FTSEMIB_1_rf, 11), round(MSE_LSTM_1_FTSEMIB, 11), round(MSE_FNN_1_FTSEMIB, 11)],
        ['GDAXI', round(MSE_AR1_1_GDAXI, 11), round(MSE_HAR_1_GDAXI, 11), round(MSE_HARlog_1_GDAXI, 11), round(MSE_GDAXI_1_rf, 11), round(MSE_LSTM_1_GDAXI, 11), round(MSE_FNN_1_GDAXI, 11)],
        ['SPX', round(MSE_AR1_1_SPX, 11), round(MSE_HAR_1_SPX, 11), round(MSE_HARlog_1_SPX, 11),  round(MSE_SPX_1_rf, 11), round(MSE_LSTM_1_SPX, 11), round(MSE_FNN_1_SPX, 11)],
        ['HSI', round(MSE_AR1_1_HSI, 11), round(MSE_HAR_1_SPX, 11), round(MSE_HARlog_1_HSI, 11),  round(MSE_HSI_1_rf, 11), round(MSE_LSTM_1_HSI, 11), round(MSE_FNN_1_HSI, 11)],
        ['IBEX', round(MSE_AR1_1_IBEX, 11), round(MSE_HAR_1_IBEX, 11), round(MSE_HARlog_1_IBEX, 11),  round(MSE_IBEX_1_rf, 11), round(MSE_LSTM_1_IBEX, 11), round(MSE_FNN_1_IBEX, 11)],
        ['IXIC', round(MSE_AR1_1_IXIC, 11), round(MSE_HAR_1_IXIC, 11), round(MSE_HARlog_1_IXIC, 11),  round(MSE_IXIC_1_rf, 11), round(MSE_LSTM_1_IXIC, 11), round(MSE_FNN_1_IXIC,11)],
        ['N225', round(MSE_AR1_1_N225, 11), round(MSE_HAR_1_N225, 11), round(MSE_HARlog_1_N225, 11),  round(MSE_N225_1_rf, 11), round(MSE_LSTM_1_N225, 11), round(MSE_FNN_1_N225, 11)],
        ['OMXC20', round(MSE_AR1_1_OMXC20, 11), round(MSE_HAR_1_OMXC20, 11), round(MSE_HARlog_1_OMXC20, 11),  round(MSE_OMXC20_1_rf, 11), round(MSE_LSTM_1_OMXC20, 11), round(MSE_FNN_1_OMXC20, 11)]]

print(tabulate(table, headers='firstrow', tablefmt = 'plain'))

