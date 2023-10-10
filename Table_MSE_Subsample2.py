# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 01:09:47 2023

@author: cesar
"""

from tabulate import tabulate

#%% Table MSE

print('Table of MSE between, AR(1), HAR and HARlog models, Random Forests, LSTM and FNN in Subsample2\n')

table = [['Index', 'MSE AR(1)', 'MSE HAR', 'MSE HARlog', 'MSE_RF', 'MSE LSTM', 'MSE FNN'], 
        ['DJI', round(MSE_AR1_1_DJI_sub2,11), round(MSE_HAR_1_DJI_sub2,11), round(MSE_HARlog_1_DJI_sub2,11), round(MSE_DJI_1_rf_sub2,11), round(MSE_LSTM_1_DJI_sub2,11), round(MSE_FNN_1_DJI_sub2,11)], 
        ['FTSE',round(MSE_AR1_1_FTSE_sub2,11), round(MSE_HAR_1_FTSE_sub2,11), round(MSE_HARlog_1_FTSE_sub2,11), round(MSE_FTSE_1_rf_sub2,11), round(MSE_LSTM_1_FTSE_sub2,11), round(MSE_FNN_1_FTSE_sub2,11)], 
        ['FTSEMIB', round(MSE_AR1_1_FTSEMIB_sub2,11), round(MSE_HAR_1_FTSEMIB_sub2,11), round(MSE_HARlog_1_FTSEMIB_sub2,11), round(MSE_FTSEMIB_1_rf_sub2,11), round(MSE_LSTM_1_FTSEMIB_sub2,11), round(MSE_FNN_1_FTSEMIB_sub2,11)],
        ['GDAXI', round(MSE_AR1_1_GDAXI_sub2,11), round(MSE_HAR_1_GDAXI_sub2,11), round(MSE_HARlog_1_GDAXI_sub2,11), round(MSE_GDAXI_1_rf_sub2,11), round(MSE_LSTM_1_GDAXI_sub2,11), round(MSE_FNN_1_GDAXI_sub2,11)],
        ['SPX', round(MSE_AR1_1_SPX_sub2,11), round(MSE_HAR_1_OMXC20_sub2,11), round(MSE_HARlog_1_SPX_sub2,11),  round(MSE_SPX_1_rf_sub2,11), round(MSE_LSTM_1_SPX_sub2,11), round(MSE_FNN_1_SPX_sub2,11)],
        ['HSI', round(MSE_AR1_1_SPX_sub2,11), round(MSE_HAR_1_HSI_sub2,11), round(MSE_HARlog_1_SPX_sub2,11),  round(MSE_SPX_1_rf_sub2,11), round(MSE_LSTM_1_SPX_sub2,11), round(MSE_FNN_1_SPX_sub2,11)],
        ['IBEX', round(MSE_AR1_1_IBEX_sub2,11), round(MSE_HAR_1_IBEX_sub2,11), round(MSE_HARlog_1_IBEX_sub2,11),  round(MSE_IBEX_1_rf_sub2,11), round(MSE_LSTM_1_IBEX_sub2,11), round(MSE_FNN_1_IBEX_sub2,11)],
        ['IXIC', round(MSE_AR1_1_IXIC_sub2,11), round(MSE_HAR_1_IXIC_sub2,11), round(MSE_HARlog_1_IXIC_sub2,11),  round(MSE_IXIC_1_rf_sub2,11), round(MSE_LSTM_1_IXIC_sub2,11), round(MSE_FNN_1_IXIC_sub2,11)],
        ['N225', round(MSE_AR1_1_N225_sub2,11), round(MSE_HAR_1_N225_sub2,11), round(MSE_HARlog_1_N225_sub2,11),  round(MSE_N225_1_rf_sub2,11), round(MSE_LSTM_1_N225_sub2,11), round(MSE_FNN_1_N225_sub2,11)],
        ['OMXC20', round(MSE_AR1_1_OMXC20_sub2,11), round(MSE_HAR_1_OMXC20_sub2,11), round(MSE_HARlog_1_OMXC20_sub2,11),  round(MSE_OMXC20_1_rf_sub2,11), round(MSE_LSTM_1_OMXC20_sub2,11), round(MSE_FNN_1_OMXC20_sub2,11)]]

print(tabulate(table, headers='firstrow', tablefmt = 'plain'))