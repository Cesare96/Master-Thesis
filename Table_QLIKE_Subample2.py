# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 01:12:19 2023

@author: cesar
"""

from tabulate import tabulate

#%%

print('Table of QLIKE between, AR(1), HAR and HARlog models, Random Forests, LSTM and FNN in Subsample2\n')

table = [['Index', 'QLIKE AR(1)', 'QLIKE HAR', 'QLIKE HARlog', 'QLIKE RF', 'QLIKE LSTM', 'QLIKE FNN'], 
        ['DJI', np.round(QLIKE_AR1_1_DJI_sub2,3), np.round(QLIKE_HAR_1_DJI_sub2,3), np.round(QLIKE_HARlog_1_DJI_sub2,3), np.round(QLIKE_rf_1_DJI_sub2,3), np.round(QLIKE_LSTM_1_DJI_sub2,3), np.round(QLIKE_FNN_1_DJI_sub2,3)], 
        ['FTSE',np.round(QLIKE_AR1_1_FTSE_sub2,3), np.round(QLIKE_HAR_1_FTSE_sub2,3), np.round(QLIKE_HARlog_1_FTSE_sub2,3), np.round(QLIKE_rf_1_FTSE_sub2,3), np.round(QLIKE_LSTM_1_FTSE_sub2,3), np.round(QLIKE_FNN_1_FTSE_sub2,3)], 
        ['FTSEMIB', np.round(QLIKE_AR1_1_FTSEMIB_sub2,3), np.round(QLIKE_HAR_1_FTSEMIB_sub2,3), np.round(QLIKE_HARlog_1_FTSEMIB_sub2,3), np.round(QLIKE_rf_1_FTSEMIB_sub2,3), np.round(QLIKE_LSTM_1_FTSEMIB_sub2,3), np.round(QLIKE_FNN_1_FTSEMIB_sub2,3)],
        ['GDAXI', np.round(QLIKE_AR1_1_GDAXI_sub2,3), np.round(QLIKE_HAR_1_GDAXI_sub2,3), np.round(QLIKE_HARlog_1_GDAXI_sub2,3), np.round(QLIKE_rf_1_GDAXI_sub2,3), np.round(QLIKE_LSTM_1_GDAXI_sub2,3), np.round(QLIKE_FNN_1_GDAXI_sub2,3)],
        ['SPX', np.round(QLIKE_AR1_1_SPX_sub2,3), np.round(QLIKE_HAR_1_SPX_sub2,3), np.round(QLIKE_HARlog_1_SPX_sub2,3),  np.round(QLIKE_rf_1_SPX_sub2,3), np.round(QLIKE_LSTM_1_SPX_sub2,3), np.round(QLIKE_FNN_1_SPX_sub2,3)],
        ['HSI', np.round(QLIKE_AR1_1_HSI_sub2,3), np.round(QLIKE_HAR_1_HSI_sub2,3), np.round(QLIKE_HARlog_1_HSI_sub2,3),  np.round(QLIKE_rf_1_HSI_sub2,3), np.round(QLIKE_LSTM_1_HSI_sub2,3), np.round(QLIKE_FNN_1_HSI_sub2,3)],
        ['IBEX', np.round(QLIKE_AR1_1_IBEX_sub2,3), np.round(QLIKE_HAR_1_IBEX_sub2,3), np.round(QLIKE_HARlog_1_IBEX_sub2,3),  np.round(QLIKE_rf_1_IBEX_sub2,3), np.round(QLIKE_LSTM_1_IBEX_sub2,3), np.round(QLIKE_FNN_1_IBEX_sub2,3)],
        ['IXIC', np.round(QLIKE_AR1_1_IXIC_sub2,3), np.round(QLIKE_HAR_1_IXIC_sub2,3), np.round(QLIKE_HARlog_1_IXIC_sub2,3),  np.round(QLIKE_rf_1_IXIC_sub2,3), np.round(QLIKE_LSTM_1_IXIC_sub2,3), np.round(QLIKE_FNN_1_IXIC_sub2,3)],
        ['N225', np.round(QLIKE_AR1_1_N225_sub2,3), np.round(QLIKE_HAR_1_N225_sub2,3), np.round(QLIKE_HARlog_1_N225_sub2,3),  np.round(QLIKE_rf_1_N225_sub2,3), np.round(QLIKE_LSTM_1_N225_sub2,3), np.round(QLIKE_FNN_1_N225_sub2,3)],
        ['OMXC20', np.round(QLIKE_AR1_1_OMXC20_sub2,3), np.round(QLIKE_HAR_1_OMXC20_sub2,3), np.round(QLIKE_HARlog_1_OMXC20_sub2,3),  np.round(QLIKE_rf_1_OMXC20_sub2,3), np.round(QLIKE_LSTM_1_OMXC20_sub2,3), np.round(QLIKE_FNN_1_OMXC20_sub2,3)]]

print(tabulate(table, headers='firstrow', tablefmt = 'plain'))