# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:14:04 2023

@author: cesar
"""
from tabulate import tabulate

#%%

print('Table of QLIKE between, AR(1), HAR and HARlog models, Random Forests, LSTM, FNN\n')

table = [['Index', 'QLIKE AR(1)', 'QLIKE HAR', 'QLIKE HARlog', 'QLIKE RF', 'QLIKE LSTM', 'QLIKE FNN'], 
        ['DJI', np.round(QLIKE_AR1_1_DJI, 4), np.round(QLIKE_HAR_1_DJI, 4), np.round(QLIKE_HARlog_1_DJI, 4), np.round(QLIKE_rf_1_DJI, 4), np.round(QLIKE_LSTM_1_DJI, 4), np.round(QLIKE_FNN_1_DJI, 4)], 
        ['FTSE', np.round(QLIKE_AR1_1_FTSE, 4), np.round(QLIKE_HAR_1_FTSE, 4), np.round(QLIKE_HARlog_1_FTSE, 4), np.round(QLIKE_rf_1_FTSE, 4), np.round(QLIKE_LSTM_1_FTSE, 4), np.round(QLIKE_FNN_1_FTSE, 4)], 
        ['FTSEMIB', np.round(QLIKE_AR1_1_FTSEMIB, 4), np.round(QLIKE_HAR_1_FTSEMIB, 4), np.round(QLIKE_HARlog_1_FTSEMIB, 4), np.round(QLIKE_rf_1_FTSEMIB, 4), np.round(QLIKE_LSTM_1_FTSEMIB, 4), np.round(QLIKE_FNN_1_FTSEMIB, 4)],
        ['GDAXI', np.round(QLIKE_AR1_1_GDAXI, 4), np.round(QLIKE_HAR_1_GDAXI, 4), np.round(QLIKE_HARlog_1_GDAXI, 4), np.round(QLIKE_rf_1_GDAXI, 4), np.round(QLIKE_LSTM_1_GDAXI, 4), np.round(QLIKE_FNN_1_GDAXI, 4)],
        ['SPX', np.round(QLIKE_AR1_1_SPX, 4), np.round(QLIKE_HAR_1_SPX, 4), np.round(QLIKE_HARlog_1_SPX, 4), np.round(QLIKE_rf_1_SPX, 4), np.round(QLIKE_LSTM_1_SPX, 4), np.round(QLIKE_FNN_1_SPX, 4)],
        ['HSI', np.round(QLIKE_AR1_1_HSI, 4), np.round(QLIKE_HAR_1_HSI, 4), np.round(QLIKE_HARlog_1_HSI, 4), np.round(QLIKE_rf_1_HSI, 4), np.round(QLIKE_LSTM_1_HSI, 4), np.round(QLIKE_FNN_1_HSI, 4)],
        ['IBEX', np.round(QLIKE_AR1_1_IBEX, 4), np.round(QLIKE_HAR_1_IBEX, 4), np.round(QLIKE_HARlog_1_IBEX, 4), np.round(QLIKE_rf_1_IBEX, 4), np.round(QLIKE_LSTM_1_IBEX, 4), np.round(QLIKE_FNN_1_IBEX, 4)],
        ['IXIC', np.round(QLIKE_AR1_1_IXIC, 4), np.round(QLIKE_HAR_1_IXIC, 4), np.round(QLIKE_HARlog_1_IXIC, 4), np.round(QLIKE_rf_1_IXIC, 4), np.round(QLIKE_LSTM_1_IXIC,4), np.round(QLIKE_FNN_1_IXIC,4)],
        ['N225', np.round(QLIKE_AR1_1_N225, 4), np.round(QLIKE_HAR_1_N225, 4), np.round(QLIKE_HARlog_1_N225, 4), np.round(QLIKE_rf_1_N225, 4), np.round(QLIKE_LSTM_1_N225, 4), np.round(QLIKE_FNN_1_N225, 4)],
        ['OMXC20', np.round(QLIKE_AR1_1_OMXC20, 4), np.round(QLIKE_HAR_1_OMXC20, 4), np.round(QLIKE_HARlog_1_OMXC20, 4), np.round(QLIKE_rf_1_OMXC20, 4), np.round(QLIKE_LSTM_1_OMXC20, 4), np.round(QLIKE_FNN_1_OMXC20, 4)]]

print(tabulate(table, headers='firstrow', tablefmt = 'plain'))