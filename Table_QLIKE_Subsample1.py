# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:00:14 2023

@author: cesar
"""

import numpy as np
from tabulate import tabulate

#%%

print('Table of QLIKE between, AR(1), HAR and HARlog models, Random Forests, LSTM, FNN Subsample1\n')

table = [['Index', 'QLIKE AR(1)', 'QLIKE HAR', 'QLIKE HARlog', 'QLIKE RF', 'QLIKE LSTM', 'QLIKE FNN'], 
        ['DJI', np.round(QLIKE_AR1_1_DJI_sub1,3), np.round(QLIKE_HAR_1_DJI_sub1,3), np.round(QLIKE_HARlog_1_DJI_sub1,3), np.round(QLIKE_rf_1_DJI_sub1,3), np.round(QLIKE_LSTM_1_DJI_sub1,3), np.round(QLIKE_FNN_1_DJI_sub1,3)], 
        ['FTSE',np.round(QLIKE_AR1_1_FTSE_sub1,3), np.round(QLIKE_HAR_1_FTSE_sub1,3), np.round(QLIKE_HARlog_1_FTSE_sub1,3), np.round(QLIKE_rf_1_FTSE_sub1,3), np.round(QLIKE_LSTM_1_FTSE_sub1,3), np.round(QLIKE_FNN_1_FTSE_sub1,3)], 
        ['FTSEMIB', np.round(QLIKE_AR1_1_FTSEMIB_sub1,3), np.round(QLIKE_HAR_1_FTSEMIB_sub1,3), np.round(QLIKE_HARlog_1_FTSEMIB_sub1,3), np.round(QLIKE_rf_1_FTSEMIB_sub1,3), np.round(QLIKE_LSTM_1_FTSEMIB_sub1,3), np.round(QLIKE_FNN_1_FTSEMIB_sub1,3)],
        ['GDAXI', np.round(QLIKE_AR1_1_GDAXI_sub1,3), np.round(QLIKE_HAR_1_GDAXI_sub1,3), np.round(QLIKE_HARlog_1_GDAXI_sub1,3), np.round(QLIKE_rf_1_GDAXI_sub1,3), np.round(QLIKE_LSTM_1_GDAXI_sub1,3), np.round(QLIKE_FNN_1_GDAXI_sub1,3)],
        ['SPX', np.round(QLIKE_AR1_1_SPX_sub1,3), np.round(QLIKE_HAR_1_SPX_sub1,3), np.round(QLIKE_HARlog_1_SPX_sub1,3),  np.round(QLIKE_rf_1_SPX_sub1,3), np.round(QLIKE_LSTM_1_SPX_sub1,3), np.round(QLIKE_FNN_1_SPX_sub1,3)],
        ['HSI', np.round(QLIKE_AR1_1_OMXC20_sub1,3), np.round(QLIKE_HAR_1_HSI_sub1,3), np.round(QLIKE_HARlog_1_OMXC20_sub1,3),  np.round(QLIKE_rf_1_OMXC20_sub1,3), np.round(QLIKE_LSTM_1_OMXC20_sub1,3), np.round(QLIKE_FNN_1_OMXC20_sub1,3)],
        ['IBEX', np.round(QLIKE_AR1_1_IBEX_sub1,3), np.round(QLIKE_HAR_1_IBEX_sub1,3), np.round(QLIKE_HARlog_1_IBEX_sub1,3),  np.round(QLIKE_rf_1_IBEX_sub1,3), np.round(QLIKE_LSTM_1_IBEX_sub1,3), np.round(QLIKE_FNN_1_IBEX_sub1,3)],
        ['IXIC', np.round(QLIKE_AR1_1_IXIC_sub1,3), np.round(QLIKE_HAR_1_IXIC_sub1,3), np.round(QLIKE_HARlog_1_IXIC_sub1,3),  np.round(QLIKE_rf_1_IXIC_sub1,3), np.round(QLIKE_LSTM_1_IXIC_sub1,3), np.round(QLIKE_FNN_1_IXIC_sub1,3)],
        ['N225', np.round(QLIKE_AR1_1_N225_sub1,3), np.round(QLIKE_HAR_1_N225_sub1,3), np.round(QLIKE_HARlog_1_N225_sub1,3),  np.round(QLIKE_rf_1_N225_sub1,3), np.round(QLIKE_LSTM_1_N225_sub1,3), np.round(QLIKE_FNN_1_N225_sub1,3)],
        ['OMXC20', np.round(QLIKE_AR1_1_OMXC20_sub1,3), np.round(QLIKE_HAR_1_OMXC20_sub1,3), np.round(QLIKE_HARlog_1_OMXC20_sub1,3),  np.round(QLIKE_rf_1_OMXC20_sub1,3), np.round(QLIKE_LSTM_1_OMXC20_sub1,3), np.round(QLIKE_FNN_1_OMXC20_sub1,3)],]

print(tabulate(table, headers='firstrow', tablefmt = 'plain'))