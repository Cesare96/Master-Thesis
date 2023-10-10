# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:14:36 2022

@author: cesar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
#from statsmodels.tsa.stattools import adfuller as adf
from pandas.plotting import autocorrelation_plot as acf

#%% Loading Data and selecting some Stock Market Indices

df = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
df.rename(columns = {'Unnamed: 0' : 'Date'}, inplace = True)
df.set_index('Date', inplace = True)
FTSE = df[df['Symbol'] == '.FTSE']['rv5']
FTSEMIB = df[df['Symbol'] == '.FTMIB']['rv5']
DJI = df[df['Symbol'] == '.DJI']['rv5']
SPX = df[df['Symbol'] == '.SPX']['rv5']
GDAXI = df[df['Symbol'] == '.GDAXI']['rv5']
HSI = df[df['Symbol'] == '.HSI']['rv5']
IBEX = df[df['Symbol'] == '.IBEX']['rv5']
IXIC = df[df['Symbol'] == '.IXIC']['rv5']
N225 = df[df['Symbol'] == '.N225']['rv5']
OMXC20 = df[df['Symbol'] == '.OMXC20']['rv5'] 

DJI.index = pd.to_datetime(DJI.index, utc = True)
DJI.index = DJI.index.tz_localize(None)
DJI.index = DJI.index.strftime('%Y-%m-%d')
DJI.index = pd.to_datetime(DJI.index, format = '%Y-%m-%d')

FTSE.index = pd.to_datetime(FTSE.index, utc = True)
FTSE.index = FTSE.index.tz_localize(None)
FTSE.index = FTSE.index.strftime('%Y-%m-%d')
FTSE.index = pd.to_datetime(FTSE.index, format = '%Y-%m-%d')

FTSEMIB.index = pd.to_datetime(FTSEMIB.index, utc = True)
FTSEMIB.index = FTSEMIB.index.tz_localize(None)
FTSEMIB.index = FTSEMIB.index.strftime('%Y-%m-%d')
FTSEMIB.index = pd.to_datetime(FTSEMIB.index, format = '%Y-%m-%d')

GDAXI.index = pd.to_datetime(GDAXI.index, utc = True)
GDAXI.index = GDAXI.index.tz_localize(None)
GDAXI.index = GDAXI.index.strftime('%Y-%m-%d')
GDAXI.index = pd.to_datetime(GDAXI.index, format = '%Y-%m-%d')

SPX.index = pd.to_datetime(SPX.index, utc = True)
SPX.index = SPX.index.tz_localize(None)
SPX.index = SPX.index.strftime('%Y-%m-%d')
SPX.index = pd.to_datetime(SPX.index, format = '%Y-%m-%d')

HSI.index = pd.to_datetime(HSI.index, utc = True)
HSI.index = HSI.index.tz_localize(None)
HSI.index = HSI.index.strftime('%Y-%m-%d')
HSI.index = pd.to_datetime(HSI.index, format = '%Y-%m-%d')

IBEX.index = pd.to_datetime(IBEX.index, utc = True)
IBEX.index = IBEX.index.tz_localize(None)
IBEX.index = IBEX.index.strftime('%Y-%m-%d')
IBEX.index = pd.to_datetime(IBEX.index, format = '%Y-%m-%d')

IXIC.index = pd.to_datetime(IXIC.index, utc = True)
IXIC.index = IXIC.index.tz_localize(None)
IXIC.index = IXIC.index.strftime('%Y-%m-%d')
IXIC.index = pd.to_datetime(IXIC.index, format = '%Y-%m-%d')

N225.index = pd.to_datetime(N225.index, utc = True)
N225.index = N225.index.tz_localize(None)
N225.index = N225.index.strftime('%Y-%m-%d')
N225.index = pd.to_datetime(N225.index, format = '%Y-%m-%d')

OMXC20.index = pd.to_datetime(OMXC20.index, utc = True)
OMXC20.index = OMXC20.index.tz_localize(None)
OMXC20.index = OMXC20.index.strftime('%Y-%m-%d')
OMXC20.index = pd.to_datetime(OMXC20.index, format = '%Y-%m-%d')

#%% Creating Regressors

RV_monthly_DJI = DJI.rolling(15).mean()
DJI_forweekly = DJI[15:]
RV_weekly_DJI = DJI_forweekly.rolling(5).mean()
RV_daily_DJI = DJI[20:]
RV_y_DJI = DJI[21:] 

RV_monthly_FTSE = FTSE.rolling(15).mean()
FTSE_forweekly = FTSE[15:]
RV_weekly_FTSE = FTSE_forweekly.rolling(5).mean()
RV_daily_FTSE = FTSE[20:]
RV_y_FTSE = FTSE[21:] 

RV_monthly_FTSEMIB = FTSEMIB.rolling(15).mean()
FTSEMIB_forweekly = FTSEMIB[15:]
RV_weekly_FTSEMIB = FTSEMIB_forweekly.rolling(5).mean()
RV_daily_FTSEMIB = FTSEMIB[20:]
RV_y_FTSEMIB = FTSEMIB[21:] 

RV_monthly_GDAXI = GDAXI.rolling(15).mean()
GDAXI_forweekly = GDAXI[15:]
RV_weekly_GDAXI = GDAXI_forweekly.rolling(5).mean()
RV_daily_GDAXI = GDAXI[20:]
RV_y_GDAXI = GDAXI[21:] 

RV_monthly_SPX = SPX.rolling(15).mean()
SPX_forweekly = SPX[15:]
RV_weekly_SPX = SPX_forweekly.rolling(5).mean()
RV_daily_SPX= SPX[20:]
RV_y_SPX = SPX[21:] 

RV_monthly_HSI = HSI.rolling(15).mean()
HSI_forweekly = HSI[15:]
RV_weekly_HSI = HSI_forweekly.rolling(5).mean()
RV_daily_HSI= HSI[20:]
RV_y_HSI = HSI[21:] 

RV_monthly_IBEX = IBEX.rolling(15).mean()
IBEX_forweekly = IBEX[15:]
RV_weekly_IBEX = IBEX_forweekly.rolling(5).mean()
RV_daily_IBEX= IBEX[20:]
RV_y_IBEX = IBEX[21:] 

RV_monthly_IXIC = IXIC.rolling(15).mean()
IXIC_forweekly = IXIC[15:]
RV_weekly_IXIC = IXIC_forweekly.rolling(5).mean()
RV_daily_IXIC= IXIC[20:]
RV_y_IXIC = IXIC[21:] 

RV_monthly_N225 = N225.rolling(15).mean()
N225_forweekly = N225[15:]
RV_weekly_N225 = N225_forweekly.rolling(5).mean()
RV_daily_N225= N225[20:]
RV_y_N225 = N225[21:] 

RV_monthly_OMXC20 = OMXC20.rolling(15).mean()
OMXC20_forweekly = OMXC20[15:]
RV_weekly_OMXC20 = OMXC20_forweekly.rolling(5).mean()
RV_daily_OMXC20= OMXC20[20:]
RV_y_OMXC20 = OMXC20[21:] 

#%% Creating the DataFrame for training the models

df_DJI = pd.concat([RV_daily_DJI, RV_weekly_DJI, RV_monthly_DJI, RV_y_DJI], axis = 1)
df_FTSE = pd.concat([RV_daily_FTSE, RV_weekly_FTSE, RV_monthly_FTSE, RV_y_FTSE], axis = 1)
df_FTSEMIB = pd.concat([RV_daily_FTSEMIB, RV_weekly_FTSEMIB, RV_monthly_FTSEMIB, RV_y_FTSEMIB], axis = 1)
df_GDAXI = pd.concat([RV_daily_GDAXI, RV_weekly_GDAXI, RV_monthly_GDAXI, RV_y_GDAXI], axis = 1)
df_SPX = pd.concat([RV_daily_SPX, RV_weekly_SPX, RV_monthly_SPX, RV_y_SPX], axis = 1)
df_HSI = pd.concat([RV_daily_HSI, RV_weekly_HSI, RV_monthly_HSI, RV_y_HSI], axis = 1)
df_IBEX = pd.concat([RV_daily_IBEX, RV_weekly_IBEX, RV_monthly_IBEX, RV_y_IBEX], axis = 1)
df_IXIC = pd.concat([RV_daily_IXIC, RV_weekly_IXIC, RV_monthly_IXIC, RV_y_IXIC], axis = 1)
df_N225 = pd.concat([RV_daily_N225, RV_weekly_N225, RV_monthly_N225, RV_y_N225], axis = 1)
df_OMXC20 = pd.concat([RV_daily_OMXC20, RV_weekly_OMXC20, RV_monthly_OMXC20, RV_y_OMXC20], axis = 1)

#%% Renaming the columns

col_names = ['daily', 'weekly', 'monthly', 'y']
df_DJI.columns = col_names
df_FTSE.columns = col_names
df_FTSEMIB.columns = col_names
df_GDAXI.columns = col_names
df_SPX.columns = col_names
df_HSI.columns = col_names
df_IBEX.columns = col_names
df_IXIC.columns = col_names
df_N225.columns =col_names
df_OMXC20.columns = col_names

df_DJI['daily'] = df_DJI.daily.shift()
df_FTSE['daily'] = df_FTSE.daily.shift()
df_FTSEMIB['daily'] = df_FTSEMIB.daily.shift()
df_GDAXI['daily'] = df_GDAXI.daily.shift()
df_SPX['daily'] = df_SPX.daily.shift()
df_HSI['daily'] = df_HSI.daily.shift()
df_IBEX['daily'] = df_IBEX.daily.shift()
df_IXIC['daily'] = df_IXIC.daily.shift()
df_N225['daily'] = df_N225.daily.shift()
df_OMXC20['daily'] = df_OMXC20.daily.shift()

df_DJI['weekly'] = df_DJI.weekly.shift(2)
df_FTSE['weekly'] = df_FTSE.weekly.shift(2)
df_FTSEMIB['weekly'] = df_FTSEMIB.weekly.shift(2)
df_GDAXI['weekly'] = df_GDAXI.weekly.shift(2)
df_SPX['weekly'] = df_SPX.weekly.shift(2)
df_HSI['weekly'] = df_HSI.weekly.shift(2)
df_IBEX['weekly'] = df_IBEX.weekly.shift(2)
df_IXIC['weekly'] = df_IXIC.weekly.shift(2)
df_N225['weekly'] = df_N225.weekly.shift(2)
df_OMXC20['weekly'] = df_OMXC20.weekly.shift(2)

df_DJI['monthly'] = df_DJI.monthly.shift(7)
df_FTSE['monthly'] = df_FTSE.monthly.shift(7)
df_FTSEMIB['monthly'] = df_FTSEMIB.monthly.shift(7)
df_GDAXI['monthly'] = df_GDAXI.monthly.shift(7)
df_SPX['monthly'] = df_SPX.monthly.shift(7)
df_HSI['monthly'] = df_HSI.monthly.shift(7)
df_IBEX['monthly'] = df_IBEX.monthly.shift(7)
df_IXIC['monthly'] = df_IXIC.monthly.shift(7)
df_N225['monthly'] = df_N225.monthly.shift(7)
df_OMXC20['monthly'] = df_OMXC20.monthly.shift(7)

#%% logDataFrame for HARlog

df_DJIlog = np.log(df_DJI)
df_FTSElog = np.log(df_FTSE)
df_FTSEMIBlog = np.log(df_FTSEMIB)
df_GDAXIlog = np.log(df_GDAXI)
df_SPXlog = np.log(df_SPX)
df_HSIlog = np.log(df_HSI)
df_IBEXlog = np.log(df_IBEX)
df_IXIClog = np.log(df_IXIC)
df_N225log = np.log(df_N225)
df_OMXC20log = np.log(df_OMXC20)

df_DJI.dropna(inplace = True)
df_FTSE.dropna(inplace = True)
df_FTSEMIB.dropna(inplace = True)
df_GDAXI.dropna(inplace = True)
df_SPX.dropna(inplace = True)
df_HSI.dropna(inplace = True)
df_IBEX.dropna(inplace = True)
df_IXIC.dropna(inplace = True)
df_N225.dropna(inplace = True)
df_OMXC20.dropna(inplace = True)

df_DJIlog.dropna(inplace = True)
df_FTSElog.dropna(inplace = True)
df_FTSEMIBlog.dropna(inplace = True)
df_GDAXIlog.dropna(inplace = True)
df_SPXlog.dropna(inplace = True)
df_HSIlog.dropna(inplace = True)
df_IBEXlog.dropna(inplace = True)
df_IXIClog.dropna(inplace = True)
df_N225log.dropna(inplace = True)
df_OMXC20log.dropna(inplace = True)

#%% Splitting the data in Train and Test Sets fo HAR and HARlog

X_DJI = df_DJI[['daily', 'weekly', 'monthly']]
X_FTSE = df_FTSE[['daily', 'weekly', 'monthly']]
X_FTSEMIB = df_FTSEMIB[['daily', 'weekly', 'monthly']]
X_GDAXI = df_GDAXI[['daily', 'weekly', 'monthly']]
X_SPX = df_SPX[['daily', 'weekly', 'monthly']]
X_HSI = df_HSI[['daily', 'weekly', 'monthly']]
X_IBEX = df_IBEX[['daily', 'weekly', 'monthly']]
X_IXIC = df_IXIC[['daily', 'weekly', 'monthly']]
X_N225 = df_N225[['daily', 'weekly', 'monthly']]
X_OMXC20 = df_OMXC20[['daily', 'weekly', 'monthly']]

X_DJIlog = df_DJIlog[['daily', 'weekly', 'monthly']]
X_FTSElog = df_FTSElog[['daily', 'weekly', 'monthly']]
X_FTSEMIBlog = df_FTSEMIBlog[['daily', 'weekly', 'monthly']]
X_GDAXIlog = df_GDAXIlog[['daily', 'weekly', 'monthly']]
X_SPXlog = df_SPXlog[['daily', 'weekly', 'monthly']]
X_HSIlog = df_HSIlog[['daily', 'weekly', 'monthly']]
X_IBEXlog = df_IBEXlog[['daily', 'weekly', 'monthly']]
X_IXIClog = df_IXIClog[['daily', 'weekly', 'monthly']]
X_N225log = df_N225log[['daily', 'weekly', 'monthly']]
X_OMXC20log = df_OMXC20log[['daily', 'weekly', 'monthly']]

y_DJI = df_DJI['y']
y_FTSE = df_FTSE['y']
y_FTSEMIB = df_FTSEMIB['y']
y_GDAXI = df_GDAXI['y']
y_SPX = df_SPX['y']
y_HSI = df_HSI['y']
y_IBEX = df_IBEX['y']
y_IXIC = df_IXIC['y']
y_N225 = df_N225['y']
y_OMXC20 = df_OMXC20['y']

y_DJIlog = df_DJIlog['y']
y_FTSElog = df_FTSElog['y']
y_FTSEMIBlog = df_FTSEMIBlog['y']
y_GDAXIlog = df_GDAXIlog['y']
y_SPXlog = df_SPXlog['y']
y_HSIlog = df_HSIlog['y']
y_IBEXlog = df_IBEXlog['y']
y_IXIClog = df_IXIClog['y']
y_N225log = df_N225log['y']
y_OMXC20log = df_OMXC20log['y']

threshold_train_DJI = '2015-08-12'
thershold_train_DJI = pd.to_datetime(threshold_train_DJI)
threshold_test_DJI = '2015-08-13'
thershold_test_DJI = pd.to_datetime(threshold_test_DJI)

threshold_train_FTSE = '2015-08-24'
thershold_train_FTSE = pd.to_datetime(threshold_train_FTSE)
threshold_test_FTSE = '2015-08-25'
thershold_test_FTSE = pd.to_datetime(threshold_test_FTSE)

threshold_train_FTSEMIB = '2018-06-06'
thershold_train_FTSEMIB = pd.to_datetime(threshold_train_FTSEMIB)
threshold_test_FTSEMIB = '2018-06-07'
thershold_test_FTSEMIB = pd.to_datetime(threshold_test_FTSEMIB)

threshold_train_GDAXI = '2015-08-10'
thershold_train_GDAXI = pd.to_datetime(threshold_train_GDAXI)
threshold_test_GDAXI = '2015-08-11'
threshold_test_GDAXI = pd.to_datetime(threshold_test_GDAXI)

threshold_train_SPX = '2015-08-17'
thershold_train_SPX = pd.to_datetime(threshold_train_SPX)
threshold_test_SPX = '2015-08-18'
thershold_test_SPX = pd.to_datetime(threshold_test_SPX)

threshold_train_HSI = '2015-08-16'
thershold_train_HSI = pd.to_datetime(threshold_train_HSI)
threshold_test_HSI = '2015-08-17'
thershold_test_HSI = pd.to_datetime(threshold_test_HSI)

threshold_train_IBEX = '2015-09-03'
thershold_train_IBEX = pd.to_datetime(threshold_train_IBEX)
threshold_test_IBEX = '2015-09-06'
thershold_test_IBEX = pd.to_datetime(threshold_test_IBEX)

threshold_train_IXIC = '2015-08-23'
thershold_train_IXIC = pd.to_datetime(threshold_train_IXIC)
threshold_test_IXIC = '2015-08-24'
thershold_test_IXIC = pd.to_datetime(threshold_test_IXIC)

threshold_train_N225 = '2015-08-11'
thershold_train_N225 = pd.to_datetime(threshold_train_N225)
threshold_test_N225 = '2015-08-12'
thershold_test_N225 = pd.to_datetime(threshold_test_N225)

threshold_train_OMXC20 = '2017-05-04'
thershold_train_OMXC20 = pd.to_datetime(threshold_train_OMXC20)
threshold_test_OMXC20 = '2017-05-07'
thershold_test_OMXC20 = pd.to_datetime(threshold_test_OMXC20)

X_train_DJI = X_DJI[:threshold_train_DJI]
X_train_FTSE = X_FTSE[:thershold_train_FTSE]
X_train_FTSEMIB = X_FTSEMIB[:threshold_train_FTSEMIB]
X_train_GDAXI = X_GDAXI[:threshold_train_GDAXI]
X_train_SPX = X_SPX[:threshold_train_SPX]
X_train_HSI = X_HSI[:threshold_train_HSI]
X_train_IBEX = X_IBEX[:threshold_train_IBEX]
X_train_IXIC = X_IXIC[:threshold_train_IXIC]
X_train_N225 = X_N225[:threshold_train_N225]
X_train_OMXC20 = X_OMXC20[:threshold_train_OMXC20]

y_train_DJI = y_DJI[:threshold_train_DJI]
y_train_FTSE = y_FTSE[:thershold_train_FTSE]
y_train_FTSEMIB = y_FTSEMIB[:threshold_train_FTSEMIB]
y_train_GDAXI = y_GDAXI[:threshold_train_GDAXI]
y_train_SPX = y_SPX[:threshold_train_SPX]
y_train_HSI = y_HSI[:threshold_train_HSI]
y_train_IBEX = y_IBEX[:threshold_train_IBEX]
y_train_IXIC = y_IXIC[:threshold_train_IXIC]
y_train_N225 = y_N225[:threshold_train_N225]
y_train_OMXC20 = y_OMXC20[:threshold_train_OMXC20]

X_test_DJI = X_DJI[threshold_test_DJI:]
X_test_FTSE = X_FTSE[threshold_test_FTSE:]
X_test_FTSEMIB = X_FTSEMIB[threshold_test_FTSEMIB:]
X_test_GDAXI = X_GDAXI[threshold_test_GDAXI:]
X_test_SPX = X_SPX[threshold_test_SPX:]
X_test_HSI = X_HSI[threshold_test_HSI:]
X_test_IBEX = X_IBEX[threshold_test_IBEX:]
X_test_IXIC = X_IXIC[threshold_test_IXIC:]
X_test_N225 = X_N225[threshold_test_N225:]
X_test_OMXC20 = X_OMXC20[threshold_test_OMXC20:]


y_test_DJI = y_DJI[threshold_test_DJI:]
y_test_DJI = y_test_DJI[21:]
y_test_FTSE = y_FTSE[threshold_test_FTSE:]
y_test_FTSE = y_test_FTSE[21:]
y_test_FTSEMIB = y_FTSEMIB[threshold_test_FTSEMIB:]
y_test_FTSEMIB = y_test_FTSEMIB[21:]
y_test_GDAXI = y_GDAXI[threshold_test_GDAXI:]
y_test_GDAXI = y_test_GDAXI[21:]
y_test_SPX = y_SPX[threshold_test_SPX:]
y_test_SPX = y_test_SPX[21:]
y_test_HSI = y_HSI[threshold_test_HSI:]
y_test_HSI = y_test_HSI[21:]
y_test_IBEX = y_IBEX[threshold_test_IBEX:]
y_test_IBEX = y_test_IBEX[21:]
y_test_IXIC = y_IXIC[threshold_test_IXIC:]
y_test_IXIC = y_test_IXIC[21:]
y_test_N225 = y_N225[threshold_test_N225:]
y_test_N225 = y_test_N225[21:]
y_test_OMXC20 = y_OMXC20[threshold_test_OMXC20:]
y_test_OMXC20 = y_test_OMXC20[21:]

X_train_DJIlog = X_DJIlog[:threshold_train_DJI]
X_train_FTSElog = X_FTSElog[:thershold_train_FTSE]
X_train_FTSEMIBlog = X_FTSEMIBlog[:threshold_train_FTSEMIB]
X_train_GDAXIlog = X_GDAXIlog[:threshold_train_GDAXI]
X_train_SPXlog = X_SPXlog[:threshold_train_SPX]
X_train_HSIlog = X_HSIlog[:threshold_train_HSI]
X_train_IBEXlog = X_IBEXlog[:threshold_train_IBEX]
X_train_IXIClog = X_IXIClog[:threshold_train_IXIC]
X_train_N225log = X_N225log[:threshold_train_N225]
X_train_OMXC20log = X_OMXC20log[:threshold_train_OMXC20]

y_train_DJIlog = y_DJIlog[:threshold_train_DJI]
y_train_FTSElog = y_FTSElog[:thershold_train_FTSE]
y_train_FTSEMIBlog = y_FTSEMIBlog[:threshold_train_FTSEMIB]
y_train_GDAXIlog = y_GDAXIlog[:threshold_train_GDAXI]
y_train_SPXlog = y_SPXlog[:threshold_train_SPX]
y_train_HSIlog = y_HSIlog[:threshold_train_HSI]
y_train_IBEXlog = y_IBEXlog[:threshold_train_IBEX]
y_train_IXIClog = y_IXIClog[:threshold_train_IXIC]
y_train_N225log = y_N225log[:threshold_train_N225]
y_train_OMXC20log = y_OMXC20log[:threshold_train_OMXC20]

X_test_DJIlog = X_DJIlog[threshold_test_DJI:]
X_test_FTSElog = X_FTSElog[threshold_test_FTSE:]
X_test_FTSEMIBlog = X_FTSEMIBlog[threshold_test_FTSEMIB:]
X_test_GDAXIlog = X_GDAXIlog[threshold_test_GDAXI:]
X_test_SPXlog = X_SPXlog[threshold_test_SPX:]
X_test_HSIlog = X_HSIlog[threshold_test_HSI:]
X_test_IBEXlog = X_IBEXlog[threshold_test_IBEX:]
X_test_IXIClog = X_IXIClog[threshold_test_IXIC:]
X_test_N225log = X_N225log[threshold_test_N225:]
X_test_OMXC20log = X_OMXC20log[threshold_test_OMXC20:]


#%% Plotting 
'''
plt.figure(figsize = (16,8))
plt.plot(X_train_DJI.daily, label = 'Train Set')
plt.plot(X_test_DJI.daily, label = 'Test Set')
plt.title('Train and Test set of DJI daily Realized Volatility')
plt.xlabel('Time')
plt.ylabel('RV')
plt.axvline(x = X_train_DJI.index[3901], color = 'black', ls = '--', linewidth = 0.5)
plt.legend(loc = 'upper left')
plt.show()
'''

#%% Fitting AR(1)

# Train and test set

# DJI

data1_AR1_DJI = DJI[1:]
data2_AR1_DJI = DJI.shift(1)[1:]
data_AR1_DJI = pd.concat((data1_AR1_DJI, data2_AR1_DJI), axis = 1)
train_AR1_DJI, test_AR1_DJI = train_test_split(data_AR1_DJI, test_size=0.30, shuffle = False)
y_train_AR1_DJI = train_AR1_DJI.iloc[:,0]
X_train_AR1_DJI = train_AR1_DJI.iloc[:,1]
y_test_AR1_DJI = test_AR1_DJI.iloc[:,0]
y_test_AR1_DJI = y_test_AR1_DJI[21:]
X_test_AR1_DJI = test_AR1_DJI.iloc[:,1]

X_train_AR1_DJI_c = sm.add_constant(X_train_AR1_DJI)
X_test_AR1_DJI_c = sm.add_constant(X_test_AR1_DJI)

model_AR1_DJI = sm.OLS(y_train_AR1_DJI, X_train_AR1_DJI_c).fit()
y_pred_AR1_1_DJI = model_AR1_DJI.predict(X_test_AR1_DJI_c[21:])
MSE_AR1_1_DJI = mean_squared_error(y_test_AR1_DJI, y_pred_AR1_1_DJI)

# FTSE

data1_AR1_FTSE = FTSE[1:]
data2_AR1_FTSE = FTSE.shift(1)[1:]
data_AR1_FTSE = pd.concat((data1_AR1_FTSE, data2_AR1_FTSE), axis = 1)
train_AR1_FTSE, test_AR1_FTSE = train_test_split(data_AR1_FTSE, test_size=0.30, shuffle = False)
y_train_AR1_FTSE = train_AR1_FTSE.iloc[:,0]
X_train_AR1_FTSE = train_AR1_FTSE.iloc[:,1]
y_test_AR1_FTSE = test_AR1_FTSE.iloc[:,0]
y_test_AR1_FTSE = y_test_AR1_FTSE[21:]
X_test_AR1_FTSE = test_AR1_FTSE.iloc[:,1]

X_train_AR1_FTSE_c = sm.add_constant(X_train_AR1_FTSE)
X_test_AR1_FTSE_c = sm.add_constant(X_test_AR1_FTSE)

model_AR1_FTSE = sm.OLS(y_train_AR1_FTSE, X_train_AR1_FTSE_c).fit()
y_pred_AR1_1_FTSE = model_AR1_FTSE.predict(X_test_AR1_FTSE_c[21:])
MSE_AR1_1_FTSE = mean_squared_error(y_test_AR1_FTSE, y_pred_AR1_1_FTSE)

# FTSEMIB

data1_AR1_FTSEMIB = FTSEMIB[1:]
data2_AR1_FTSEMIB = FTSEMIB.shift(1)[1:]
data_AR1_FTSEMIB = pd.concat((data1_AR1_FTSEMIB, data2_AR1_FTSEMIB), axis = 1)
train_AR1_FTSEMIB, test_AR1_FTSEMIB = train_test_split(data_AR1_FTSEMIB, test_size=0.30, shuffle = False)
y_train_AR1_FTSEMIB = train_AR1_FTSEMIB.iloc[:,0]
X_train_AR1_FTSEMIB = train_AR1_FTSEMIB.iloc[:,1]
y_test_AR1_FTSEMIB = test_AR1_FTSEMIB.iloc[:,0]
y_test_AR1_FTSEMIB = y_test_AR1_FTSEMIB[21:]
X_test_AR1_FTSEMIB = test_AR1_FTSEMIB.iloc[:,1]

X_train_AR1_FTSEMIB_c = sm.add_constant(X_train_AR1_FTSEMIB)
X_test_AR1_FTSEMIB_c = sm.add_constant(X_test_AR1_FTSEMIB)

model_AR1_FTSEMIB = sm.OLS(y_train_AR1_FTSEMIB, X_train_AR1_FTSEMIB_c).fit()
y_pred_AR1_1_FTSEMIB = model_AR1_FTSEMIB.predict(X_test_AR1_FTSEMIB_c[21:])
MSE_AR1_1_FTSEMIB = mean_squared_error(y_test_AR1_FTSEMIB, y_pred_AR1_1_FTSEMIB)

# GDAXI

data1_AR1_GDAXI = GDAXI[1:]
data2_AR1_GDAXI = GDAXI.shift(1)[1:]
data_AR1_GDAXI = pd.concat((data1_AR1_GDAXI, data2_AR1_GDAXI), axis = 1)
train_AR1_GDAXI, test_AR1_GDAXI = train_test_split(data_AR1_GDAXI, test_size=0.30, shuffle = False)
y_train_AR1_GDAXI = train_AR1_GDAXI.iloc[:,0]
X_train_AR1_GDAXI = train_AR1_GDAXI.iloc[:,1]
y_test_AR1_GDAXI = test_AR1_GDAXI.iloc[:,0]
y_test_AR1_GDAXI = y_test_AR1_GDAXI[21:]
X_test_AR1_GDAXI = test_AR1_GDAXI.iloc[:,1]

X_train_AR1_GDAXI_c = sm.add_constant(X_train_AR1_GDAXI)
X_test_AR1_GDAXI_c = sm.add_constant(X_test_AR1_GDAXI)

model_AR1_GDAXI = sm.OLS(y_train_AR1_GDAXI, X_train_AR1_GDAXI_c).fit()
y_pred_AR1_1_GDAXI = model_AR1_GDAXI.predict(X_test_AR1_GDAXI_c[21:])
MSE_AR1_1_GDAXI = mean_squared_error(y_test_AR1_GDAXI, y_pred_AR1_1_GDAXI)

# SPX

data1_AR1_SPX = SPX[1:]
data2_AR1_SPX = SPX.shift(1)[1:]
data_AR1_SPX = pd.concat((data1_AR1_SPX, data2_AR1_SPX), axis = 1)
train_AR1_SPX, test_AR1_SPX = train_test_split(data_AR1_SPX, test_size=0.30, shuffle = False)
y_train_AR1_SPX = train_AR1_SPX.iloc[:,0]
X_train_AR1_SPX = train_AR1_SPX.iloc[:,1]
y_test_AR1_SPX = test_AR1_SPX.iloc[:,0]
y_test_AR1_SPX = y_test_AR1_SPX[21:]
X_test_AR1_SPX = test_AR1_SPX.iloc[:,1]

X_train_AR1_SPX_c = sm.add_constant(X_train_AR1_SPX)
X_test_AR1_SPX_c = sm.add_constant(X_test_AR1_SPX)

model_AR1_SPX = sm.OLS(y_train_AR1_SPX, X_train_AR1_SPX_c).fit()
y_pred_AR1_1_SPX = model_AR1_SPX.predict(X_test_AR1_SPX_c[21:])
MSE_AR1_1_SPX = mean_squared_error(y_test_AR1_SPX, y_pred_AR1_1_SPX)

# HSI

data1_AR1_HSI = HSI[1:]
data2_AR1_HSI = HSI.shift(1)[1:]
data_AR1_HSI = pd.concat((data1_AR1_HSI, data2_AR1_HSI), axis = 1)
train_AR1_HSI, test_AR1_HSI = train_test_split(data_AR1_HSI, test_size=0.30, shuffle = False)
y_train_AR1_HSI = train_AR1_HSI.iloc[:,0]
X_train_AR1_HSI = train_AR1_HSI.iloc[:,1]
y_test_AR1_HSI = test_AR1_HSI.iloc[:,0]
y_test_AR1_HSI = y_test_AR1_HSI[21:]
X_test_AR1_HSI = test_AR1_HSI.iloc[:,1]

X_train_AR1_HSI_c = sm.add_constant(X_train_AR1_HSI)
X_test_AR1_HSI_c = sm.add_constant(X_test_AR1_HSI)

model_AR1_HSI = sm.OLS(y_train_AR1_HSI, X_train_AR1_HSI_c).fit()
y_pred_AR1_1_HSI = model_AR1_HSI.predict(X_test_AR1_HSI_c[21:])
MSE_AR1_1_HSI = mean_squared_error(y_test_AR1_HSI, y_pred_AR1_1_HSI)

# IBEX

data1_AR1_IBEX = IBEX[1:]
data2_AR1_IBEX = IBEX.shift(1)[1:]
data_AR1_IBEX = pd.concat((data1_AR1_IBEX, data2_AR1_IBEX), axis = 1)
train_AR1_IBEX, test_AR1_IBEX = train_test_split(data_AR1_IBEX, test_size=0.30, shuffle = False)
y_train_AR1_IBEX = train_AR1_IBEX.iloc[:,0]
X_train_AR1_IBEX = train_AR1_IBEX.iloc[:,1]
y_test_AR1_IBEX = test_AR1_IBEX.iloc[:,0]
y_test_AR1_IBEX = y_test_AR1_IBEX[21:]
X_test_AR1_IBEX = test_AR1_IBEX.iloc[:,1]

X_train_AR1_IBEX_c = sm.add_constant(X_train_AR1_IBEX)
X_test_AR1_IBEX_c = sm.add_constant(X_test_AR1_IBEX)

model_AR1_IBEX = sm.OLS(y_train_AR1_IBEX, X_train_AR1_IBEX_c).fit()
y_pred_AR1_1_IBEX = model_AR1_IBEX.predict(X_test_AR1_IBEX_c[21:])
MSE_AR1_1_IBEX = mean_squared_error(y_test_AR1_IBEX, y_pred_AR1_1_IBEX)

# IXIC

data1_AR1_IXIC = IXIC[1:]
data2_AR1_IXIC = IXIC.shift(1)[1:]
data_AR1_IXIC = pd.concat((data1_AR1_IXIC, data2_AR1_IXIC), axis = 1)
train_AR1_IXIC, test_AR1_IXIC = train_test_split(data_AR1_IXIC, test_size=0.30, shuffle = False)
y_train_AR1_IXIC = train_AR1_IXIC.iloc[:,0]
X_train_AR1_IXIC = train_AR1_IXIC.iloc[:,1]
y_test_AR1_IXIC = test_AR1_IXIC.iloc[:,0]
y_test_AR1_IXIC = y_test_AR1_IXIC[21:]
X_test_AR1_IXIC = test_AR1_IXIC.iloc[:,1]

X_train_AR1_IXIC_c = sm.add_constant(X_train_AR1_IXIC)
X_test_AR1_IXIC_c = sm.add_constant(X_test_AR1_IXIC)

model_AR1_IXIC = sm.OLS(y_train_AR1_IXIC, X_train_AR1_IXIC_c).fit()
y_pred_AR1_1_IXIC = model_AR1_IXIC.predict(X_test_AR1_IXIC_c[21:])
MSE_AR1_1_IXIC = mean_squared_error(y_test_AR1_IXIC, y_pred_AR1_1_IXIC)

# N225

data1_AR1_N225 = N225[1:]
data2_AR1_N225 = N225.shift(1)[1:]
data_AR1_N225 = pd.concat((data1_AR1_N225, data2_AR1_N225), axis = 1)
train_AR1_N225, test_AR1_N225 = train_test_split(data_AR1_N225, test_size=0.30, shuffle = False)
y_train_AR1_N225 = train_AR1_N225.iloc[:,0]
X_train_AR1_N225 = train_AR1_N225.iloc[:,1]
y_test_AR1_N225 = test_AR1_N225.iloc[:,0]
y_test_AR1_N225 = y_test_AR1_N225[21:]
X_test_AR1_N225 = test_AR1_N225.iloc[:,1]

X_train_AR1_N225_c = sm.add_constant(X_train_AR1_N225)
X_test_AR1_N225_c = sm.add_constant(X_test_AR1_N225)

model_AR1_N225 = sm.OLS(y_train_AR1_N225, X_train_AR1_N225_c).fit()
y_pred_AR1_1_N225 = model_AR1_N225.predict(X_test_AR1_N225_c[21:])
MSE_AR1_1_N225 = mean_squared_error(y_test_AR1_N225, y_pred_AR1_1_N225)

# OMXC20

data1_AR1_OMXC20 = OMXC20[1:]
data2_AR1_OMXC20 = OMXC20.shift(1)[1:]
data_AR1_OMXC20 = pd.concat((data1_AR1_OMXC20, data2_AR1_OMXC20), axis = 1)
train_AR1_OMXC20, test_AR1_OMXC20 = train_test_split(data_AR1_OMXC20, test_size=0.30, shuffle = False)
y_train_AR1_OMXC20 = train_AR1_OMXC20.iloc[:,0]
X_train_AR1_OMXC20 = train_AR1_OMXC20.iloc[:,1]
y_test_AR1_OMXC20 = test_AR1_OMXC20.iloc[:,0]
y_test_AR1_OMXC20 = y_test_AR1_OMXC20[21:]
X_test_AR1_OMXC20 = test_AR1_OMXC20.iloc[:,1]

X_train_AR1_OMXC20_c = sm.add_constant(X_train_AR1_OMXC20)
X_test_AR1_OMXC20_c = sm.add_constant(X_test_AR1_OMXC20)

model_AR1_OMXC20 = sm.OLS(y_train_AR1_OMXC20, X_train_AR1_OMXC20_c).fit()
y_pred_AR1_1_OMXC20 = model_AR1_OMXC20.predict(X_test_AR1_OMXC20_c[21:])
MSE_AR1_1_OMXC20 = mean_squared_error(y_test_AR1_OMXC20, y_pred_AR1_1_OMXC20)

#%% Fitting HAR and HARlog models

X_train_DJI_c = sm.add_constant(X_train_DJI)
X_train_FTSE_c = sm.add_constant(X_train_FTSE)         
X_train_FTSEMIB_c = sm.add_constant(X_train_FTSEMIB)
X_train_GDAXI_c = sm.add_constant(X_train_GDAXI)
X_train_SPX_c = sm.add_constant(X_train_SPX)
X_train_HSI_c = sm.add_constant(X_train_HSI)
X_train_IBEX_c = sm.add_constant(X_train_IBEX)
X_train_IXIC_c = sm.add_constant(X_train_IXIC)
X_train_N225_c = sm.add_constant(X_train_N225)
X_train_OMXC20_c = sm.add_constant(X_train_OMXC20)

X_test_DJI_c = sm.add_constant(X_test_DJI)
X_test_FTSE_c = sm.add_constant(X_test_FTSE)         
X_test_FTSEMIB_c = sm.add_constant(X_test_FTSEMIB)
X_test_GDAXI_c = sm.add_constant(X_test_GDAXI)
X_test_SPX_c = sm.add_constant(X_test_SPX)
X_test_HSI_c = sm.add_constant(X_test_HSI)
X_test_IBEX_c = sm.add_constant(X_test_IBEX)
X_test_IXIC_c = sm.add_constant(X_test_IXIC)
X_test_N225_c = sm.add_constant(X_test_N225)
X_test_OMXC20_c = sm.add_constant(X_test_OMXC20)

X_train_DJIlog_c = sm.add_constant(X_train_DJIlog)
X_train_FTSElog_c = sm.add_constant(X_train_FTSElog)         
X_train_FTSEMIBlog_c = sm.add_constant(X_train_FTSEMIBlog)
X_train_GDAXIlog_c = sm.add_constant(X_train_GDAXIlog)
X_train_SPXlog_c = sm.add_constant(X_train_SPXlog)
X_train_HSIlog_c = sm.add_constant(X_train_HSIlog)
X_train_IBEXlog_c = sm.add_constant(X_train_IBEXlog)
X_train_IXIClog_c = sm.add_constant(X_train_IXIClog)
X_train_N225log_c = sm.add_constant(X_train_N225log)
X_train_OMXC20log_c = sm.add_constant(X_train_OMXC20log)

X_test_DJIlog_c = sm.add_constant(X_test_DJIlog)
X_test_FTSElog_c = sm.add_constant(X_test_FTSElog)         
X_test_FTSEMIBlog_c = sm.add_constant(X_test_FTSEMIBlog)
X_test_GDAXIlog_c = sm.add_constant(X_test_GDAXIlog)
X_test_SPXlog_c = sm.add_constant(X_test_SPXlog)
X_test_HSIlog_c = sm.add_constant(X_test_HSIlog)
X_test_IBEXlog_c = sm.add_constant(X_test_IBEXlog)
X_test_IXIClog_c = sm.add_constant(X_test_IXIClog)
X_test_N225log_c = sm.add_constant(X_test_N225log)
X_test_OMXC20log_c = sm.add_constant(X_test_OMXC20log)

regHAR_DJI = sm.OLS(y_train_DJI, X_train_DJI_c).fit()
regHAR_FTSE = sm.OLS(y_train_FTSE, X_train_FTSE_c).fit()
regHAR_FTSEMIB = sm.OLS(y_train_FTSEMIB, X_train_FTSEMIB_c).fit()
regHAR_GDAXI = sm.OLS(y_train_GDAXI, X_train_GDAXI_c).fit()
regHAR_SPX = sm.OLS(y_train_SPX, X_train_SPX_c).fit()
regHAR_HSI = sm.OLS(y_train_HSI, X_train_HSI_c).fit()
regHAR_IBEX = sm.OLS(y_train_IBEX, X_train_IBEX_c).fit()
regHAR_IXIC = sm.OLS(y_train_IXIC, X_train_IXIC_c).fit()
regHAR_N225 = sm.OLS(y_train_N225, X_train_N225_c).fit()
regHAR_OMXC20 = sm.OLS(y_train_OMXC20, X_train_OMXC20_c).fit()

regHARlog_DJI = sm.OLS(y_train_DJIlog, X_train_DJIlog_c).fit()
regHARlog_FTSE = sm.OLS(y_train_FTSElog, X_train_FTSElog_c).fit()
regHARlog_FTSEMIB = sm.OLS(y_train_FTSEMIBlog, X_train_FTSEMIBlog_c).fit()
regHARlog_GDAXI = sm.OLS(y_train_GDAXIlog, X_train_GDAXIlog_c).fit()
regHARlog_SPX = sm.OLS(y_train_SPXlog, X_train_SPXlog_c).fit()
regHARlog_HSI = sm.OLS(y_train_HSIlog, X_train_HSIlog_c).fit()
regHARlog_IBEX = sm.OLS(y_train_IBEXlog, X_train_IBEXlog_c).fit()
regHARlog_IXIC = sm.OLS(y_train_IXIClog, X_train_IXIClog_c).fit()
regHARlog_N225 = sm.OLS(y_train_N225log, X_train_N225log_c).fit()
regHARlog_OMXC20 = sm.OLS(y_train_OMXC20log, X_train_OMXC20log_c).fit()

#%% HAR predictions

fcHAR_DJI_1 = regHAR_DJI.predict(X_test_DJI_c[21:])
fcHAR_FTSE_1 = regHAR_FTSE.predict(X_test_FTSE_c[21:])
fcHAR_FTSEMIB_1 = regHAR_FTSEMIB.predict(X_test_FTSEMIB_c[21:])
fcHAR_GDAXI_1 = regHAR_GDAXI.predict(X_test_GDAXI_c[21:])
fcHAR_SPX_1 = regHAR_SPX.predict(X_test_SPX_c[21:])
fcHAR_HSI_1 = regHAR_HSI.predict(X_test_HSI_c[21:])
fcHAR_IBEX_1 = regHAR_IBEX.predict(X_test_IBEX_c[21:])
fcHAR_IXIC_1 = regHAR_IXIC.predict(X_test_IXIC_c[21:])
fcHAR_N225_1 = regHAR_N225.predict(X_test_N225_c[21:])
fcHAR_OMXC20_1 = regHAR_OMXC20.predict(X_test_OMXC20_c[21:])

#%% Checking residuals for HAR models 


resDJI = regHAR_DJI.resid
resFTSE = regHAR_FTSE.resid
resFTSEMIB = regHAR_FTSEMIB.resid
resGDAXI = regHAR_GDAXI.resid
resSPX = regHAR_SPX.resid
resHSI = regHAR_HSI.resid
resIBEX = regHAR_IBEX.resid
resIXIC = regHAR_IXIC.resid
resN225 = regHAR_N225.resid
resOMXC20 = regHAR_OMXC20.resid

figure_1 = plt.figure(figsize = (16,8), dpi = 150)
acf(resDJI)
plt.title('Residuals of DJI')
figure_2 = plt.figure(figsize = (16,8), dpi = 150)
acf(resFTSE)
plt.title('Residuals of FTSE')
figure_3 = plt.figure(figsize = (16,8), dpi = 150)
acf(resFTSEMIB)
plt.title('Residuals of FTSEMIB')
figure_4 = plt.figure(figsize = (16,8), dpi = 150)
acf(resGDAXI)
plt.title('Residuals of GDAXI')
figure_5 = plt.figure(figsize = (16,8), dpi = 150)
acf(resSPX)
plt.title('Residuals of SPX')
plt.show()

'''
Lj_DJI = sm.stats.acorr_ljungbox(resDJI, lags=[10])
Lj_FTSE = sm.stats.acorr_ljungbox(resFTSE, lags=[10])
Lj_FTSEMIB = sm.stats.acorr_ljungbox(resFTSEMIB, lags=[10])
Lj_GDAXI = sm.stats.acorr_ljungbox(resGDAXI, lags=[10])
Lj_SPX = sm.stats.acorr_ljungbox(resSPX, lags=[10])
Lj_DJI
Lj_FTSE
Lj_FTSEMIB
Lj_GDAXI
Lj_SPX
'''

resDJI2 = resDJI**2
resFTSE2 = resFTSE**2
resFTSEMIB2 = resFTSEMIB**2
resGDAXI2 = resGDAXI**2
resSPX2 = resSPX**2
resHSI2 = resHSI**2
resIBEX2 = resIBEX**2
resIXIC2 = resIXIC**2
resN2252 = resN225**2
resOMXC202 = resOMXC20**2

figure_6 = plt.figure(figsize = (16,8), dpi = 150)
acf(resDJI2)
plt.title('Residuals squared of DJI')
figure_7 = plt.figure(figsize = (16,8), dpi = 150)
acf(resFTSE2)
plt.title('Residuals squared of FTSE')
figure_8 = plt.figure(figsize = (16,8), dpi = 150)
acf(resFTSEMIB2)
plt.title('Residuals squared of FTSEMIB')
figure_9= plt.figure(figsize = (16,8), dpi = 150)
acf(resGDAXI2)
plt.title('Residuals squared of GDAXI')
figure_10 = plt.figure(figsize = (16,8), dpi = 150)
acf(resSPX2)
plt.title('Residuals squared of SPX')
plt.show()

'''
Lj_DJI2 = sm.stats.acorr_ljungbox(resDJI, lags=[10])
Lj_FTSE2 = sm.stats.acorr_ljungbox(resFTSE, lags=[10])
Lj_FTSEMIB2 = sm.stats.acorr_ljungbox(resFTSEMIB, lags=[10])
Lj_GDAXI2 = sm.stats.acorr_ljungbox(resGDAXI, lags=[10])
Lj_SPX2 = sm.stats.acorr_ljungbox(resSPX, lags=[10])
Lj_DJI2
Lj_FTSE2
Lj_FTSEMIB2
Lj_GDAXI2
Lj_SPX2
'''

#%% Checking residuals for HARlog models

reslogDJI = regHARlog_DJI.resid
reslogFTSE = regHARlog_FTSE.resid
reslogFTSEMIB = regHARlog_FTSEMIB.resid
reslogGDAXI = regHARlog_GDAXI.resid
reslogSPX = regHARlog_SPX.resid
reslogHSI = regHARlog_HSI.resid
reslogIBEX = regHARlog_IBEX.resid
reslogIXIC = regHARlog_IXIC.resid
reslogN225 = regHARlog_N225.resid
reslogOMXC20 = regHARlog_OMXC20.resid

figure_11 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogDJI)
plt.title('Log Residuals of DJI')
figure_12 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogFTSE)
plt.title('Log Residuals of FTSE')
figure_13 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogFTSEMIB)
plt.title('Log Residuals of FTSEMIB')
figure_14 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogGDAXI)
plt.title('Log Residuals of GDAXI')
figure_15 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogSPX)
plt.title('Log Residuals of SPX')
plt.show()

'''
Lj_logDJI = sm.stats.acorr_ljungbox(reslogDJI, lags=[10])
Lj_logFTSE = sm.stats.acorr_ljungbox(reslogFTSE, lags=[10])
Lj_logFTSEMIB = sm.stats.acorr_ljungbox(reslogFTSEMIB, lags=[10])
Lj_logGDAXI = sm.stats.acorr_ljungbox(reslogGDAXI, lags=[10])
Lj_logSPX = sm.stats.acorr_ljungbox(reslogSPX, lags=[10])
Lj_logDJI
Lj_logFTSE
Lj_logFTSEMIB
Lj_logGDAXI
Lj_logSPX

'''

reslogDJI2 = reslogDJI**2
reslogFTSE2 = reslogFTSE**2
reslogFTSEMIB2 = reslogFTSEMIB**2
reslogGDAXI2 = reslogGDAXI**2
reslogSPX2 = reslogSPX**2
reslogHSI2 = reslogHSI**2
reslogIBEX2 = reslogIBEX**2
reslogIXIC2 = reslogIXIC**2
reslogN2252 = reslogN225**2
reslogOMXC20 = reslogOMXC20**2

figure_16 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogDJI2)
plt.title('Log Residuals squared of DJI')
figure_17 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogFTSE2)
plt.title('Log Residuals squared of FTSE')
figure_18 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogFTSEMIB2)
plt.title('Log Residuals squared of FTSEMIB')
figure_19= plt.figure(figsize = (16,8), dpi = 150)
acf(reslogGDAXI2)
plt.title('Log Residuals squared of GDAXI')
figure_20 = plt.figure(figsize = (16,8), dpi = 150)
acf(reslogSPX2)
plt.title('Log Residuals squared of SPX')
plt.show()

'''
Lj_logDJI2 = sm.stats.acorr_ljungbox(reslogDJI2, lags=[10])
Lj_logFTSE2 = sm.stats.acorr_ljungbox(reslogFTSE2, lags=[10])
Lj_logFTSEMIB2 = sm.stats.acorr_ljungbox(reslogFTSEMIB2, lags=[10])
Lj_logGDAXI2 = sm.stats.acorr_ljungbox(reslogGDAXI2, lags=[10])
Lj_logSPX2 = sm.stats.acorr_ljungbox(reslogSPX2, lags=[10])
Lj_logDJI2
Lj_logFTSE2
Lj_logFTSEMIB2
Lj_logGDAXI2
Lj_logSPX2
'''


#%% Adjusting the one-step ahead forecasts for HARlog models

fcHARlog_DJI_1 = regHARlog_DJI.predict(X_test_DJIlog_c[21:])
fcHARlog_FTSE_1 = regHARlog_FTSE.predict(X_test_FTSElog_c[21:])
fcHARlog_FTSEMIB_1 = regHARlog_FTSEMIB.predict(X_test_FTSEMIBlog_c[21:])
fcHARlog_GDAXI_1 = regHARlog_GDAXI.predict(X_test_GDAXIlog_c[21:])
fcHARlog_SPX_1 = regHARlog_SPX.predict(X_test_SPXlog_c[21:])
fcHARlog_HSI_1 = regHARlog_HSI.predict(X_test_HSIlog_c[21:])
fcHARlog_IBEX_1 = regHARlog_IBEX.predict(X_test_IBEXlog_c[21:])
fcHARlog_IXIC_1 = regHARlog_IXIC.predict(X_test_IXIClog_c[21:])
fcHARlog_N225_1 = regHARlog_N225.predict(X_test_N225log_c[21:])
fcHARlog_OMXC20_1 = regHARlog_OMXC20.predict(X_test_OMXC20log_c[21:])

fcHARlog_DJI_1_adj = np.exp(fcHARlog_DJI_1)*np.exp(np.var(reslogDJI)/2)
fcHARlog_FTSE_1_adj = np.exp(fcHARlog_FTSE_1)*np.exp(np.var(reslogFTSE)/2)
fcHARlog_FTSEMIB_1_adj = np.exp(fcHARlog_FTSEMIB_1)*np.exp(np.var(reslogFTSEMIB)/2)
fcHARlog_GDAXI_1_adj = np.exp(fcHARlog_GDAXI_1)*np.exp(np.var(reslogGDAXI)/2)
fcHARlog_SPX_1_adj = np.exp(fcHARlog_SPX_1)*np.exp(np.var(reslogSPX)/2)
fcHARlog_HSI_1_adj = np.exp(fcHARlog_HSI_1)*np.exp(np.var(reslogHSI)/2)
fcHARlog_IBEX_1_adj = np.exp(fcHARlog_IBEX_1)*np.exp(np.var(reslogIBEX)/2)
fcHARlog_IXIC_1_adj = np.exp(fcHARlog_IXIC_1)*np.exp(np.var(reslogIXIC)/2)
fcHARlog_N225_1_adj = np.exp(fcHARlog_N225_1)*np.exp(np.var(reslogN225)/2)
fcHARlog_OMXC20_1_adj = np.exp(fcHARlog_OMXC20_1)*np.exp(np.var(reslogOMXC20)/2)

#%% Computing the MSE for HAR and HARlog

MSE_HAR_1_DJI = mean_squared_error(y_test_DJI, fcHAR_DJI_1)
MSE_HAR_1_FTSE = mean_squared_error(y_test_FTSE, fcHAR_FTSE_1)
MSE_HAR_1_FTSEMIB = mean_squared_error(y_test_FTSEMIB, fcHAR_FTSEMIB_1)
MSE_HAR_1_GDAXI = mean_squared_error(y_test_GDAXI, fcHAR_GDAXI_1)
MSE_HAR_1_SPX = mean_squared_error(y_test_SPX, fcHAR_SPX_1)
MSE_HAR_1_HSI = mean_squared_error(y_test_HSI, fcHAR_HSI_1)
MSE_HAR_1_IBEX = mean_squared_error(y_test_IBEX, fcHAR_IBEX_1)
MSE_HAR_1_IXIC = mean_squared_error(y_test_IXIC, fcHAR_IXIC_1)
MSE_HAR_1_N225 = mean_squared_error(y_test_N225, fcHAR_N225_1)
MSE_HAR_1_OMXC20 = mean_squared_error(y_test_OMXC20, fcHAR_OMXC20_1)

MSE_HARlog_1_DJI = mean_squared_error(y_test_DJI, fcHARlog_DJI_1_adj)
MSE_HARlog_1_FTSE = mean_squared_error(y_test_FTSE, fcHARlog_FTSE_1_adj)
MSE_HARlog_1_FTSEMIB = mean_squared_error(y_test_FTSEMIB, fcHARlog_FTSEMIB_1_adj)
MSE_HARlog_1_GDAXI = mean_squared_error(y_test_GDAXI, fcHARlog_GDAXI_1_adj)
MSE_HARlog_1_SPX = mean_squared_error(y_test_SPX, fcHARlog_SPX_1_adj)
MSE_HARlog_1_HSI = mean_squared_error(y_test_HSI, fcHARlog_HSI_1_adj)
MSE_HARlog_1_IBEX = mean_squared_error(y_test_IBEX, fcHARlog_IBEX_1_adj)
MSE_HARlog_1_IXIC = mean_squared_error(y_test_IXIC, fcHARlog_IXIC_1_adj)
MSE_HARlog_1_N225 = mean_squared_error(y_test_N225, fcHARlog_N225_1_adj)
MSE_HARlog_1_OMXC20 = mean_squared_error(y_test_OMXC20, fcHARlog_OMXC20_1_adj)

#%% Compute the QLIKE loss

# AR(1) DJI 

y_forecastvalues = np.array(y_pred_AR1_1_DJI)
y_actualvalues = np.array(y_test_AR1_DJI)
qlikeDJI_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_AR1_1_Out.append(iteration)
QLIKE_AR1_1_DJI = sum(qlikeDJI_AR1_1_Out)/len(y_actualvalues)

# AR(1) FTSE

y_forecastvalues = np.array(y_pred_AR1_1_FTSE)
y_actualvalues = np.array(y_test_AR1_FTSE)
qlikeFTSE_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_AR1_1_Out.append(iteration)
QLIKE_AR1_1_FTSE = sum(qlikeFTSE_AR1_1_Out)/len(y_actualvalues)

# AR(1) FTSEMIB

y_forecastvalues = np.array(y_pred_AR1_1_FTSEMIB)
y_actualvalues = np.array(y_test_AR1_FTSEMIB)
qlikeFTSEMIB_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_AR1_1_Out.append(iteration)
QLIKE_AR1_1_FTSEMIB = sum(qlikeFTSEMIB_AR1_1_Out)/len(y_actualvalues)

# AR(1) GDAXI

y_forecastvalues = np.array(y_pred_AR1_1_GDAXI)
y_actualvalues = np.array(y_test_AR1_GDAXI)
qlikeGDAXI_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_AR1_1_Out.append(iteration)
QLIKE_AR1_1_GDAXI = sum(qlikeGDAXI_AR1_1_Out)/len(y_actualvalues)

# AR(1) SPX

y_forecastvalues = np.array(y_pred_AR1_1_SPX)
y_actualvalues = np.array(y_test_AR1_SPX)
qlikeSPX_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_AR1_1_Out.append(iteration)
QLIKE_AR1_1_SPX = sum(qlikeSPX_AR1_1_Out)/len(y_actualvalues)

# AR(1) HSI

y_forecastvalues = np.array(y_pred_AR1_1_HSI)
y_actualvalues = np.array(y_test_AR1_HSI)
qlikeHSI_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_AR1_1_Out.append(iteration)
QLIKE_AR1_1_HSI = sum(qlikeHSI_AR1_1_Out)/len(y_actualvalues)

# AR(1) IBEX

y_forecastvalues = np.array(y_pred_AR1_1_IBEX)
y_actualvalues = np.array(y_test_AR1_IBEX)
qlikeIBEX_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_AR1_1_Out.append(iteration)
QLIKE_AR1_1_IBEX = sum(qlikeIBEX_AR1_1_Out)/len(y_actualvalues)

# AR(1) IXIC

y_forecastvalues = np.array(y_pred_AR1_1_IXIC)
y_actualvalues = np.array(y_test_AR1_IXIC)
qlikeIXIC_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_AR1_1_Out.append(iteration)
QLIKE_AR1_1_IXIC = sum(qlikeIXIC_AR1_1_Out)/len(y_actualvalues)

# AR(1) N225

y_forecastvalues = np.array(y_pred_AR1_1_N225)
y_actualvalues = np.array(y_test_AR1_N225)
qlikeN225_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_AR1_1_Out.append(iteration)
QLIKE_AR1_1_N225 = sum(qlikeN225_AR1_1_Out)/len(y_actualvalues)

# AR(1) OMXC20

y_forecastvalues = np.array(y_pred_AR1_1_OMXC20)
y_actualvalues = np.array(y_test_AR1_OMXC20)
qlikeOMXC20_AR1_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_AR1_1_Out.append(iteration)
QLIKE_AR1_1_OMXC20 = sum(qlikeOMXC20_AR1_1_Out)/len(y_actualvalues)

# HAR DJI

y_forecastvalues = np.array(fcHAR_DJI_1)
y_actualvalues = np.array(y_test_DJI)
qlikeDJI_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HAR_1_Out.append(iteration)
QLIKE_HAR_1_DJI = sum(qlikeDJI_HAR_1_Out)/len(y_actualvalues)

# HAR FTSE

y_forecastvalues = np.array(fcHAR_FTSE_1)
y_actualvalues = np.array(y_test_FTSE)
qlikeFTSE_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HAR_1_Out.append(iteration)
QLIKE_HAR_1_FTSE = sum(qlikeFTSE_HAR_1_Out)/len(y_actualvalues)

# HAR FTSEMIB

y_forecastvalues = np.array(fcHAR_FTSEMIB_1)
y_actualvalues = np.array(y_test_FTSEMIB)
qlikeFTSEMIB_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HAR_1_Out.append(iteration)
QLIKE_HAR_1_FTSEMIB = sum(qlikeFTSEMIB_HAR_1_Out)/len(y_actualvalues)

# HAR GDAXI

y_forecastvalues = np.array(fcHAR_GDAXI_1)
y_actualvalues = np.array(y_test_GDAXI)
qlikeGDAXI_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HAR_1_Out.append(iteration)
QLIKE_HAR_1_GDAXI = sum(qlikeGDAXI_HAR_1_Out)/len(y_actualvalues)

# HAR SPX

y_forecastvalues = np.array(fcHAR_SPX_1)
y_actualvalues = np.array(y_test_SPX)
qlikeSPX_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HAR_1_Out.append(iteration)
QLIKE_HAR_1_SPX = sum(qlikeSPX_HAR_1_Out)/len(y_actualvalues)

# HAR HSI

y_forecastvalues = np.array(fcHAR_HSI_1)
y_actualvalues = np.array(y_test_HSI)
qlikeHSI_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HAR_1_Out.append(iteration)
QLIKE_HAR_1_HSI = sum(qlikeHSI_HAR_1_Out)/len(y_actualvalues)

# HAR IBEX

y_forecastvalues = np.array(fcHAR_IBEX_1)
y_actualvalues = np.array(y_test_IBEX)
qlikeIBEX_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HAR_1_Out.append(iteration)
QLIKE_HAR_1_IBEX = sum(qlikeIBEX_HAR_1_Out)/len(y_actualvalues)

# HAR IXIC

y_forecastvalues = np.array(fcHAR_IXIC_1)
y_actualvalues = np.array(y_test_IXIC)
qlikeIXIC_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HAR_1_Out.append(iteration)
QLIKE_HAR_1_IXIC = sum(qlikeIXIC_HAR_1_Out)/len(y_actualvalues)

# HAR N225

y_forecastvalues = np.array(fcHAR_N225_1)
y_actualvalues = np.array(y_test_N225)
qlikeN225_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HAR_1_Out.append(iteration)
QLIKE_HAR_1_N225 = sum(qlikeN225_HAR_1_Out)/len(y_actualvalues)

# HAR OMXC20

y_forecastvalues = np.array(fcHAR_OMXC20_1)
y_actualvalues = np.array(y_test_OMXC20)
qlikeOMXC20_HAR_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HAR_1_Out.append(iteration)
QLIKE_HAR_1_OMXC20 = sum(qlikeOMXC20_HAR_1_Out)/len(y_actualvalues)

# HARlog DJI

y_forecastvalues = np.array(fcHARlog_DJI_1_adj)
y_actualvalues = np.array(y_test_DJI)
qlikeDJI_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_DJI = sum(qlikeDJI_HARlog_1_Out)/len(y_actualvalues)

# HARlog FTSE

y_forecastvalues = np.array(fcHARlog_FTSE_1_adj)
y_actualvalues = np.array(y_test_FTSE)
qlikeFTSE_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_FTSE = sum(qlikeFTSE_HARlog_1_Out)/len(y_actualvalues)

# HARlog FTSEMIB

y_forecastvalues = np.array(fcHARlog_FTSEMIB_1_adj)
y_actualvalues = np.array(y_test_FTSEMIB)
qlikeFTSEMIB_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_FTSEMIB = sum(qlikeFTSEMIB_HARlog_1_Out)/len(y_actualvalues)

# HARlog GDAXI

y_forecastvalues = np.array(fcHARlog_GDAXI_1_adj)
y_actualvalues = np.array(y_test_GDAXI)
qlikeGDAXI_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_GDAXI = sum(qlikeGDAXI_HARlog_1_Out)/len(y_actualvalues)

# HARlog SPX

y_forecastvalues = np.array(fcHARlog_SPX_1_adj)
y_actualvalues = np.array(y_test_SPX)
qlikeSPX_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_SPX = sum(qlikeSPX_HARlog_1_Out)/len(y_actualvalues)

# HARlog HSI

y_forecastvalues = np.array(fcHARlog_HSI_1_adj)
y_actualvalues = np.array(y_test_HSI)
qlikeHSI_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_HSI = sum(qlikeHSI_HARlog_1_Out)/len(y_actualvalues)
    
# HARlog IBEX

y_forecastvalues = np.array(fcHARlog_IBEX_1_adj)
y_actualvalues = np.array(y_test_IBEX)
qlikeIBEX_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_IBEX = sum(qlikeIBEX_HARlog_1_Out)/len(y_actualvalues)

# HARlog IXIC

y_forecastvalues = np.array(fcHARlog_IXIC_1_adj)
y_actualvalues = np.array(y_test_IXIC)
qlikeIXIC_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_IXIC = sum(qlikeIXIC_HARlog_1_Out)/len(y_actualvalues)

# HARlog N225

y_forecastvalues = np.array(fcHARlog_N225_1_adj)
y_actualvalues = np.array(y_test_N225)
qlikeN225_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_N225 = sum(qlikeN225_HARlog_1_Out)/len(y_actualvalues)

# HARlog OMXC20

y_forecastvalues = np.array(fcHARlog_OMXC20_1_adj)
y_actualvalues = np.array(y_test_OMXC20)
qlikeOMXC20_HARlog_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HARlog_1_Out.append(iteration)
QLIKE_HARlog_1_OMXC20 = sum(qlikeOMXC20_HARlog_1_Out)/len(y_actualvalues)

#%% In-sample Loss Functions

# AR(1)

y_pred_AR1_1_DJI_In = model_AR1_DJI.predict(X_train_AR1_DJI_c[20:])
MSE_AR1_1_DJI_In = mean_squared_error(y_train_AR1_DJI.iloc[20:], y_pred_AR1_1_DJI_In)
mseDJI_AR1_In = []
for i in np.arange(len(y_train_AR1_DJI[20:])):
    mse = (y_pred_AR1_1_DJI_In.iloc[i]-y_train_AR1_DJI.iloc[i])**2
    mseDJI_AR1_In.append(mse)
mseDJI_AR1_In = np.array(mseDJI_AR1_In)

y_pred_AR1_1_FTSE_In = model_AR1_FTSE.predict(X_train_AR1_FTSE_c[20:])
MSE_AR1_1_FTSE_In = mean_squared_error(y_train_AR1_FTSE.iloc[20:], y_pred_AR1_1_FTSE_In)
mseFTSE_AR1_In = []
for i in np.arange(len(y_train_AR1_FTSE[20:])):
    mse = (y_pred_AR1_1_FTSE_In.iloc[i]-y_train_AR1_FTSE.iloc[i])**2
    mseFTSE_AR1_In.append(mse)
mseFTSE_AR1_In = np.array(mseFTSE_AR1_In)

y_pred_AR1_1_FTSEMIB_In = model_AR1_FTSEMIB.predict(X_train_AR1_FTSEMIB_c[20:])
MSE_AR1_1_FTSEMIB_In = mean_squared_error(y_train_AR1_FTSEMIB.iloc[20:], y_pred_AR1_1_FTSEMIB_In)
mseFTSEMIB_AR1_In = []
for i in np.arange(len(y_train_AR1_FTSEMIB[20:])):
    mse = (y_pred_AR1_1_FTSEMIB_In.iloc[i]-y_train_AR1_FTSEMIB.iloc[i])**2
    mseFTSEMIB_AR1_In.append(mse)
mseFTSEMIB_AR1_In = np.array(mseFTSEMIB_AR1_In)

y_pred_AR1_1_GDAXI_In = model_AR1_GDAXI.predict(X_train_AR1_GDAXI_c[20:])
MSE_AR1_1_GDAXI_In = mean_squared_error(y_train_AR1_GDAXI.iloc[20:], y_pred_AR1_1_GDAXI_In)
mseGDAXI_AR1_In = []
for i in np.arange(len(y_train_AR1_GDAXI[20:])):
    mse = (y_pred_AR1_1_GDAXI_In.iloc[i]-y_train_AR1_GDAXI.iloc[i])**2
    mseGDAXI_AR1_In.append(mse)
mseGDAXI_AR1_In = np.array(mseGDAXI_AR1_In)

y_pred_AR1_1_SPX_In = model_AR1_SPX.predict(X_train_AR1_SPX_c[20:])
MSE_AR1_1_SPX_In = mean_squared_error(y_train_AR1_SPX.iloc[20:], y_pred_AR1_1_SPX_In)
mseSPX_AR1_In = []
for i in np.arange(len(y_train_AR1_SPX[20:])):
    mse = (y_pred_AR1_1_SPX_In.iloc[i]-y_train_AR1_SPX.iloc[i])**2
    mseSPX_AR1_In.append(mse)
mseSPX_AR1_In = np.array(mseSPX_AR1_In)

y_pred_AR1_1_HSI_In = model_AR1_HSI.predict(X_train_AR1_HSI_c[20:])
MSE_AR1_1_HSI_In = mean_squared_error(y_train_AR1_HSI.iloc[20:], y_pred_AR1_1_HSI_In)
mseHSI_AR1_In = []
for i in np.arange(len(y_train_AR1_HSI[20:])):
    mse = (y_pred_AR1_1_HSI_In.iloc[i]-y_train_AR1_HSI.iloc[i])**2
    mseHSI_AR1_In.append(mse)
mseHSI_AR1_In = np.array(mseHSI_AR1_In)

y_pred_AR1_1_IBEX_In = model_AR1_IBEX.predict(X_train_AR1_IBEX_c[20:])
MSE_AR1_1_IBEX_In = mean_squared_error(y_train_AR1_IBEX.iloc[20:], y_pred_AR1_1_IBEX_In)
mseIBEX_AR1_In = []
for i in np.arange(len(y_train_AR1_IBEX[20:])):
    mse = (y_pred_AR1_1_IBEX_In.iloc[i]-y_train_AR1_IBEX.iloc[i])**2
    mseIBEX_AR1_In.append(mse)
mseIBEX_AR1_In = np.array(mseIBEX_AR1_In)

y_pred_AR1_1_IXIC_In = model_AR1_IXIC.predict(X_train_AR1_IXIC_c[20:])
MSE_AR1_1_IXIC_In = mean_squared_error(y_train_AR1_IXIC.iloc[20:], y_pred_AR1_1_IXIC_In)
mseIXIC_AR1_In = []
for i in np.arange(len(y_train_AR1_IXIC[20:])):
    mse = (y_pred_AR1_1_IXIC_In.iloc[i]-y_train_AR1_IXIC.iloc[i])**2
    mseIXIC_AR1_In.append(mse)
mseIXIC_AR1_In = np.array(mseIXIC_AR1_In)

y_pred_AR1_1_N225_In = model_AR1_N225.predict(X_train_AR1_N225_c[20:])
MSE_AR1_1_N225_In = mean_squared_error(y_train_AR1_N225.iloc[20:], y_pred_AR1_1_N225_In)
mseN225_AR1_In = []
for i in np.arange(len(y_train_AR1_N225[20:])):
    mse = (y_pred_AR1_1_N225_In.iloc[i]-y_train_AR1_N225.iloc[i])**2
    mseN225_AR1_In.append(mse)
mseN225_AR1_In = np.array(mseN225_AR1_In)

y_pred_AR1_1_OMXC20_In = model_AR1_OMXC20.predict(X_train_AR1_OMXC20_c[20:])
MSE_AR1_1_OMXC20_In = mean_squared_error(y_train_AR1_OMXC20.iloc[20:], y_pred_AR1_1_OMXC20_In)
mseOMXC20_AR1_In = []
for i in np.arange(len(y_train_AR1_OMXC20[20:])):
    mse = (y_pred_AR1_1_OMXC20_In.iloc[i]-y_train_AR1_OMXC20.iloc[i])**2
    mseOMXC20_AR1_In.append(mse)
mseOMXC20_AR1_In = np.array(mseOMXC20_AR1_In)

# HAR

fcHAR_DJI_1_In = regHAR_DJI.predict(X_train_DJI_c)
fcHAR_FTSE_1_In = regHAR_FTSE.predict(X_train_FTSE_c)
fcHAR_FTSEMIB_1_In = regHAR_FTSEMIB.predict(X_train_FTSEMIB_c)
fcHAR_GDAXI_1_In = regHAR_GDAXI.predict(X_train_GDAXI_c)
fcHAR_SPX_1_In = regHAR_SPX.predict(X_train_SPX_c)
fcHAR_HSI_1_In = regHAR_HSI.predict(X_train_HSI_c)
fcHAR_IBEX_1_In = regHAR_IBEX.predict(X_train_IBEX_c)
fcHAR_IXIC_1_In = regHAR_IXIC.predict(X_train_IXIC_c)
fcHAR_N225_1_In = regHAR_N225.predict(X_train_N225_c)
fcHAR_OMXC20_1_In = regHAR_OMXC20.predict(X_train_OMXC20_c)

MSE_HAR_1_DJI_In = mean_squared_error(y_train_DJI, fcHAR_DJI_1_In)
mseDJI_HAR_In = []
for i in np.arange(len(y_train_DJI)):
    mse = (fcHAR_DJI_1_In.iloc[i]-y_train_DJI.iloc[i])**2
    mseDJI_HAR_In.append(mse)
mseDJI_HAR_In = np.array(mseDJI_HAR_In)

MSE_HAR_1_FTSE_In = mean_squared_error(y_train_FTSE, fcHAR_FTSE_1_In)
mseFTSE_HAR_In = []
for i in np.arange(len(y_train_FTSE)):
    mse = (fcHAR_FTSE_1_In.iloc[i]-y_train_FTSE.iloc[i])**2
    mseFTSE_HAR_In.append(mse)
mseFTSE_HAR_In = np.array(mseFTSE_HAR_In)

MSE_HAR_1_FTSEMIB_In = mean_squared_error(y_train_FTSEMIB, fcHAR_FTSEMIB_1_In)
mseFTSEMIB_HAR_In = []
for i in np.arange(len(y_train_FTSEMIB)):
    mse = (fcHAR_FTSEMIB_1_In.iloc[i]-y_train_FTSEMIB.iloc[i])**2
    mseFTSEMIB_HAR_In.append(mse)
mseFTSEMIB_HAR_In = np.array(mseFTSEMIB_HAR_In)

MSE_HAR_1_GDAXI_In = mean_squared_error(y_train_GDAXI, fcHAR_GDAXI_1_In)
mseGDAXI_HAR_In = []
for i in np.arange(len(y_train_GDAXI)):
    mse = (fcHAR_GDAXI_1_In.iloc[i]-y_train_GDAXI.iloc[i])**2
    mseGDAXI_HAR_In.append(mse)
mseGDAXI_HAR_In = np.array(mseGDAXI_HAR_In)

MSE_HAR_1_SPX_In = mean_squared_error(y_train_SPX, fcHAR_SPX_1_In)
mseSPX_HAR_In = []
for i in np.arange(len(y_train_SPX)):
    mse = (fcHAR_SPX_1_In.iloc[i]-y_train_SPX.iloc[i])**2
    mseSPX_HAR_In.append(mse)
mseSPX_HAR_In = np.array(mseSPX_HAR_In)

MSE_HAR_1_HSI_In = mean_squared_error(y_train_HSI, fcHAR_HSI_1_In)
mseHSI_HAR_In = []
for i in np.arange(len(y_train_HSI)):
    mse = (fcHAR_HSI_1_In.iloc[i]-y_train_HSI.iloc[i])**2
    mseHSI_HAR_In.append(mse)
mseHSI_HAR_In = np.array(mseHSI_HAR_In)

MSE_HAR_1_IBEX_In = mean_squared_error(y_train_IBEX, fcHAR_IBEX_1_In)
mseIBEX_HAR_In = []
for i in np.arange(len(y_train_IBEX)):
    mse = (fcHAR_IBEX_1_In.iloc[i]-y_train_IBEX.iloc[i])**2
    mseIBEX_HAR_In.append(mse)
mseIBEX_HAR_In = np.array(mseIBEX_HAR_In)

MSE_HAR_1_IXIC_In = mean_squared_error(y_train_IXIC, fcHAR_IXIC_1_In)
mseIXIC_HAR_In = []
for i in np.arange(len(y_train_IXIC)):
    mse = (fcHAR_IXIC_1_In.iloc[i]-y_train_IXIC.iloc[i])**2
    mseIXIC_HAR_In.append(mse)
mseIXIC_HAR_In = np.array(mseIXIC_HAR_In)

MSE_HAR_1_N225_In = mean_squared_error(y_train_N225, fcHAR_N225_1_In)
mseN225_HAR_In = []
for i in np.arange(len(y_train_N225)):
    mse = (fcHAR_N225_1_In.iloc[i]-y_train_N225.iloc[i])**2
    mseN225_HAR_In.append(mse)
mseN225_HAR_In = np.array(mseN225_HAR_In)

MSE_HAR_1_OMXC20_In = mean_squared_error(y_train_OMXC20, fcHAR_OMXC20_1_In)
mseOMXC20_HAR_In = []
for i in np.arange(len(y_train_OMXC20)):
    mse = (fcHAR_OMXC20_1_In.iloc[i]-y_train_OMXC20.iloc[i])**2
    mseOMXC20_HAR_In.append(mse)
mseOMXC20_HAR_In = np.array(mseOMXC20_HAR_In)

# HARlog

fcHARlog_DJI_1_In = regHARlog_DJI.predict(X_train_DJIlog_c)
fcHARlog_FTSE_1_In = regHARlog_FTSE.predict(X_train_FTSElog_c)
fcHARlog_FTSEMIB_1_In = regHARlog_FTSEMIB.predict(X_train_FTSEMIBlog_c)
fcHARlog_GDAXI_1_In = regHARlog_GDAXI.predict(X_train_GDAXIlog_c)
fcHARlog_SPX_1_In = regHARlog_SPX.predict(X_train_SPXlog_c)
fcHARlog_HSI_1_In = regHARlog_HSI.predict(X_train_HSIlog_c)
fcHARlog_IBEX_1_In = regHARlog_IBEX.predict(X_train_IBEXlog_c)
fcHARlog_IXIC_1_In = regHARlog_IXIC.predict(X_train_IXIClog_c)
fcHARlog_N225_1_In = regHARlog_N225.predict(X_train_N225log_c)
fcHARlog_OMXC20_1_In = regHARlog_OMXC20.predict(X_train_OMXC20log_c)

fcHARlog_DJI_1_adj_In = np.exp(fcHARlog_DJI_1_In)*np.exp(np.var(reslogDJI)/2)
fcHARlog_FTSE_1_adj_In = np.exp(fcHARlog_FTSE_1_In)*np.exp(np.var(reslogFTSE)/2)
fcHARlog_FTSEMIB_1_adj_In = np.exp(fcHARlog_FTSEMIB_1_In)*np.exp(np.var(reslogFTSEMIB)/2)
fcHARlog_GDAXI_1_adj_In = np.exp(fcHARlog_GDAXI_1_In)*np.exp(np.var(reslogGDAXI)/2)
fcHARlog_SPX_1_adj_In = np.exp(fcHARlog_SPX_1_In)*np.exp(np.var(reslogSPX)/2)
fcHARlog_HSI_1_adj_In = np.exp(fcHARlog_HSI_1_In)*np.exp(np.var(reslogHSI)/2)
fcHARlog_IBEX_1_adj_In = np.exp(fcHARlog_IBEX_1_In)*np.exp(np.var(reslogIBEX)/2)
fcHARlog_IXIC_1_adj_In = np.exp(fcHARlog_IXIC_1_In)*np.exp(np.var(reslogIXIC)/2)
fcHARlog_N225_1_adj_In = np.exp(fcHARlog_N225_1_In)*np.exp(np.var(reslogN225)/2)
fcHARlog_OMXC20_1_adj_In = np.exp(fcHARlog_OMXC20_1_In)*np.exp(np.var(reslogOMXC20)/2)

MSE_HARlog_1_DJI_In = mean_squared_error(y_train_DJI, fcHARlog_DJI_1_adj_In)
mseDJI_HARlog_In = []
for i in np.arange(len(y_train_DJI)):
    mse = (fcHARlog_DJI_1_adj_In.iloc[i]-y_train_DJI.iloc[i])**2
    mseDJI_HARlog_In.append(mse)
mseDJI_HARlog_In = np.array(mseDJI_HARlog_In)

MSE_HARlog_1_FTSE_In = mean_squared_error(y_train_FTSE, fcHARlog_FTSE_1_adj_In)
mseFTSE_HARlog_In = []
for i in np.arange(len(y_train_FTSE)):
    mse = (fcHARlog_FTSE_1_adj_In.iloc[i]-y_train_FTSE.iloc[i])**2
    mseFTSE_HARlog_In.append(mse)
mseFTSE_HARlog_In = np.array(mseFTSE_HARlog_In)

MSE_HARlog_1_FTSEMIB_In = mean_squared_error(y_train_FTSEMIB, fcHARlog_FTSEMIB_1_adj_In)
mseFTSEMIB_HARlog_In = []
for i in np.arange(len(y_train_FTSEMIB)):
    mse = (fcHARlog_FTSEMIB_1_adj_In.iloc[i]-y_train_FTSEMIB.iloc[i])**2
    mseFTSEMIB_HARlog_In.append(mse)
mseFTSEMIB_HARlog_In = np.array(mseFTSEMIB_HARlog_In)

MSE_HARlog_1_GDAXI_In = mean_squared_error(y_train_GDAXI, fcHARlog_GDAXI_1_adj_In)
mseGDAXI_HARlog_In = []
for i in np.arange(len(y_train_GDAXI)):
    mse = (fcHARlog_GDAXI_1_adj_In.iloc[i]-y_train_GDAXI.iloc[i])**2
    mseGDAXI_HARlog_In.append(mse)
mseGDAXI_HARlog_In = np.array(mseGDAXI_HARlog_In)

MSE_HARlog_1_SPX_In = mean_squared_error(y_train_SPX, fcHARlog_SPX_1_adj_In)
mseSPX_HARlog_In = []
for i in np.arange(len(y_train_SPX)):
    mse = (fcHARlog_SPX_1_adj_In.iloc[i]-y_train_SPX.iloc[i])**2
    mseSPX_HARlog_In.append(mse)
mseSPX_HARlog_In = np.array(mseSPX_HARlog_In)

MSE_HARlog_1_HSI_In = mean_squared_error(y_train_HSI, fcHARlog_HSI_1_adj_In)
mseHSI_HARlog_In = []
for i in np.arange(len(y_train_HSI)):
    mse = (fcHARlog_HSI_1_adj_In.iloc[i]-y_train_HSI.iloc[i])**2
    mseHSI_HARlog_In.append(mse)
mseHSI_HARlog_In = np.array(mseHSI_HARlog_In)

MSE_HARlog_1_IBEX_In = mean_squared_error(y_train_IBEX, fcHARlog_IBEX_1_adj_In)
mseIBEX_HARlog_In = []
for i in np.arange(len(y_train_IBEX)):
    mse = (fcHARlog_IBEX_1_adj_In.iloc[i]-y_train_IBEX.iloc[i])**2
    mseIBEX_HARlog_In.append(mse)
mseIBEX_HARlog_In = np.array(mseIBEX_HARlog_In)

MSE_HARlog_1_IXIC_In = mean_squared_error(y_train_IXIC, fcHARlog_IXIC_1_adj_In)
mseIXIC_HARlog_In = []
for i in np.arange(len(y_train_IXIC)):
    mse = (fcHARlog_IXIC_1_adj_In.iloc[i]-y_train_IXIC.iloc[i])**2
    mseIXIC_HARlog_In.append(mse)
mseIXIC_HARlog_In = np.array(mseIXIC_HARlog_In)

MSE_HARlog_1_N225_In = mean_squared_error(y_train_N225, fcHARlog_N225_1_adj_In)
mseN225_HARlog_In = []
for i in np.arange(len(y_train_N225)):
    mse = (fcHARlog_N225_1_adj_In.iloc[i]-y_train_N225.iloc[i])**2
    mseN225_HARlog_In.append(mse)
mseN225_HARlog_In = np.array(mseN225_HARlog_In)

MSE_HARlog_1_OMXC20_In = mean_squared_error(y_train_OMXC20, fcHARlog_OMXC20_1_adj_In)
mseOMXC20_HARlog_In = []
for i in np.arange(len(y_train_OMXC20)):
    mse = (fcHARlog_OMXC20_1_adj_In.iloc[i]-y_train_OMXC20.iloc[i])**2
    mseOMXC20_HARlog_In.append(mse)
mseOMXC20_HARlog_In = np.array(mseOMXC20_HARlog_In)

#%% Out-of-Sample Loss function

# AR1

mseDJI_AR1_Out = []
for i in np.arange(len(y_test_AR1_DJI)):
    mse = (y_pred_AR1_1_DJI.iloc[i]-y_test_AR1_DJI.iloc[i])**2
    mseDJI_AR1_Out.append(mse)
mseDJI_AR1_Out = np.array(mseDJI_AR1_Out)

mseFTSE_AR1_Out = []
for i in np.arange(len(y_test_AR1_FTSE)):
    mse = (y_pred_AR1_1_FTSE.iloc[i]-y_test_AR1_FTSE.iloc[i])**2
    mseFTSE_AR1_Out.append(mse)
mseFTSE_AR1_Out = np.array(mseFTSE_AR1_Out)

mseFTSEMIB_AR1_Out = []
for i in np.arange(len(y_test_AR1_FTSEMIB)):
    mse = (y_pred_AR1_1_FTSEMIB.iloc[i]-y_test_AR1_FTSEMIB.iloc[i])**2
    mseFTSEMIB_AR1_Out.append(mse)
mseFTSEMIB_AR1_Out = np.array(mseFTSEMIB_AR1_Out)

mseGDAXI_AR1_Out = []
for i in np.arange(len(y_test_AR1_GDAXI)):
    mse = (y_pred_AR1_1_GDAXI.iloc[i]-y_test_AR1_GDAXI.iloc[i])**2
    mseGDAXI_AR1_Out.append(mse)
mseGDAXI_AR1_Out = np.array(mseGDAXI_AR1_Out)

mseSPX_AR1_Out = []
for i in np.arange(len(y_test_AR1_SPX)):
    mse = (y_pred_AR1_1_SPX.iloc[i]-y_test_AR1_SPX.iloc[i])**2
    mseSPX_AR1_Out.append(mse)
mseSPX_AR1_Out = np.array(mseSPX_AR1_Out)

mseHSI_AR1_Out = []
for i in np.arange(len(y_test_AR1_HSI)):
    mse = (y_pred_AR1_1_HSI.iloc[i]-y_test_AR1_HSI.iloc[i])**2
    mseHSI_AR1_Out.append(mse)
mseHSI_AR1_Out = np.array(mseHSI_AR1_Out)

mseIBEX_AR1_Out = []
for i in np.arange(len(y_test_AR1_IBEX)):
    mse = (y_pred_AR1_1_IBEX.iloc[i]-y_test_AR1_IBEX.iloc[i])**2
    mseIBEX_AR1_Out.append(mse)
mseIBEX_AR1_Out = np.array(mseIBEX_AR1_Out)

mseIXIC_AR1_Out = []
for i in np.arange(len(y_test_AR1_IXIC)):
    mse = (y_pred_AR1_1_IXIC.iloc[i]-y_test_AR1_IXIC.iloc[i])**2
    mseIXIC_AR1_Out.append(mse)
mseIXIC_AR1_Out = np.array(mseIXIC_AR1_Out)

mseN225_AR1_Out = []
for i in np.arange(len(y_test_AR1_N225)):
    mse = (y_pred_AR1_1_N225.iloc[i]-y_test_AR1_N225.iloc[i])**2
    mseN225_AR1_Out.append(mse)
mseN225_AR1_Out = np.array(mseN225_AR1_Out)

mseOMXC20_AR1_Out = []
for i in np.arange(len(y_test_AR1_OMXC20)):
    mse = (y_pred_AR1_1_OMXC20.iloc[i]-y_test_AR1_OMXC20.iloc[i])**2
    mseOMXC20_AR1_Out.append(mse)
mseOMXC20_AR1_Out = np.array(mseOMXC20_AR1_Out)

# HAR

mseDJI_HAR_Out = []
for i in np.arange(len(y_test_DJI)):
    mse = (fcHAR_DJI_1.iloc[i]-y_test_DJI.iloc[i])**2
    mseDJI_HAR_Out.append(mse)
mseDJI_HAR_Out = np.array(mseDJI_HAR_Out)

mseFTSE_HAR_Out = []
for i in np.arange(len(y_test_FTSE)):
    mse = (fcHAR_FTSE_1.iloc[i]-y_test_FTSE.iloc[i])**2
    mseFTSE_HAR_Out.append(mse)
mseFTSE_HAR_Out = np.array(mseFTSE_HAR_Out)

mseFTSEMIB_HAR_Out = []
for i in np.arange(len(y_test_FTSEMIB)):
    mse = (fcHAR_FTSEMIB_1.iloc[i]-y_test_FTSEMIB.iloc[i])**2
    mseFTSEMIB_HAR_Out.append(mse)
mseFTSEMIB_HAR_Out = np.array(mseFTSEMIB_HAR_Out)

mseGDAXI_HAR_Out = []
for i in np.arange(len(y_test_GDAXI)):
    mse = (fcHAR_GDAXI_1.iloc[i]-y_test_GDAXI.iloc[i])**2
    mseGDAXI_HAR_Out.append(mse)
mseGDAXI_HAR_Out = np.array(mseGDAXI_HAR_Out)

mseSPX_HAR_Out = []
for i in np.arange(len(y_test_SPX)):
    mse = (fcHAR_SPX_1.iloc[i]-y_test_SPX.iloc[i])**2
    mseSPX_HAR_Out.append(mse)
mseSPX_HAR_Out = np.array(mseSPX_HAR_Out)

mseHSI_HAR_Out = []
for i in np.arange(len(y_test_HSI)):
    mse = (fcHAR_HSI_1.iloc[i]-y_test_HSI.iloc[i])**2
    mseHSI_HAR_Out.append(mse)
mseHSI_HAR_Out = np.array(mseHSI_HAR_Out)

mseIBEX_HAR_Out = []
for i in np.arange(len(y_test_IBEX)):
    mse = (fcHAR_IBEX_1.iloc[i]-y_test_IBEX.iloc[i])**2
    mseIBEX_HAR_Out.append(mse)
mseIBEX_HAR_Out = np.array(mseIBEX_HAR_Out)

mseIXIC_HAR_Out = []
for i in np.arange(len(y_test_IXIC)):
    mse = (fcHAR_IXIC_1.iloc[i]-y_test_IXIC.iloc[i])**2
    mseIXIC_HAR_Out.append(mse)
mseIXIC_HAR_Out = np.array(mseIXIC_HAR_Out)

mseN225_HAR_Out = []
for i in np.arange(len(y_test_N225)):
    mse = (fcHAR_N225_1.iloc[i]-y_test_N225.iloc[i])**2
    mseN225_HAR_Out.append(mse)
mseN225_HAR_Out = np.array(mseN225_HAR_Out)

mseOMXC20_HAR_Out = []
for i in np.arange(len(y_test_OMXC20)):
    mse = (fcHAR_OMXC20_1.iloc[i]-y_test_OMXC20.iloc[i])**2
    mseOMXC20_HAR_Out.append(mse)
mseOMXC20_HAR_Out = np.array(mseOMXC20_HAR_Out)

# HARlog

mseDJI_HARlog_Out = []
for i in np.arange(len(y_test_DJI)):
    mse = (fcHARlog_DJI_1_adj.iloc[i]-y_test_DJI.iloc[i])**2
    mseDJI_HARlog_Out.append(mse)
mseDJI_HARlog_Out = np.array(mseDJI_HARlog_Out)

mseFTSE_HARlog_Out = []
for i in np.arange(len(y_test_FTSE)):
    mse = (fcHARlog_FTSE_1_adj.iloc[i]-y_test_FTSE.iloc[i])**2
    mseFTSE_HARlog_Out.append(mse)
mseFTSE_HARlog_Out = np.array(mseFTSE_HARlog_Out)

mseFTSEMIB_HARlog_Out = []
for i in np.arange(len(y_test_FTSEMIB)):
    mse = (fcHARlog_FTSEMIB_1_adj.iloc[i]-y_test_FTSEMIB.iloc[i])**2
    mseFTSEMIB_HARlog_Out.append(mse)
mseFTSEMIB_HARlog_Out = np.array(mseFTSEMIB_HARlog_Out)

mseGDAXI_HARlog_Out = []
for i in np.arange(len(y_test_GDAXI)):
    mse = (fcHARlog_GDAXI_1_adj.iloc[i]-y_test_GDAXI.iloc[i])**2
    mseGDAXI_HARlog_Out.append(mse)
mseGDAXI_HARlog_Out = np.array(mseGDAXI_HARlog_Out)

mseSPX_HARlog_Out = []
for i in np.arange(len(y_test_SPX)):
    mse = (fcHARlog_SPX_1_adj.iloc[i]-y_test_SPX.iloc[i])**2
    mseSPX_HARlog_Out.append(mse)
mseSPX_HARlog_Out = np.array(mseSPX_HARlog_Out)

mseHSI_HARlog_Out = []
for i in np.arange(len(y_test_HSI)):
    mse = (fcHARlog_HSI_1_adj.iloc[i]-y_test_HSI.iloc[i])**2
    mseHSI_HARlog_Out.append(mse)
mseHSI_HARlog_Out = np.array(mseHSI_HARlog_Out)

mseIBEX_HARlog_Out = []
for i in np.arange(len(y_test_IBEX)):
    mse = (fcHARlog_IBEX_1_adj.iloc[i]-y_test_IBEX.iloc[i])**2
    mseIBEX_HARlog_Out.append(mse)
mseIBEX_HARlog_Out = np.array(mseIBEX_HARlog_Out)

mseIXIC_HARlog_Out = []
for i in np.arange(len(y_test_IXIC)):
    mse = (fcHARlog_IXIC_1_adj.iloc[i]-y_test_IXIC.iloc[i])**2
    mseIXIC_HARlog_Out.append(mse)
mseIXIC_HARlog_Out = np.array(mseIXIC_HARlog_Out)

mseN225_HARlog_Out = []
for i in np.arange(len(y_test_N225)):
    mse = (fcHARlog_N225_1_adj.iloc[i]-y_test_N225.iloc[i])**2
    mseN225_HARlog_Out.append(mse)
mseN225_HARlog_Out = np.array(mseN225_HARlog_Out)

mseOMXC20_HARlog_Out = []
for i in np.arange(len(y_test_OMXC20)):
    mse = (fcHARlog_OMXC20_1_adj.iloc[i]-y_test_OMXC20.iloc[i])**2
    mseOMXC20_HARlog_Out.append(mse)
mseOMXC20_HARlog_Out = np.array(mseOMXC20_HARlog_Out)

#%% File for time series In-sample MSE

indices = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']

arrays_In = {
    'mse_AR1': {
        'DJI': mseDJI_AR1_In,
        'FTSE': mseFTSE_AR1_In,
        'FTSEMIB': mseFTSEMIB_AR1_In,
        'GDAXI': mseGDAXI_AR1_In,
        'SPX': mseSPX_AR1_In,
        'HSI': mseHSI_AR1_In,
        'IBEX': mseIBEX_AR1_In,
        'IXIC': mseIXIC_AR1_In,
        'N225': mseN225_AR1_In,
        'OMXC20': mseOMXC20_AR1_In
    },
    'mse_HAR': {
        'DJI': mseDJI_HAR_In,
        'FTSE': mseFTSE_HAR_In,
        'FTSEMIB': mseFTSEMIB_HAR_In,
        'GDAXI': mseGDAXI_HAR_In,
        'SPX': mseSPX_HAR_In,
        'HSI': mseHSI_HAR_In,
        'IBEX': mseIBEX_HAR_In,
        'IXIC': mseIXIC_HAR_In,
        'N225': mseN225_HAR_In,
        'OMXC20': mseOMXC20_HAR_In
    },
    'mse_HARlog': {
        'DJI': mseDJI_HARlog_In,
        'FTSE': mseFTSE_HARlog_In,
        'FTSEMIB': mseFTSEMIB_HARlog_In,
        'GDAXI': mseGDAXI_HARlog_In,
        'SPX': mseSPX_HARlog_In,
        'HSI': mseHSI_HARlog_In,
        'IBEX': mseIBEX_HARlog_In,
        'IXIC': mseIXIC_HARlog_In,
        'N225': mseN225_HARlog_In,
        'OMXC20': mseOMXC20_HARlog_In
                    }
    }


for k1 in arrays_In:
    if k1 == 'mse_AR1':    
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_AR1_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
    elif k1 == 'mse_HAR':
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_HAR_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
    elif k1 == 'mse_HARlog':
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_HARlog_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
            
#%% File for time series Out-of-sample MSE

arrays_Out = {
    'mse_AR1': {
        'DJI': mseDJI_AR1_Out,
        'FTSE': mseFTSE_AR1_Out,
        'FTSEMIB': mseFTSEMIB_AR1_Out,
        'GDAXI': mseGDAXI_AR1_Out,
        'SPX': mseSPX_AR1_Out,
        'HSI': mseHSI_AR1_Out,
        'IBEX': mseIBEX_AR1_Out,
        'IXIC': mseIXIC_AR1_Out,
        'N225': mseN225_AR1_Out,
        'OMXC20': mseOMXC20_AR1_Out
    },
    'mse_HAR': {
        'DJI': mseDJI_HAR_Out,
        'FTSE': mseFTSE_HAR_Out,
        'FTSEMIB': mseFTSEMIB_HAR_Out,
        'GDAXI': mseGDAXI_HAR_Out,
        'SPX': mseSPX_HAR_Out,
        'HSI': mseHSI_HAR_Out,
        'IBEX': mseIBEX_HAR_Out,
        'IXIC': mseIXIC_HAR_Out,
        'N225': mseN225_HAR_Out,
        'OMXC20': mseOMXC20_HAR_Out
    },
    'mse_HARlog': {
        'DJI': mseDJI_HARlog_Out,
        'FTSE': mseFTSE_HARlog_Out,
        'FTSEMIB': mseFTSEMIB_HARlog_Out,
        'GDAXI': mseGDAXI_HARlog_Out,
        'SPX': mseSPX_HARlog_Out,
        'HSI': mseHSI_HARlog_Out,
        'IBEX': mseIBEX_HARlog_Out,
        'IXIC': mseIXIC_HARlog_Out,
        'N225': mseN225_HARlog_Out,
        'OMXC20': mseOMXC20_HARlog_Out
                    }
    }


for k1 in arrays_Out:
    if k1 == 'mse_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_AR1_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HAR_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'mse_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_HARlog_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%% QLIKE Out

arrays_Out = {
    'qlike_AR1': {
        'DJI': qlikeDJI_AR1_1_Out,
        'FTSE': qlikeFTSE_AR1_1_Out,
        'FTSEMIB': qlikeFTSEMIB_AR1_1_Out,
        'GDAXI': qlikeGDAXI_AR1_1_Out,
        'SPX': qlikeSPX_AR1_1_Out,
        'HSI': qlikeHSI_AR1_1_Out,
        'IBEX': qlikeIBEX_AR1_1_Out,
        'IXIC': qlikeIXIC_AR1_1_Out,
        'N225': qlikeN225_AR1_1_Out,
        'OMXC20': qlikeOMXC20_AR1_1_Out
    },
    'qlike_HAR': {
        'DJI': qlikeDJI_HAR_1_Out,
        'FTSE': qlikeFTSE_HAR_1_Out,
        'FTSEMIB': qlikeFTSEMIB_HAR_1_Out,
        'GDAXI': qlikeGDAXI_HAR_1_Out,
        'SPX': qlikeSPX_HAR_1_Out,
        'HSI': qlikeHSI_HAR_1_Out,
        'IBEX': qlikeIBEX_HAR_1_Out,
        'IXIC': qlikeIXIC_HAR_1_Out,
        'N225': qlikeN225_HAR_1_Out,
        'OMXC20': qlikeOMXC20_HAR_1_Out
    },
    'qlike_HARlog': {
        'DJI': qlikeDJI_HARlog_1_Out, 
        'FTSE': qlikeFTSE_HARlog_1_Out,
        'FTSEMIB': qlikeFTSEMIB_HARlog_1_Out,
        'GDAXI': qlikeGDAXI_HARlog_1_Out,
        'SPX': qlikeSPX_HARlog_1_Out,
        'HSI': qlikeHSI_HARlog_1_Out,
        'IBEX': qlikeIBEX_HARlog_1_Out,
        'IXIC': qlikeIXIC_HARlog_1_Out,
        'N225': qlikeN225_HARlog_1_Out,
        'OMXC20': qlikeOMXC20_HARlog_1_Out
                    }
    }


for k1 in arrays_Out:
    if k1 == 'qlike_AR1':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_AR1_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HAR':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HAR_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
    elif k1 == 'qlike_HARlog':
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_HARlog_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%% Qlike In-Sample

# AR(1) DJI 

y_forecastvalues = np.array(y_pred_AR1_1_DJI_In)
y_actualvalues = np.array(y_train_AR1_DJI[20:])
qlikeDJI_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_AR1_1_In.append(iteration)
QLIKE_AR1_1_DJI_In = sum(qlikeDJI_AR1_1_In)/len(y_actualvalues)

# AR(1) FTSE

y_forecastvalues = np.array(y_pred_AR1_1_FTSE_In)
y_actualvalues = np.array(y_train_AR1_FTSE[20:])
qlikeFTSE_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_AR1_1_In.append(iteration)
QLIKE_AR1_1_FTSE_In = sum(qlikeFTSE_AR1_1_In)/len(y_actualvalues)

# AR(1) FTSEMIB

y_forecastvalues = np.array(y_pred_AR1_1_FTSEMIB_In)
y_actualvalues = np.array(y_train_AR1_FTSEMIB[20:])
qlikeFTSEMIB_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_AR1_1_In.append(iteration)
QLIKE_AR1_1_FTSEMIB_In = sum(qlikeFTSEMIB_AR1_1_In)/len(y_actualvalues)

# AR(1) GDAXI

y_forecastvalues = np.array(y_pred_AR1_1_GDAXI_In)
y_actualvalues = np.array(y_train_AR1_GDAXI[20:])
qlikeGDAXI_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_AR1_1_In.append(iteration)
QLIKE_AR1_1_GDAXI_In = sum(qlikeGDAXI_AR1_1_In)/len(y_actualvalues)

# AR(1) SPX

y_forecastvalues = np.array(y_pred_AR1_1_SPX_In)
y_actualvalues = np.array(y_train_AR1_SPX[20:])
qlikeSPX_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_AR1_1_In.append(iteration)
QLIKE_AR1_1_SPX_In = sum(qlikeSPX_AR1_1_In)/len(y_actualvalues)

# AR(1) HSI

y_forecastvalues = np.array(y_pred_AR1_1_HSI_In)
y_actualvalues = np.array(y_train_AR1_HSI[20:])
qlikeHSI_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_AR1_1_In.append(iteration)
QLIKE_AR1_1_HSI_In = sum(qlikeHSI_AR1_1_In)/len(y_actualvalues)

# AR(1) IBEX

y_forecastvalues = np.array(y_pred_AR1_1_IBEX_In)
y_actualvalues = np.array(y_train_AR1_IBEX[20:])
qlikeIBEX_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_AR1_1_In.append(iteration)
QLIKE_AR1_1_IBEX_In = sum(qlikeIBEX_AR1_1_In)/len(y_actualvalues)

# AR(1) IXIC

y_forecastvalues = np.array(y_pred_AR1_1_IXIC_In)
y_actualvalues = np.array(y_train_AR1_IXIC[20:])
qlikeIXIC_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_AR1_1_In.append(iteration)
QLIKE_AR1_1_IXIC_In = sum(qlikeIXIC_AR1_1_In)/len(y_actualvalues)

# AR(1) N225

y_forecastvalues = np.array(y_pred_AR1_1_N225_In)
y_actualvalues = np.array(y_train_AR1_N225[20:])
qlikeN225_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_AR1_1_In.append(iteration)
QLIKE_AR1_1_N225_In = sum(qlikeN225_AR1_1_In)/len(y_actualvalues)

# AR(1) OMXC20

y_forecastvalues = np.array(y_pred_AR1_1_OMXC20_In)
y_actualvalues = np.array(y_train_AR1_OMXC20[20:])
qlikeOMXC20_AR1_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_AR1_1_In.append(iteration)
QLIKE_AR1_1_OMXC20_In = sum(qlikeOMXC20_AR1_1_In)/len(y_actualvalues)

# HAR DJI

y_forecastvalues = np.array(fcHAR_DJI_1_In)
y_actualvalues = np.array(y_train_DJI)
qlikeDJI_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HAR_1_In.append(iteration)
QLIKE_HAR_1_DJI_In = sum(qlikeDJI_HAR_1_In)/len(y_actualvalues)

# HAR FTSE

y_forecastvalues = np.array(fcHAR_FTSE_1_In)
y_actualvalues = np.array(y_train_FTSE)
qlikeFTSE_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HAR_1_In.append(iteration)
QLIKE_HAR_1_FTSE_In = sum(qlikeFTSE_HAR_1_In)/len(y_actualvalues)

# HAR FTSEMIB

y_forecastvalues = np.array(fcHAR_FTSEMIB_1_In)
y_actualvalues = np.array(y_train_FTSEMIB)
qlikeFTSEMIB_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HAR_1_In.append(iteration)
QLIKE_HAR_1_FTSEMIB_In = sum(qlikeFTSEMIB_HAR_1_In)/len(y_actualvalues)

# HAR GDAXI

y_forecastvalues = np.array(fcHAR_GDAXI_1_In)
y_actualvalues = np.array(y_train_GDAXI)
qlikeGDAXI_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HAR_1_In.append(iteration)
QLIKE_HAR_1_GDAXI_In = sum(qlikeGDAXI_HAR_1_In)/len(y_actualvalues)

# HAR SPX

y_forecastvalues = np.array(fcHAR_SPX_1_In)
y_actualvalues = np.array(y_train_SPX)
qlikeSPX_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HAR_1_In.append(iteration)
QLIKE_HAR_1_SPX_In = sum(qlikeSPX_HAR_1_In)/len(y_actualvalues)

# HAR HSI

y_forecastvalues = np.array(fcHAR_HSI_1_In)
y_actualvalues = np.array(y_train_HSI)
qlikeHSI_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HAR_1_In.append(iteration)
QLIKE_HAR_1_HSI_In = sum(qlikeHSI_HAR_1_In)/len(y_actualvalues)

# HAR IBEX

y_forecastvalues = np.array(fcHAR_IBEX_1_In)
y_actualvalues = np.array(y_train_IBEX)
qlikeIBEX_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HAR_1_In.append(iteration)
QLIKE_HAR_1_IBEX_In = sum(qlikeIBEX_HAR_1_In)/len(y_actualvalues)

# HAR IXIC

y_forecastvalues = np.array(fcHAR_IXIC_1_In)
y_actualvalues = np.array(y_train_IXIC)
qlikeIXIC_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HAR_1_In.append(iteration)
QLIKE_HAR_1_IXIC_In = sum(qlikeIXIC_HAR_1_In)/len(y_actualvalues)

# HAR N225

y_forecastvalues = np.array(fcHAR_N225_1_In)
y_actualvalues = np.array(y_train_N225)
qlikeN225_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HAR_1_In.append(iteration)
QLIKE_HAR_1_N225_In = sum(qlikeN225_HAR_1_In)/len(y_actualvalues)

# HAR OMXC20

y_forecastvalues = np.array(fcHAR_OMXC20_1_In)
y_actualvalues = np.array(y_train_OMXC20)
qlikeOMXC20_HAR_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HAR_1_In.append(iteration)
QLIKE_HAR_1_OMXC20_In = sum(qlikeOMXC20_HAR_1_In)/len(y_actualvalues)

# HARlog DJI

y_forecastvalues = np.array(fcHARlog_DJI_1_adj_In)
y_actualvalues = np.array(y_train_DJI)
qlikeDJI_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_DJI_In = sum(qlikeDJI_HARlog_1_In)/len(y_actualvalues)

# HARlog FTSE

y_forecastvalues = np.array(fcHARlog_FTSE_1_adj_In)
y_actualvalues = np.array(y_train_FTSE)
qlikeFTSE_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_FTSE_In = sum(qlikeFTSE_HARlog_1_In)/len(y_actualvalues)

# HARlog FTSEMIB

y_forecastvalues = np.array(fcHARlog_FTSEMIB_1_adj_In)
y_actualvalues = np.array(y_train_FTSEMIB)
qlikeFTSEMIB_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_FTSEMIB_In = sum(qlikeFTSEMIB_HARlog_1_In)/len(y_actualvalues)

# HARlog GDAXI

y_forecastvalues = np.array(fcHARlog_GDAXI_1_adj_In)
y_actualvalues = np.array(y_train_GDAXI)
qlikeGDAXI_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_GDAXI_In = sum(qlikeGDAXI_HARlog_1_In)/len(y_actualvalues)

# HARlog SPX

y_forecastvalues = np.array(fcHARlog_SPX_1_adj_In)
y_actualvalues = np.array(y_train_SPX)
qlikeSPX_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_SPX_In = sum(qlikeSPX_HARlog_1_In)/len(y_actualvalues)

# HARlog HSI

y_forecastvalues = np.array(fcHARlog_HSI_1_adj_In)
y_actualvalues = np.array(y_train_HSI)
qlikeHSI_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_HSI_In = sum(qlikeHSI_HARlog_1_In)/len(y_actualvalues)
    
# HARlog IBEX

y_forecastvalues = np.array(fcHARlog_IBEX_1_adj_In)
y_actualvalues = np.array(y_train_IBEX)
qlikeIBEX_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_IBEX_In = sum(qlikeIBEX_HARlog_1_In)/len(y_actualvalues)

# HARlog IXIC

y_forecastvalues = np.array(fcHARlog_IXIC_1_adj_In)
y_actualvalues = np.array(y_train_IXIC)
qlikeIXIC_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_IXIC_In = sum(qlikeIXIC_HARlog_1_In)/len(y_actualvalues)

# HARlog N225

y_forecastvalues = np.array(fcHARlog_N225_1_adj_In)
y_actualvalues = np.array(y_train_N225)
qlikeN225_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_N225_In = sum(qlikeN225_HARlog_1_In)/len(y_actualvalues)

# HARlog OMXC20

y_forecastvalues = np.array(fcHARlog_OMXC20_1_adj_In)
y_actualvalues = np.array(y_train_OMXC20)
qlikeOMXC20_HARlog_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_HARlog_1_In.append(iteration)
QLIKE_HARlog_1_OMXC20_In = sum(qlikeOMXC20_HARlog_1_In)/len(y_actualvalues)

#%% QLIKE In

arrays_In = {
    'qlike_AR1': {
        'DJI': qlikeDJI_AR1_1_In,
        'FTSE': qlikeFTSE_AR1_1_In,
        'FTSEMIB': qlikeFTSEMIB_AR1_1_In,
        'GDAXI': qlikeGDAXI_AR1_1_In,
        'SPX': qlikeSPX_AR1_1_In,
        'HSI': qlikeHSI_AR1_1_In,
        'IBEX': qlikeIBEX_AR1_1_In,
        'IXIC': qlikeIXIC_AR1_1_In,
        'N225': qlikeN225_AR1_1_In,
        'OMXC20': qlikeOMXC20_AR1_1_In
    },
    'qlike_HAR': {
        'DJI': qlikeDJI_HAR_1_In,
        'FTSE': qlikeFTSE_HAR_1_In,
        'FTSEMIB': qlikeFTSEMIB_HAR_1_In,
        'GDAXI': qlikeGDAXI_HAR_1_In,
        'SPX': qlikeSPX_HAR_1_In,
        'HSI': qlikeHSI_HAR_1_In,
        'IBEX': qlikeIBEX_HAR_1_In,
        'IXIC': qlikeIXIC_HAR_1_In,
        'N225': qlikeN225_HAR_1_In,
        'OMXC20': qlikeOMXC20_HAR_1_In
    },
    'qlike_HARlog': {
        'DJI': qlikeDJI_HARlog_1_In, 
        'FTSE': qlikeFTSE_HARlog_1_In,
        'FTSEMIB': qlikeFTSEMIB_HARlog_1_In,
        'GDAXI': qlikeGDAXI_HARlog_1_In,
        'SPX': qlikeSPX_HARlog_1_In,
        'HSI': qlikeHSI_HARlog_1_In,
        'IBEX': qlikeIBEX_HARlog_1_In,
        'IXIC': qlikeIXIC_HARlog_1_In,
        'N225': qlikeN225_HARlog_1_In,
        'OMXC20': qlikeOMXC20_HARlog_1_In
                    }
    }


for k1 in arrays_In:
    if k1 == 'qlike_AR1':    
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_AR1_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
    elif k1 == 'qlike_HAR':
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_HAR_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
    elif k1 == 'qlike_HARlog':
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_HARlog_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')

#%%

DJI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\DJI.csv', index=True)
FTSE.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\FTSE.csv', index=True)
FTSEMIB.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\FTSEMIB.csv', index=True)
GDAXI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\GDAXI.csv', index=True)
SPX.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\SPX.csv', index=True)
HSI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\HSI.csv', index=True)
IBEX.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\IBEX.csv', index=True)
IXIC.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\IXIC.csv', index=True)
N225.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\N225.csv', index=True)
OMXC20.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\OMXC20.csv', index=True)

#%%

y_pred_AR1_1_DJI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\DJI.csv', index=True)
y_pred_AR1_1_FTSE.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\FTSE.csv', index=True)
y_pred_AR1_1_FTSEMIB.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\FTSEMIB.csv', index=True)
y_pred_AR1_1_GDAXI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\GDAXI.csv', index=True)
y_pred_AR1_1_SPX.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\SPX.csv', index=True)
y_pred_AR1_1_HSI.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\HSI.csv', index=True)
y_pred_AR1_1_IBEX.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\IBEX.csv', index=True)
y_pred_AR1_1_IXIC.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\IXIC.csv', index=True)
y_pred_AR1_1_N225.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\N225.csv', index=True)
y_pred_AR1_1_OMXC20.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\OMXC20.csv', index=True)

#%%

fcHAR_DJI_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_DJI_in.csv', index=True)
fcHAR_FTSE_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_FTSE_in.csv', index=True)
fcHAR_FTSEMIB_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_FTSEMIB_in.csv', index=True)
fcHAR_GDAXI_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_GDAXI_in.csv', index=True)
fcHAR_SPX_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_SPX_in.csv', index=True)
fcHAR_HSI_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_HSI_in.csv', index=True)
fcHAR_IBEX_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_IBEX_in.csv', index=True)
fcHAR_IXIC_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_IXIC_in.csv', index=True)
fcHAR_N225_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_N225_in.csv', index=True)
fcHAR_OMXC20_1_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_OMXC20_in.csv', index=True)

fcHARlog_DJI_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_DJI_in.csv', index=True)
fcHARlog_FTSE_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_FTSE_in.csv', index=True)
fcHARlog_FTSEMIB_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_FTSEMIB_in.csv', index=True)
fcHARlog_GDAXI_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_GDAXI_in.csv', index=True)
fcHARlog_SPX_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_SPX_in.csv', index=True)
fcHARlog_HSI_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_HSI_in.csv', index=True)
fcHARlog_IBEX_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_IBEX_in.csv', index=True)
fcHARlog_IXIC_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_IXIC_in.csv', index=True)
fcHARlog_N225_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_N225_in.csv', index=True)
fcHARlog_OMXC20_1_adj_In.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_OMXC20_in.csv', index=True)

fcHAR_DJI_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_DJI.csv', index=True)
fcHAR_FTSE_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_FTSE.csv', index=True)
fcHAR_FTSEMIB_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_FTSEMIB.csv', index=True)
fcHAR_GDAXI_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_GDAXI.csv', index=True)
fcHAR_SPX_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_SPX.csv', index=True)
fcHAR_HSI_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_HSI.csv', index=True)
fcHAR_IBEX_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_IBEX.csv', index=True)
fcHAR_IXIC_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_IXIC.csv', index=True)
fcHAR_N225_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_N225.csv', index=True)
fcHAR_OMXC20_1.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_har_OMXC20.csv', index=True)

fcHARlog_DJI_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_DJI.csv', index=True)
fcHARlog_FTSE_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_FTSE.csv', index=True)
fcHARlog_FTSEMIB_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_FTSEMIB.csv', index=True)
fcHARlog_GDAXI_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_GDAXI.csv', index=True)
fcHARlog_SPX_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_SPX.csv', index=True)
fcHARlog_HSI_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_HSI.csv', index=True)
fcHARlog_IBEX_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_IBEX.csv', index=True)
fcHARlog_IXIC_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_IXIC.csv', index=True)
fcHARlog_N225_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_N225.csv', index=True)
fcHARlog_OMXC20_1_adj.to_csv(r'C:\Users\cesar\Desktop\Master Thesis\forecast_harlog_OMXC20.csv', index=True)








