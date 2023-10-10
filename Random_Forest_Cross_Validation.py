# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:38:46 2023

@author: cesar
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#%% Optimal parameters form cross validation

lag_DJI = 17
param_DJI = {'max_depth':None, 'n_estimators':100}

lag_FTSE = 1
param_FTSE = {'max_depth':5, 'n_estimators':200}

lag_FTSEMIB = 1
param_FTSEMIB = {'max_depth':5, 'n_estimators':100}

lag_GDAXI = 1
param_GDAXI = {'max_depth':5, 'n_estimators':10}

lag_SPX = 21
param_SPX = {'max_depth':20, 'n_estimators':200}

lag_HSI = 1
param_HSI = {'max_depth':5, 'n_estimators':100}

lag_IBEX = 1
param_IBEX = {'max_depth':5, 'n_estimators':200}

lag_IXIC = 21
param_IXIC = {'max_depth':5, 'n_estimators':200}

lag_N225 = 1
param_N225 = {'max_depth':5, 'n_estimators':200}

lag_OMXC20 = 1
param_OMXC20 = {'max_depth':5, 'n_estimators':50}

#%% Random Forest (Cross Validation)

data_cv = np.array(DJI).reshape(-1,1).ravel()
#data_cv.name = 'y'
select_lag = 21
tscv = TimeSeriesSplit(n_splits=10)
scoring='neg_mean_squared_error'
random.seed(1)

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, None]
}

# crea una random forest regressor
model = RandomForestRegressor(random_state=1)

# esegui la ricerca sulla griglia di parametri
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring=scoring)

best_lag = 0
best_score = float('inf')
for lag in range(1, select_lag+1):
    # crea le feature basate sui lag della serie temporale
    X = np.zeros((len(data_cv) - lag, lag))
    for i in range(lag, len(data_cv)):
        X[i - lag] = data_cv[i - lag:i]

        # Crea le etichette
    y = data_cv[lag:]
    
    # divide il dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=False)

    grid_search.fit(X_train, y_train.ravel())

    if grid_search.best_score_ < best_score:
        best_lag = lag
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

print("I migliori parametri sono:", best_params)
print("Il miglior numero di lag è:", best_lag)

#%%

###################################################
# Parameters for DJI
###################################################

# =============================================================================
# best_params_DJI = best_params
# best_lag_DJI = best_lag
# =============================================================================

data = DJI
best_params_DJI = param_DJI
best_lag_DJI = lag_DJI

# addestra una nuova random forest regressor con i migliori parametri
best_regressor_DJI = RandomForestRegressor(n_estimators=best_params_DJI['n_estimators'],
                                       max_depth=best_params_DJI['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_DJI_rf = np.zeros((len(data) - best_lag_DJI, best_lag_DJI))
X_DJI_rf = pd.DataFrame(index=data.index[best_lag_DJI:], columns=[f'lag_{i}' for i in range(1, best_lag_DJI+1)])

for i in range(best_lag_DJI, len(data)):
    X_DJI_rf.loc[data.index[i]] = data[i - best_lag_DJI:i].T.values

# Crea le etichette
y_DJI_rf = data[best_lag_DJI:]

# Train and test set

X_train_DJI_rf = X_DJI_rf[:threshold_train_DJI]
y_train_DJI_rf = np.ravel(y_DJI_rf[:threshold_train_DJI])
X_test_DJI_rf = X_DJI_rf[threshold_test_DJI:]
y_test_DJI_rf = y_DJI_rf[threshold_test_DJI:]
y_test_DJI_rf = y_test_DJI_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_DJI.fit(X_train_DJI_rf, y_train_DJI_rf)

# effettua la previsione sul test set
y_pred_1_DJI_rf = best_regressor_DJI.predict(X_test_DJI_rf[21:])

# MSE Out of Sample and QLIKE loss

MSE_DJI_1_rf = mean_squared_error(y_test_DJI_rf, y_pred_1_DJI_rf)

y_forecastvalues = np.array(y_pred_1_DJI_rf)
y_actualvalues = np.array(y_test_DJI_rf)
qlikeDJI_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_rf_1_Out.append(iteration)
qlikeDJI_rf_1_Out = np.array(qlikeDJI_rf_1_Out)
QLIKE_rf_1_DJI = sum(qlikeDJI_rf_1_Out)/len(y_actualvalues)

###################################################
# Parameters for FTSE
###################################################

# =============================================================================
# best_params_FTSE = best_params
# best_lag_FTSE = best_lag
# =============================================================================

data = FTSE
best_params_FTSE = param_FTSE
best_lag_FTSE = lag_FTSE

# addestra una nuova random forest regressor con i migliori parametri
best_regressor_FTSE = RandomForestRegressor(n_estimators=best_params_FTSE['n_estimators'],
                                       max_depth=best_params_FTSE['max_depth'],
                                       random_state=1)

# crea le features basate sul miglior numero di lag della serie temporale


X_FTSE_rf = np.zeros((len(data) - best_lag_FTSE, best_lag_FTSE))
X_FTSE_rf = pd.DataFrame(index=data.index[best_lag_FTSE:], columns=[f'lag_{i}' for i in range(1, best_lag_FTSE+1)])

for i in range(best_lag_FTSE, len(data)):
    X_FTSE_rf.loc[data.index[i]] = data[i - best_lag_FTSE:i].T.values

# Crea le etichette
y_FTSE_rf = data[best_lag_FTSE:]

# Train and test set

X_train_FTSE_rf = X_FTSE_rf[:threshold_train_FTSE]
y_train_FTSE_rf = y_FTSE_rf[:threshold_train_FTSE]
X_test_FTSE_rf = X_FTSE_rf[threshold_test_FTSE:]
y_test_FTSE_rf = y_FTSE_rf[threshold_test_FTSE:]
y_test_FTSE_rf = y_test_FTSE_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_FTSE.fit(X_train_FTSE_rf, y_train_FTSE_rf)

# effettua la previsione sul test set
y_pred_1_FTSE_rf = best_regressor_FTSE.predict(X_test_FTSE_rf[21:])

# MSE Out of Sample

MSE_FTSE_1_rf = mean_squared_error(y_test_FTSE_rf, y_pred_1_FTSE_rf)

y_forecastvalues = np.array(y_pred_1_FTSE_rf)
y_actualvalues = np.array(y_test_FTSE_rf)
qlikeFTSE_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_rf_1_Out.append(iteration)
qlikeFTSE_rf_1_Out = np.array(qlikeFTSE_rf_1_Out)
QLIKE_rf_1_FTSE = sum(qlikeFTSE_rf_1_Out)/len(y_actualvalues)

###################################################
# Prameters for FTSEMIB
###################################################

# =============================================================================
# best_params_FTSEMIB = best_params
# best_lag_FTSEMIB = best_lag
# =============================================================================

data = FTSEMIB
best_params_FTSEMIB = param_FTSEMIB
best_lag_FTSEMIB = lag_FTSEMIB

# addestra una nuova random forest regressor con i migliori parametri
best_regressor_FTSEMIB = RandomForestRegressor(n_estimators=best_params_FTSEMIB['n_estimators'],
                                       max_depth=best_params_FTSEMIB['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_FTSEMIB_rf = np.zeros((len(data) - best_lag_FTSEMIB, best_lag_FTSEMIB))
X_FTSEMIB_rf = pd.DataFrame(index=data.index[best_lag_FTSEMIB:], columns=[f'lag_{i}' for i in range(1, best_lag_FTSEMIB+1)])

for i in range(best_lag_FTSEMIB, len(data)):
    X_FTSEMIB_rf.loc[data.index[i]] = data[i - best_lag_FTSEMIB:i].T.values

# Crea le etichette
y_FTSEMIB_rf = data[best_lag_FTSEMIB:]

# Train and test set

X_train_FTSEMIB_rf = X_FTSEMIB_rf[:threshold_train_FTSEMIB]
y_train_FTSEMIB_rf = np.ravel(y_FTSEMIB_rf[:threshold_train_FTSEMIB])
X_test_FTSEMIB_rf = X_FTSEMIB_rf[threshold_test_FTSEMIB:]
y_test_FTSEMIB_rf = y_FTSEMIB_rf[threshold_test_FTSEMIB:] 
y_test_FTSEMIB_rf = y_test_FTSEMIB_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_FTSEMIB.fit(X_train_FTSEMIB_rf, y_train_FTSEMIB_rf)

# effettua la previsione sul test set
y_pred_1_FTSEMIB_rf = best_regressor_FTSEMIB.predict(X_test_FTSEMIB_rf[21:])

# MSE Out of Sample

MSE_FTSEMIB_1_rf = mean_squared_error(y_test_FTSEMIB_rf, y_pred_1_FTSEMIB_rf)

y_forecastvalues = np.array(y_pred_1_FTSEMIB_rf)
y_actualvalues = np.array(y_test_FTSEMIB_rf)
qlikeFTSEMIB_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_rf_1_Out.append(iteration)
qlikeFTSEMIB_rf_1_Out = np.array(qlikeFTSEMIB_rf_1_Out)
QLIKE_rf_1_FTSEMIB = sum(qlikeFTSEMIB_rf_1_Out)/len(y_actualvalues)

###################################################
# Parameters for GDAXI
###################################################

# =============================================================================
# best_params_GDAXI = best_params
# best_lag_GDAXI = best_lag
# =============================================================================

data = GDAXI
best_params_GDAXI = param_GDAXI
best_lag_GDAXI = lag_GDAXI

# addestra una nuova random forest regressor con i migliori parametri
best_regressor_GDAXI = RandomForestRegressor(n_estimators=best_params_GDAXI['n_estimators'],
                                       max_depth=best_params_GDAXI['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_GDAXI_rf = np.zeros((len(data) - best_lag_GDAXI, best_lag_GDAXI))
X_GDAXI_rf = pd.DataFrame(index=data.index[best_lag_GDAXI:], columns=[f'lag_{i}' for i in range(1, best_lag_GDAXI+1)])

for i in range(best_lag_GDAXI, len(data)):
    X_GDAXI_rf.loc[data.index[i]] = data[i - best_lag_GDAXI:i].T.values

# Crea le etichette
y_GDAXI_rf = data[best_lag_GDAXI:]

# Train and test set

X_train_GDAXI_rf = X_GDAXI_rf[:threshold_train_GDAXI]
y_train_GDAXI_rf = y_GDAXI_rf[:threshold_train_GDAXI]
X_test_GDAXI_rf = X_GDAXI_rf[threshold_test_GDAXI:]
y_test_GDAXI_rf = y_GDAXI_rf[threshold_test_GDAXI:]
y_test_GDAXI_rf = y_test_GDAXI_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_GDAXI.fit(X_train_GDAXI_rf, y_train_GDAXI_rf)

# effettua la previsione sul test set
y_pred_1_GDAXI_rf = best_regressor_GDAXI.predict(X_test_GDAXI_rf[21:])

# MSE Out of Sample

MSE_GDAXI_1_rf = mean_squared_error(y_test_GDAXI_rf, y_pred_1_GDAXI_rf)

y_forecastvalues = np.array(y_pred_1_GDAXI_rf)
y_actualvalues = np.array(y_test_GDAXI_rf)
qlikeGDAXI_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_rf_1_Out.append(iteration)
qlikeGDAXI_rf_1_Out = np.array(qlikeGDAXI_rf_1_Out)
QLIKE_rf_1_GDAXI = sum(qlikeGDAXI_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for SPX
###################################################

# =============================================================================
# best_params_SPX = best_params
# best_lag_SPX = best_lag
# =============================================================================

data = SPX
best_params_SPX = param_SPX
best_lag_SPX = lag_SPX


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_SPX = RandomForestRegressor(n_estimators=best_params_SPX['n_estimators'],
                                       max_depth=best_params_SPX['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_SPX_rf = np.zeros((len(data) - best_lag_SPX, best_lag_SPX))
X_SPX_rf = pd.DataFrame(index=data.index[best_lag_SPX:], columns=[f'lag_{i}' for i in range(1, best_lag_SPX+1)])

for i in range(best_lag_SPX, len(data)):
    X_SPX_rf.loc[data.index[i]] = data[i - best_lag_SPX:i].T.values

# Crea le etichette
y_SPX_rf = data[best_lag_SPX:]

# Train and test set

X_train_SPX_rf = X_SPX_rf[:threshold_train_SPX]
y_train_SPX_rf = y_SPX_rf[:threshold_train_SPX]
X_test_SPX_rf = X_SPX_rf[threshold_test_SPX:]
y_test_SPX_rf = y_SPX_rf[threshold_test_SPX:]
y_test_SPX_rf = y_test_SPX_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_SPX.fit(X_train_SPX_rf, y_train_SPX_rf)

# effettua la previsione sul test set
y_pred_1_SPX_rf = best_regressor_SPX.predict(X_test_SPX_rf[21:])

# MSE Out of Sample

MSE_SPX_1_rf = mean_squared_error(y_test_SPX_rf, y_pred_1_SPX_rf)

y_forecastvalues = np.array(y_pred_1_SPX_rf)
y_actualvalues = np.array(y_test_SPX_rf)
qlikeSPX_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_rf_1_Out.append(iteration)
qlikeSPX_rf_1_Out = np.array(qlikeSPX_rf_1_Out)
QLIKE_rf_1_SPX = sum(qlikeSPX_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for HSI
###################################################

# =============================================================================
# best_params_HSI = best_params
# best_lag_HSI = best_lag
# =============================================================================

data = HSI
best_params_HSI = param_HSI
best_lag_HSI = lag_HSI


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_HSI = RandomForestRegressor(n_estimators=best_params_HSI['n_estimators'],
                                       max_depth=best_params_HSI['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_HSI_rf = np.zeros((len(data) - best_lag_HSI, best_lag_HSI))
X_HSI_rf = pd.DataFrame(index=data.index[best_lag_HSI:], columns=[f'lag_{i}' for i in range(1, best_lag_HSI+1)])

for i in range(best_lag_HSI, len(data)):
    X_HSI_rf.loc[data.index[i]] = data[i - best_lag_HSI:i].T.values

# Crea le etichette
y_HSI_rf = data[best_lag_HSI:]

# Train and test set

X_train_HSI_rf = X_HSI_rf[:threshold_train_HSI]
y_train_HSI_rf = y_HSI_rf[:threshold_train_HSI]
X_test_HSI_rf = X_HSI_rf[threshold_test_HSI:]
y_test_HSI_rf = y_HSI_rf[threshold_test_HSI:]
y_test_HSI_rf = y_test_HSI_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_HSI.fit(X_train_HSI_rf, y_train_HSI_rf)

# effettua la previsione sul test set
y_pred_1_HSI_rf = best_regressor_HSI.predict(X_test_HSI_rf[21:])

# MSE Out of Sample

MSE_HSI_1_rf = mean_squared_error(y_test_HSI_rf, y_pred_1_HSI_rf)

y_forecastvalues = np.array(y_pred_1_HSI_rf)
y_actualvalues = np.array(y_test_HSI_rf)
qlikeHSI_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_rf_1_Out.append(iteration)
qlikeHSI_rf_1_Out = np.array(qlikeHSI_rf_1_Out)
QLIKE_rf_1_HSI = sum(qlikeHSI_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for IBEX
###################################################

# =============================================================================
# best_params_IBEX = best_params
# best_lag_IBEX = best_lag
# =============================================================================

data = IBEX
best_params_IBEX = param_IBEX
best_lag_IBEX = lag_IBEX


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_IBEX = RandomForestRegressor(n_estimators=best_params_IBEX['n_estimators'],
                                       max_depth=best_params_IBEX['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_IBEX_rf = np.zeros((len(data) - best_lag_IBEX, best_lag_IBEX))
X_IBEX_rf = pd.DataFrame(index=data.index[best_lag_IBEX:], columns=[f'lag_{i}' for i in range(1, best_lag_IBEX+1)])

for i in range(best_lag_IBEX, len(data)):
    X_IBEX_rf.loc[data.index[i]] = data[i - best_lag_IBEX:i].T.values

# Crea le etichette
y_IBEX_rf = data[best_lag_IBEX:]

# Train and test set

X_train_IBEX_rf = X_IBEX_rf[:threshold_train_IBEX]
y_train_IBEX_rf = y_IBEX_rf[:threshold_train_IBEX]
X_test_IBEX_rf = X_IBEX_rf[threshold_test_IBEX:]
y_test_IBEX_rf = y_IBEX_rf[threshold_test_IBEX:]
y_test_IBEX_rf = y_test_IBEX_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_IBEX.fit(X_train_IBEX_rf, y_train_IBEX_rf)

# effettua la previsione sul test set
y_pred_1_IBEX_rf = best_regressor_IBEX.predict(X_test_IBEX_rf[21:])

# MSE Out of Sample

MSE_IBEX_1_rf = mean_squared_error(y_test_IBEX_rf, y_pred_1_IBEX_rf)

y_forecastvalues = np.array(y_pred_1_IBEX_rf)
y_actualvalues = np.array(y_test_IBEX_rf)
qlikeIBEX_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_rf_1_Out.append(iteration)
qlikeIBEX_rf_1_Out = np.array(qlikeIBEX_rf_1_Out)
QLIKE_rf_1_IBEX = sum(qlikeIBEX_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for IXIC
###################################################

# =============================================================================
# best_params_IXIC = best_params
# best_lag_IXIC = best_lag
# =============================================================================

data = IXIC
best_params_IXIC = param_IXIC
best_lag_IXIC = lag_IXIC


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_IXIC = RandomForestRegressor(n_estimators=best_params_IXIC['n_estimators'],
                                       max_depth=best_params_IXIC['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_IXIC_rf = np.zeros((len(data) - best_lag_IXIC, best_lag_IXIC))
X_IXIC_rf = pd.DataFrame(index=data.index[best_lag_IXIC:], columns=[f'lag_{i}' for i in range(1, best_lag_IXIC+1)])

for i in range(best_lag_IXIC, len(data)):
    X_IXIC_rf.loc[data.index[i]] = data[i - best_lag_IXIC:i].T.values

# Crea le etichette
y_IXIC_rf = data[best_lag_IXIC:]

# Train and test set

X_train_IXIC_rf = X_IXIC_rf[:threshold_train_IXIC]
y_train_IXIC_rf = y_IXIC_rf[:threshold_train_IXIC]
X_test_IXIC_rf = X_IXIC_rf[threshold_test_IXIC:]
y_test_IXIC_rf = y_IXIC_rf[threshold_test_IXIC:]
y_test_IXIC_rf = y_test_IXIC_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_IXIC.fit(X_train_IXIC_rf, y_train_IXIC_rf)

# effettua la previsione sul test set
y_pred_1_IXIC_rf = best_regressor_IXIC.predict(X_test_IXIC_rf[21:])

# MSE Out of Sample

MSE_IXIC_1_rf = mean_squared_error(y_test_IXIC_rf, y_pred_1_IXIC_rf)

y_forecastvalues = np.array(y_pred_1_IXIC_rf)
y_actualvalues = np.array(y_test_IXIC_rf)
qlikeIXIC_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_rf_1_Out.append(iteration)
qlikeIXIC_rf_1_Out = np.array(qlikeIXIC_rf_1_Out)
QLIKE_rf_1_IXIC = sum(qlikeIXIC_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for N225
###################################################

# =============================================================================
# best_params_N225 = best_params
# best_lag_N225 = best_lag
# =============================================================================

data = N225
best_params_N225 = param_N225
best_lag_N225 = lag_N225


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_N225 = RandomForestRegressor(n_estimators=best_params_N225['n_estimators'],
                                       max_depth=best_params_N225['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_N225_rf = np.zeros((len(data) - best_lag_N225, best_lag_N225))
X_N225_rf = pd.DataFrame(index=data.index[best_lag_N225:], columns=[f'lag_{i}' for i in range(1, best_lag_N225+1)])

for i in range(best_lag_N225, len(data)):
    X_N225_rf.loc[data.index[i]] = data[i - best_lag_N225:i].T.values

# Crea le etichette
y_N225_rf = data[best_lag_N225:]

# Train and test set

X_train_N225_rf = X_N225_rf[:threshold_train_N225]
y_train_N225_rf = y_N225_rf[:threshold_train_N225]
X_test_N225_rf = X_N225_rf[threshold_test_N225:]
y_test_N225_rf = y_N225_rf[threshold_test_N225:]
y_test_N225_rf = y_test_N225_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_N225.fit(X_train_N225_rf, y_train_N225_rf)

# effettua la previsione sul test set
y_pred_1_N225_rf = best_regressor_N225.predict(X_test_N225_rf[21:])

# MSE Out of Sample

MSE_N225_1_rf = mean_squared_error(y_test_N225_rf, y_pred_1_N225_rf)

y_forecastvalues = np.array(y_pred_1_N225_rf)
y_actualvalues = np.array(y_test_N225_rf)
qlikeN225_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_rf_1_Out.append(iteration)
qlikeN225_rf_1_Out = np.array(qlikeN225_rf_1_Out)
QLIKE_rf_1_N225 = sum(qlikeN225_rf_1_Out)/len(y_actualvalues)

###################################################
#Parameters for OMXC20
###################################################

# =============================================================================
# best_params_OMXC20 = best_params
# best_lag_OMXC20 = best_lag
# =============================================================================

data = OMXC20
best_params_OMXC20 = param_OMXC20
best_lag_OMXC20 = lag_OMXC20


# addestra una nuova random forest regressor con i migliori parametri
best_regressor_OMXC20 = RandomForestRegressor(n_estimators=best_params_OMXC20['n_estimators'],
                                       max_depth=best_params_OMXC20['max_depth'],
                                       random_state=1)

# crea le feature basate sul miglior numero di lag della serie temporale


X_OMXC20_rf = np.zeros((len(data) - best_lag_OMXC20, best_lag_OMXC20))
X_OMXC20_rf = pd.DataFrame(index=data.index[best_lag_OMXC20:], columns=[f'lag_{i}' for i in range(1, best_lag_OMXC20+1)])

for i in range(best_lag_OMXC20, len(data)):
    X_OMXC20_rf.loc[data.index[i]] = data[i - best_lag_OMXC20:i].T.values

# Crea le etichette
y_OMXC20_rf = data[best_lag_OMXC20:]

# Train and test set

X_train_OMXC20_rf = X_OMXC20_rf[:threshold_train_OMXC20]
y_train_OMXC20_rf = y_OMXC20_rf[:threshold_train_OMXC20]
X_test_OMXC20_rf = X_OMXC20_rf[threshold_test_OMXC20:]
y_test_OMXC20_rf = y_OMXC20_rf[threshold_test_OMXC20:]
y_test_OMXC20_rf = y_test_OMXC20_rf[21:]

# addestra il modello sui dati di addestramento
best_regressor_OMXC20.fit(X_train_OMXC20_rf, y_train_OMXC20_rf)

# effettua la previsione sul test set
y_pred_1_OMXC20_rf = best_regressor_OMXC20.predict(X_test_OMXC20_rf[21:])

# MSE Out of Sample

MSE_OMXC20_1_rf = mean_squared_error(y_test_OMXC20_rf, y_pred_1_OMXC20_rf)

y_forecastvalues = np.array(y_pred_1_OMXC20_rf)
y_actualvalues = np.array(y_test_OMXC20_rf)
qlikeOMXC20_rf_1_Out = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_rf_1_Out.append(iteration)
qlikeOMXC20_rf_1_Out = np.array(qlikeOMXC20_rf_1_Out)
QLIKE_rf_1_OMXC20 = sum(qlikeOMXC20_rf_1_Out)/len(y_actualvalues)

#%% In-sample Loss Function

y_pred_1_DJI_rf_In = best_regressor_DJI.predict(X_train_DJI_rf[4:])
MSE_DJI_1_rf_In = mean_squared_error(y_train_DJI_rf[4:], y_pred_1_DJI_rf_In)
mseDJI_rf_In = []
for i in np.arange(len(y_pred_1_DJI_rf_In)):
    mse = (y_pred_1_DJI_rf_In.reshape(-1,1)[i]-y_train_DJI_rf[4:][i])**2
    mseDJI_rf_In.append(mse)
mseDJI_rf_In = np.array(mseDJI_rf_In)

y_pred_1_FTSE_rf_In = best_regressor_FTSE.predict(X_train_FTSE_rf[20:])
MSE_FTSE_1_rf_In = mean_squared_error(y_train_FTSE_rf[20:], y_pred_1_FTSE_rf_In)
mseFTSE_rf_In = []
for i in np.arange(len(y_pred_1_FTSE_rf_In)):
    mse = (y_pred_1_FTSE_rf_In.reshape(-1,1)[i]-y_train_FTSE_rf[20:][i])**2
    mseFTSE_rf_In.append(mse)
mseFTSE_rf_In = np.array(mseFTSE_rf_In)

y_pred_1_FTSEMIB_rf_In = best_regressor_FTSEMIB.predict(X_train_FTSEMIB_rf[20:])
MSE_FTSEMIB_1_rf_In = mean_squared_error(y_train_FTSEMIB_rf[20:], y_pred_1_FTSEMIB_rf_In)
mseFTSEMIB_rf_In = []
for i in np.arange(len(y_pred_1_FTSEMIB_rf_In)):
    mse = (y_pred_1_FTSEMIB_rf_In.reshape(-1,1)[i]-y_train_FTSEMIB_rf[20:][i])**2
    mseFTSEMIB_rf_In.append(mse)
mseFTSEMIB_rf_In = np.array(mseFTSEMIB_rf_In)

y_pred_1_GDAXI_rf_In = best_regressor_GDAXI.predict(X_train_GDAXI_rf[20:])
MSE_GDAXI_1_rf_In = mean_squared_error(y_train_GDAXI_rf[20:], y_pred_1_GDAXI_rf_In)
mseGDAXI_rf_In = []
for i in np.arange(len(y_pred_1_GDAXI_rf_In)):
    mse = (y_pred_1_GDAXI_rf_In.reshape(-1,1)[i]-y_train_GDAXI_rf[20:][i])**2
    mseGDAXI_rf_In.append(mse)
mseGDAXI_rf_In = np.array(mseGDAXI_rf_In)

y_pred_1_SPX_rf_In = best_regressor_SPX.predict(X_train_SPX_rf)
MSE_SPX_1_rf_In = mean_squared_error(y_train_SPX_rf, y_pred_1_SPX_rf_In)
mseSPX_rf_In = []
for i in np.arange(len(y_pred_1_SPX_rf_In)):
    mse = (y_pred_1_SPX_rf_In.reshape(-1,1)[i]-y_train_SPX_rf[i])**2
    mseSPX_rf_In.append(mse)
mseSPX_rf_In = np.array(mseSPX_rf_In)

y_pred_1_HSI_rf_In = best_regressor_HSI.predict(X_train_HSI_rf[20:])
MSE_HSI_1_rf_In = mean_squared_error(y_train_HSI_rf[20:], y_pred_1_HSI_rf_In)
mseHSI_rf_In = []
for i in np.arange(len(y_pred_1_HSI_rf_In)):
    mse = (y_pred_1_HSI_rf_In.reshape(-1,1)[i]-y_train_HSI_rf[20:][i])**2
    mseHSI_rf_In.append(mse)
mseHSI_rf_In = np.array(mseHSI_rf_In)

y_pred_1_IBEX_rf_In = best_regressor_IBEX.predict(X_train_IBEX_rf[20:])
MSE_IBEX_1_rf_In = mean_squared_error(y_train_IBEX_rf[20:], y_pred_1_IBEX_rf_In)
mseIBEX_rf_In = []
for i in np.arange(len(y_pred_1_IBEX_rf_In)):
    mse = (y_pred_1_IBEX_rf_In.reshape(-1,1)[i]-y_train_IBEX_rf[20:][i])**2
    mseIBEX_rf_In.append(mse)
mseIBEX_rf_In = np.array(mseIBEX_rf_In)

y_pred_1_IXIC_rf_In = best_regressor_IXIC.predict(X_train_IXIC_rf)
MSE_IXIC_1_rf_In = mean_squared_error(y_train_IXIC_rf, y_pred_1_IXIC_rf_In)
mseIXIC_rf_In = []
for i in np.arange(len(y_pred_1_IXIC_rf_In)):
    mse = (y_pred_1_IXIC_rf_In.reshape(-1,1)[i]-y_train_IXIC_rf[i])**2
    mseIXIC_rf_In.append(mse)
mseIXIC_rf_In = np.array(mseIXIC_rf_In)

y_pred_1_N225_rf_In = best_regressor_N225.predict(X_train_N225_rf[20:])
MSE_N225_1_rf_In = mean_squared_error(y_train_N225_rf[20:], y_pred_1_N225_rf_In)
mseN225_rf_In = []
for i in np.arange(len(y_pred_1_N225_rf_In)):
    mse = (y_pred_1_N225_rf_In.reshape(-1,1)[i]-y_train_N225_rf[20:][i])**2
    mseN225_rf_In.append(mse)
mseN225_rf_In = np.array(mseN225_rf_In)

y_pred_1_OMXC20_rf_In = best_regressor_OMXC20.predict(X_train_OMXC20_rf[20:])
MSE_OMXC20_1_rf_In = mean_squared_error(y_train_OMXC20_rf[20:], y_pred_1_OMXC20_rf_In)
mseOMXC20_rf_In = []
for i in np.arange(len(y_pred_1_OMXC20_rf_In)):
    mse = (y_pred_1_OMXC20_rf_In.reshape(-1,1)[i]-y_train_OMXC20_rf[20:][i])**2
    mseOMXC20_rf_In.append(mse)
mseOMXC20_rf_In = np.array(mseOMXC20_rf_In)

#%% In sample

arrays_In = {
    'mse_rf': {
        'DJI': mseDJI_rf_In,
        'FTSE': mseFTSE_rf_In,
        'FTSEMIB': mseFTSEMIB_rf_In,
        'GDAXI': mseGDAXI_rf_In,
        'SPX': mseSPX_rf_In,
        'HSI': mseHSI_rf_In,
        'IBEX': mseIBEX_rf_In,
        'IXIC': mseIXIC_rf_In,
        'N225': mseN225_rf_In,
        'OMXC20': mseOMXC20_rf_In
            }
        }

for k1 in arrays_In:
    if k1 == 'mse_rf':    
        for k2 in arrays_In[k1]:
            nome_file = 'mse{}_rf_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
            
#%% Out-of sample MSE

mseDJI_rf_Out = []
for i in np.arange(len(y_test_DJI_rf)):
    mse = (y_pred_1_DJI_rf[i]-y_test_DJI_rf[i])**2
    mseDJI_rf_Out.append(mse)
mseDJI_rf_Out = np.array(mseDJI_rf_Out)

mseFTSE_rf_Out = []
for i in np.arange(len(y_test_FTSE_rf)):
    mse = (y_pred_1_FTSE_rf[i]-y_test_FTSE_rf[i])**2
    mseFTSE_rf_Out.append(mse)
mseFTSE_rf_Out = np.array(mseFTSE_rf_Out)

mseFTSEMIB_rf_Out = []
for i in np.arange(len(y_test_FTSEMIB_rf)):
    mse = (y_pred_1_FTSEMIB_rf[i]-y_test_FTSEMIB_rf[i])**2
    mseFTSEMIB_rf_Out.append(mse)
mseFTSEMIB_rf_Out = np.array(mseFTSEMIB_rf_Out)

mseGDAXI_rf_Out = []
for i in np.arange(len(y_test_GDAXI_rf)):
    mse = (y_pred_1_GDAXI_rf[i]-y_test_GDAXI_rf[i])**2
    mseGDAXI_rf_Out.append(mse)
mseGDAXI_rf_Out = np.array(mseGDAXI_rf_Out)

mseSPX_rf_Out = []
for i in np.arange(len(y_test_SPX_rf)):
    mse = (y_pred_1_SPX_rf[i]-y_test_SPX_rf[i])**2
    mseSPX_rf_Out.append(mse)
mseSPX_rf_Out = np.array(mseSPX_rf_Out)

mseHSI_rf_Out = []
for i in np.arange(len(y_test_HSI_rf)):
    mse = (y_pred_1_HSI_rf[i]-y_test_HSI_rf[i])**2
    mseHSI_rf_Out.append(mse)
mseHSI_rf_Out = np.array(mseHSI_rf_Out)

mseIBEX_rf_Out = []
for i in np.arange(len(y_test_IBEX_rf)):
    mse = (y_pred_1_IBEX_rf[i]-y_test_IBEX_rf[i])**2
    mseIBEX_rf_Out.append(mse)
mseIBEX_rf_Out = np.array(mseIBEX_rf_Out)

mseIXIC_rf_Out = []
for i in np.arange(len(y_test_IXIC_rf)):
    mse = (y_pred_1_IXIC_rf[i]-y_test_IXIC_rf[i])**2
    mseIXIC_rf_Out.append(mse)
mseIXIC_rf_Out = np.array(mseIXIC_rf_Out)

mseN225_rf_Out = []
for i in np.arange(len(y_test_N225_rf)):
    mse = (y_pred_1_N225_rf[i]-y_test_N225_rf[i])**2
    mseN225_rf_Out.append(mse)
mseN225_rf_Out = np.array(mseN225_rf_Out)

mseOMXC20_rf_Out = []
for i in np.arange(len(y_test_OMXC20_rf)):
    mse = (y_pred_1_OMXC20_rf[i]-y_test_OMXC20_rf[i])**2
    mseOMXC20_rf_Out.append(mse)
mseOMXC20_rf_Out = np.array(mseOMXC20_rf_Out)

#%% out of Sample

arrays_Out = {
    'mse_rf': {
        'DJI': mseDJI_rf_Out,
        'FTSE': mseFTSE_rf_Out,
        'FTSEMIB': mseFTSEMIB_rf_Out,
        'GDAXI': mseGDAXI_rf_Out,
        'SPX': mseSPX_rf_Out,
        'HSI': mseHSI_rf_Out,
        'IBEX': mseIBEX_rf_Out,
        'IXIC': mseIXIC_rf_Out,
        'N225': mseN225_rf_Out,
        'OMXC20': mseOMXC20_rf_Out
            }
        }

for k1 in arrays_Out:
    if k1 == 'mse_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'mse{}_rf_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')

#%% Qlike Out

arrays_Out = {
    'qlike_rf': {
        'DJI': qlikeDJI_rf_1_Out,
        'FTSE': qlikeFTSE_rf_1_Out,
        'FTSEMIB': qlikeFTSEMIB_rf_1_Out,
        'GDAXI': qlikeGDAXI_rf_1_Out,
        'SPX': qlikeSPX_rf_1_Out,
        'HSI': qlikeHSI_rf_1_Out,
        'IBEX': qlikeIBEX_rf_1_Out,
        'IXIC': qlikeIXIC_rf_1_Out,
        'N225': qlikeN225_rf_1_Out,
        'OMXC20': qlikeOMXC20_rf_1_Out
            }
        }

for k1 in arrays_Out:
    if k1 == 'qlike_rf':    
        for k2 in arrays_Out[k1]:
            nome_file = 'qlike{}_rf_Out.csv'.format(k2)
            np.savetxt(nome_file, arrays_Out[k1][k2], delimiter=',')
            
#%% Qlike In
 
y_forecastvalues = np.array(y_pred_1_DJI_rf_In)
y_actualvalues = np.array(y_train_DJI_rf[4:])
qlikeDJI_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeDJI_rf_1_In.append(iteration)
qlikeDJI_rf_1_In = np.array(qlikeDJI_rf_1_In)
QLIKE_rf_1_DJI_In = sum(qlikeDJI_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_FTSE_rf_In)
y_actualvalues = np.array(y_train_FTSE_rf[20:])
qlikeFTSE_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSE_rf_1_In.append(iteration)
qlikeFTSE_rf_1_In = np.array(qlikeFTSE_rf_1_In)
QLIKE_rf_1_FTSE_In = sum(qlikeFTSE_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_FTSEMIB_rf_In)
y_actualvalues = np.array(y_train_FTSEMIB_rf[20:])
qlikeFTSEMIB_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeFTSEMIB_rf_1_In.append(iteration)
qlikeFTSEMIB_rf_1_In = np.array(qlikeFTSEMIB_rf_1_In)
QLIKE_rf_1_FTSEMIB_In = sum(qlikeFTSEMIB_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_GDAXI_rf_In)
y_actualvalues = np.array(y_train_GDAXI_rf[20:])
qlikeGDAXI_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeGDAXI_rf_1_In.append(iteration)
qlikeGDAXI_rf_1_In = np.array(qlikeGDAXI_rf_1_In)
QLIKE_rf_1_GDAXI_In = sum(qlikeGDAXI_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_SPX_rf_In)
y_actualvalues = np.array(y_train_SPX_rf)
qlikeSPX_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeSPX_rf_1_In.append(iteration)
qlikeSPX_rf_1_In = np.array(qlikeSPX_rf_1_In)
QLIKE_rf_1_SPX_In = sum(qlikeSPX_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_HSI_rf_In)
y_actualvalues = np.array(y_train_HSI_rf[20:])
qlikeHSI_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeHSI_rf_1_In.append(iteration)
qlikeHSI_rf_1_In = np.array(qlikeHSI_rf_1_In)
QLIKE_rf_1_HSI_In = sum(qlikeHSI_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_IBEX_rf_In)
y_actualvalues = np.array(y_train_IBEX_rf[20:])
qlikeIBEX_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIBEX_rf_1_In.append(iteration)
qlikeIBEX_rf_1_In = np.array(qlikeIBEX_rf_1_In)
QLIKE_rf_1_IBEX_In = sum(qlikeIBEX_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_IXIC_rf_In)
y_actualvalues = np.array(y_train_IXIC_rf)
qlikeIXIC_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeIXIC_rf_1_In.append(iteration)
qlikeIXIC_rf_1_In = np.array(qlikeIXIC_rf_1_In)
QLIKE_rf_1_IXIC_In = sum(qlikeIXIC_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_N225_rf_In)
y_actualvalues = np.array(y_train_N225_rf[20:])
qlikeN225_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeN225_rf_1_In.append(iteration)
qlikeN225_rf_1_In = np.array(qlikeN225_rf_1_In)
QLIKE_rf_1_N225_In = sum(qlikeN225_rf_1_In)/len(y_actualvalues)

y_forecastvalues = np.array(y_pred_1_OMXC20_rf_In)
y_actualvalues = np.array(y_train_OMXC20_rf[20:])
qlikeOMXC20_rf_1_In = []
for i in range(0, len(y_actualvalues)):
    iteration = y_actualvalues[i]/y_forecastvalues[i] - np.log(y_actualvalues[i]/y_forecastvalues[i]) - 1
    qlikeOMXC20_rf_1_In.append(iteration)
qlikeOMXC20_rf_1_In = np.array(qlikeOMXC20_rf_1_In)
QLIKE_rf_1_OMXC20_In = sum(qlikeOMXC20_rf_1_In)/len(y_actualvalues)

#%% Qlike In

arrays_In = {
    'qlike_rf': {
        'DJI': qlikeDJI_rf_1_In,
        'FTSE': qlikeFTSE_rf_1_In,
        'FTSEMIB': qlikeFTSEMIB_rf_1_In,
        'GDAXI': qlikeGDAXI_rf_1_In,
        'SPX': qlikeSPX_rf_1_In,
        'HSI': qlikeHSI_rf_1_In,
        'IBEX': qlikeIBEX_rf_1_In,
        'IXIC': qlikeIXIC_rf_1_In,
        'N225': qlikeN225_rf_1_In,
        'OMXC20': qlikeOMXC20_rf_1_In
            }
        }

for k1 in arrays_In:
    if k1 == 'qlike_rf':    
        for k2 in arrays_In[k1]:
            nome_file = 'qlike{}_rf_In.csv'.format(k2)
            np.savetxt(nome_file, arrays_In[k1][k2], delimiter=',')
            
#%%

symbols = ['DJI', 'FTSE', 'FTSEMIB', 'GDAXI', 'SPX', 'HSI', 'IBEX', 'IXIC', 'N225', 'OMXC20']
predictions = [y_pred_1_DJI_rf, y_pred_1_FTSE_rf, y_pred_1_FTSEMIB_rf, y_pred_1_GDAXI_rf, y_pred_1_SPX_rf, y_pred_1_HSI_rf, y_pred_1_IBEX_rf, y_pred_1_IXIC_rf, y_pred_1_N225_rf, y_pred_1_OMXC20_rf]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastrf_{symbol}.csv', prediction, delimiter=',')


predictions = [y_pred_1_DJI_rf_In, y_pred_1_FTSE_rf_In, y_pred_1_FTSEMIB_rf_In, y_pred_1_GDAXI_rf_In, y_pred_1_SPX_rf_In, y_pred_1_HSI_rf_In, y_pred_1_IBEX_rf_In, y_pred_1_IXIC_rf_In, y_pred_1_N225_rf_In, y_pred_1_OMXC20_rf_In]

for symbol, prediction in zip(symbols, predictions):
    np.savetxt(f'forecastrf_{symbol}_in.csv', prediction, delimiter=',')


