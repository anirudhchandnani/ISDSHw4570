# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:00:19 2023

@author: anirudh
"""
#############################adding auto-regression in hw4v4
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import scale
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from fredapi import Fred
from xgboost import XGBRegressor

# Function to retrieve data from FRED and convert it to DataFrame
def fredr_df(series_id, start_date):
    fred = Fred(api_key='baa35ec4364451630365ce6deef315cf')
    data = fred.get_series(series_id, start_date=start_date, frequency='q')
    df = pd.DataFrame(data, columns=[series_id])
    return df,data

###pce, unrate, indpro
#### lag to 2 

# Load data for series (forecast series first)
series_ids = ['GDPC1','INDPRO','PCE','UNRATE','CORESTICKM159SFRBATL']
# series_ids = ['GDPC1','INDPRO','PCE','UNRATE','CARQGSP','CORESTICKM159SFRBATL'] ## 0.6606, MASE: 0.6576667181590172 with lagging two all indicators and 1 level of auto regression 
# series_ids = ['GDPC1','INDPRO','PCE','UNRATE','GDPPOT','CORESTICKM159SFRBATL','CARQGSP'] ## 0.6606
# series_ids = ['GDPC1','INDPRO','PCE','UNRATE','HSN1F','CORESTICKM159SFRBATL','CARQGSP']
# series_ids = ['GDPC1' ,'UNRATE', 'CPIAUCSL']

start_date = '1978-07-01'

time_series_df = None
for series in series_ids:
    fred_df ,fred_df1 = fredr_df(series, start_date)
    time_series_df = pd.concat([time_series_df, fred_df], axis=1)

# time_series_df['CARQGSP'] = time_series_df['CARQGSP'].fillna(time_series_df['CARQGSP'].mode()[0])
#time_series_df = time_series_df.reset_index()    

#time_series_df = pd.read_csv('timeseriesdf.csv')
#time_series_df = time_series_df.set_index('index')
#time_series_df = time_series_df_subset
cutoff = '1978-07-01' ####taking 2 extra rows as lagging by 2 so less nulls
recent  = '2023-10-01'

#time_series_df.head()
time_series_df = time_series_df[time_series_df.index>= cutoff ]
time_series_df = time_series_df[time_series_df.index<= recent ]

time_series_df .head()
time_series_df .tail()
# Function to calculate Mean Absolute Scaled Error (MASE)
def get_MASE(y, yhat):
    comp = pd.concat([y, yhat], axis=1, join='inner')
    comp['AE'] = np.abs(comp.iloc[:, 0] - comp.iloc[:, 1])
    sum_abs_diff_y = np.sum(np.abs(np.diff(y.dropna())))
    TT = len(comp)
    MASE = np.sum(comp['AE']) / ((TT / (TT - 1)) * sum_abs_diff_y)
    return MASE




def one_step_GDP_growth_forecast(time_series_df, seed=1):


    differenced = np.diff(np.log(time_series_df), axis=0)
    y = differenced[:, 0]
    x = np.roll(differenced[:, 1:], shift=1, axis=0)
    x_names = [f"{col}.lag1" for col in range(x.shape[1])]
    x = pd.DataFrame(x, columns=x_names)
    # x2 = np.roll(differenced[:, 1:], shift=2, axis=0)
    # x2_names = [f"{col}.lag2" for col in range(x2.shape[1])]
    # x2 = pd.DataFrame(x2, columns=x2_names)
    # x3 = np.roll(differenced[:, 1:], shift=3, axis=0)
    # x3_names = [f"{col}.lag3" for col in range(x3.shape[1])]
    # x3 = pd.DataFrame(x3, columns=x3_names)
    # transformed = pd.concat([pd.Series(y, name="y"), x,x2,x3], axis=1)
    transformed = pd.concat([pd.Series(y, name="y"), x], axis=1)
    transformed['y_shifted'] = transformed.iloc[:, 0].shift()
        
    # transformed['y_shifted2'] = transformed.iloc[:, 0].shift(2)
    x_new = transformed.tail(1).iloc[:, 1:]
    transformed = transformed.dropna()

    x_matrix = transformed.iloc[:, 1:].values
    y_vector = transformed.iloc[:, 0].values

    x_train = x_matrix
    y_train = y_vector

    # Scale (normalize) data
    x_train_scaled = scale(x_train)
    y_train_scaled = scale(y_train)

    x_new_scaled = (x_new - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    
    # x_new_scaled1 = (x_new - np.mean(x_train_scaled, axis=0)) / np.std(x_train_scaled, axis=0)

    
    seed = 1
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Model
    hidden_units_1 = 5
    hidden_units_2 = 3
    model = Sequential([
        Dense(hidden_units_1, input_shape=(x_train.shape[1],), activation="relu"),
        Dense(hidden_units_1, input_shape=(x_train.shape[1],), activation="relu"),
        Dense(hidden_units_1, input_shape=(x_train.shape[1],), activation="relu"),
        Dropout(0.2),
        # Dense(hidden_units_1, input_shape=(x_train.shape[1],), activation="relu"),
        # Dense(hidden_units_2, activation="sigmoid"),
        # Dense(hidden_units_2, activation="sigmoid"),
        Dense(hidden_units_2, activation="sigmoid"),
        Dropout(0.2),
        Dense(1)
    ])

    # Regression-specific compilation
    model.compile(
        loss="mse",
        optimizer=RMSprop(),
        metrics=["mean_absolute_error", "mse"]
    )

    verbose = 1
    validation_split = 0.2
    epochs = 100
    batch_size = 32
    patience = 10

    callbacks = None
    if not np.isnan(patience):
        callbacks = [EarlyStopping(patience=patience, mode="auto")]

    fit = model.fit(
        x=x_train_scaled,
        y=y_train_scaled,
        shuffle=True,
        verbose=verbose,
        validation_split=validation_split,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size
    )

    predictions = model.predict(x_new_scaled)
    predictions = predictions * np.std(y_train) + np.mean(y_train)
    return predictions[0][0],y
    


cutoff = '1978-07-01' ####taking 2 extra rows as lagging by 2 so less nulls
recent  = '2023-10-01'
backtest_start = '2007-01-01'
gdp_forecast_backtest = pd.DataFrame()
gdp_forecast_backtest = pd.DataFrame()
predicted_list = []
##time_series_df.head()
##time_series_df = time_series_df[time_series_df.index>= cutoff ]
##time_series_df = time_series_df[time_series_df.index<= recent ]

##time_series_df.isna().sum()
##threshold = 5
##time_series_df = time_series_df.dropna(axis=1, thresh=len(time_series_df) - threshold + 1)
backtest_quarters = time_series_df.index
backtest_quarters = backtest_quarters[(backtest_quarters >= backtest_start)&(backtest_quarters <= recent)]

q =  '2007-01-01'
for q in backtest_quarters:
    print('============================================================','Backtesting', q, '============================================================')
    # Subset
    time_series_df_subset = time_series_df[time_series_df.index <= q]
    #Forecast
    one_step_fcst,y = one_step_GDP_growth_forecast(time_series_df=time_series_df_subset)
    # Append
    #gdp_forecast_backtest = pd.concat([gdp_forecast_backtest, pd.Series([one_step_fcst], index=[q])])
    predicted_list.append(one_step_fcst)
# Attach actuals
# np.array(y).flatten().shape
datazip = list(zip(np.array(y).flatten()[-68:], predicted_list))
df = pd.DataFrame(datazip, columns=['actual', 'predicted'])


df.index = time_series_df.index[-68:]

# Plot and report MASE
plt.plot(df.index, df['actual'], label='Actual GDP Growth')
plt.plot(df.index, df['predicted'], label='predicted GDP Growth')
plt.legend(loc='upper left')
plt.show()

# MASE
MASE = get_MASE( df['actual'] ,df['predicted'])
print('MASE:', MASE)
#MASE: 0.6868105907660272 without auto-regression
#MASE: 0.6800446061041617 with auto-regression relu and sigmoid
#MASE: 0.6912325300003485 with auto-regression sigmoid and sigmoid
#MASE: 0.7122362211320781 with auto-regression relu and relu
#MASE: 0.6967508669682251 with auto-regression softmax and sigmoid
#MASE: 0.6932866240977438 with auto-regression softmax and softmax

#MASE: 0.6838819321741822 with 2 levels of auto-regression relu and sigmoid

#MASE: MASE: 0.6818307405594899 with 2 levels auto-regression and 2 levels of lagging all series relu and sigmoid

#############BEST MASE = 0.656590277607401