<<<<<<<< HEAD:src/models/mlp/mlp.py
# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# Right now it is a hard copy, not implemented to our data and it includes a simple RNN model as well
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from src.models.benchmarks.model import Model
from data.data_handler import get_data
from math import sqrt
import datetime as dt
import os
import pandas as pd
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
#tf.random.set_seed(42)
from random import seed
seed(42)


def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    if window % 2 == 0:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
        ts_ma = pd.Series(ts_init).rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    # Note that the statistical benchmarks, implemented in R, use the same seasonality test, but with ACF1 being
    # squared This difference between the two scripts was mentioned after the end of the competition and, therefore,
    # no changes have been made to the existing code so that the results of the original submissions are reproducible
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


def split_into_train_test(data, in_num, fh):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs
    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
    train, test = data[:-fh], data[-(fh + in_num):]

    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test, y_test


@ignore_warnings(category=ConvergenceWarning)
def mlp_bench(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer
    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []

    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)

    model.fit(x_train, y_train)

    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)


class MLP(Model):

    def train(self, start_test_period):
        model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                             max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                             random_state=42)
        # Get data based on test_period
        train = get_data(start_test_period - timedelta(days=14), start_test_period, ['System Price'], os.getcwd())
        x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
        self.model = model.fit(x_train, y_train)
        return self.model

    def forecast(self, df: pd.DataFrame):  # df: ["Date", "Hour", "Forecast", "Upper", "Lower"]
        # Check if fitted here ...
        index_array = df.index
        y_hat_test = []
        last_prediction = self.model.predict(index_array.shape[0])[0]
        for i in range(0, len(index_array)):
            y_hat_test.append(last_prediction)
            x_test[0] = np.roll(x_test[0], -1)
            x_test[0, (len(x_test[0]) - 1)] = last_prediction
            last_prediction = self.model.predict(x_test)[0]

        return np.asarray(y_hat_test)


def smape(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

import sys
np.set_printoptions(threshold=sys.maxsize)
def main():
    fh = 2  # forecasting horizon
    freq = 1  # data frequency TODO: check this one
    in_size = 3  # number of points used as input for each forecast
    err_MLP_sMAPE = []
    err_MLP_MASE = []

    # ===== In this example we produce forecasts for 100 randomly generated timeseries =====

    # data_all = np.array(np.random.random_integers(0, 100, (1, 1*7*24 + 2*7*24)), dtype=np.float32)
    # number_of_ts = 1
    # ts_length = (fh + in_size) + filler
    # data_all = np.array(np.random.randint(0, 100 + 1, size=(number_of_ts, ts_length)), dtype=np.float32)
    # for i in range(0, number_of_ts):
    #     for j in range(0, ts_length):
    #         data_all[i, j] = j * 10 + data_all[i, j]
    """
    ['01.01.2018', '28.02.2018'],
               ['01.03.2018', '28.04.2018'],
               ['01.05.2018', '28.06.2018'],
               ['01.07.2018', '28.08.2018']
    """
    counter = 0
    periods = [['01.02.2018', '31.03.2018']]
    data_all = np.zeros(shape=(len(periods), 1416))
    for i in range(0, len(periods)):
        df = get_data(periods[i][0], periods[i][1], ['System Price'], os.getcwd())
        print(df['System Price'].isnull().values.any())
        df['Date'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
        df.drop(['Hour'], axis=1, inplace=True)
        data_all[i] = df['System Price'].to_numpy()

    # ===== Main loop which goes through all timeseries =====
    for j in range(len(data_all)):
        ts = data_all[j]
        plt.figure()
        plt.plot(np.linspace(start=0, stop=len(ts), num=1416), ts)
        plt.show()

        # remove seasonality
        seasonality_in = deseasonalize(ts, freq)
        for i in range(0, len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]

        # de-trending
        a, b = detrend(ts)

        for i in range(0, len(ts)):
            ts[i] = ts[i] - ((a * i) + b)
        """
        # Get data to train model on, based on test period (?) TODO: Is this correct?
        start_test_period = dt.datetime.strptime(periods[j][0], '%d.%m.%Y')
        train = get_data(start_test_period - dt.timedelta(days=14), start_test_period, ['System Price'], os.getcwd())
        """
        x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)

        # MLP benchmark - Produce forecasts
        y_hat_test_MLP = mlp_bench(x_train, y_train, x_test, fh)

        # for i in range(0, 29):
        #     y_hat_test_MLP = np.vstack((y_hat_test_MLP, mlp_bench(x_train, y_train, x_test, fh)))
        # y_pred_int = []
        # y_hat_test_MLP = np.median(y_hat_test_MLP, axis=0)

        # add trend
        for i in range(0, len(ts)):
            ts[i] = ts[i] + ((a * i) + b)

        for i in range(0, fh):
            y_hat_test_MLP[i] = y_hat_test_MLP[i] + ((a * (len(ts) + i + 1)) + b)

        # add seasonality, does nothing when no seasonality
        for i in range(0, len(ts)):
            ts[i] = ts[i] * seasonality_in[i % freq] / 100

        for i in range(len(ts), len(ts) + fh):
            y_hat_test_MLP[i - len(ts)] = y_hat_test_MLP[i - len(ts)] * seasonality_in[i % freq] / 100

        # check if negative or extreme
        for i in range(len(y_hat_test_MLP)):
            if y_hat_test_MLP[i] < 0:
                y_hat_test_MLP[i] = 0

            if y_hat_test_MLP[i] > (1000 * max(ts)):
                y_hat_test_MLP[i] = max(ts)

        x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)

        # Calculate errors
        err_MLP_sMAPE.append(smape(y_test, y_hat_test_MLP))
        err_MLP_MASE.append(mase(ts[:-fh], y_test, y_hat_test_MLP, freq))

        # With training period
        plt.figure()
        x_axis = np.linspace(start=0, stop=len(ts), num=len(ts))
        plt.plot(x_axis[-fh*2:], ts[-fh*2:], label='True')
        plt.plot(x_axis[-len(y_test):], y_hat_test_MLP, label='MLP')
        plt.plot([len(ts) - fh, len(ts) - fh], [0, 100], linestyle='--')
        plt.title(f'Period: {periods[counter][0]} - {periods[counter][1]}    sMAPE: {round(err_MLP_sMAPE[counter], 3)}  MAsE: {round(err_MLP_MASE[counter], 3)}')
        plt.legend(loc='upper left')
        plt.show()

        # MLP and true test
        plt.figure()
        plt.plot(x_axis[-len(y_test):], y_test, label='True')
        plt.plot(x_axis[-len(y_test):], y_hat_test_MLP, label='MLP')
        plt.title(f'Period: {periods[counter][0]} - {periods[counter][1]}    sMAPE: {round(err_MLP_sMAPE[counter], 3)}  MAsE: {round(err_MLP_MASE[counter], 3)}')
        plt.legend(loc='upper right')
        plt.show()

        counter = counter + 1
        print("-------------TS ID: ", counter, "-------------")

    print("\n\n---------FINAL RESULTS---------")
    print("=============sMAPE=============\n")
    print("#### MLP ####\n", np.mean(err_MLP_sMAPE), "\n")
    print("==============MASE=============")
    print("#### MLP ####\n", np.mean(err_MLP_MASE), "\n")


# main()


if __name__ == '__main__':
    #model_ = MLP('MLP')
    #model_.train('15.01.2018')
    main()
========
# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# Right now it is a hard copy, not implemented to our data and it includes a simple RNN model as well
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy

from tensorflow import random
random.set_seed(42)
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as ker
from math import sqrt
import numpy as np
import tensorflow as tf
import pandas as pd
import gc


def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    if window % 2 == 0:
        #ts_ma = pd.rolling_mean(ts_init, window, center=True)
        ts_ma = np.convolve(ts_init, np.ones(window), 'valid') / window
        #ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
        ts_ma = np.convolve(ts_ma, np.ones(2), 'valid') / 2
        ts_ma = np.roll(ts_ma, -1)
    else:
        #ts_ma = pd.rolling_mean(ts_init, window, center=True)
        ts_ma = np.convolve(ts_init, np.ones(window), 'valid') / window

    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    
    # Note that the statistical benchmarks, implemented in R, use the same seasonality test, but with ACF1 being squared
    # This difference between the two scripts was mentioned after the end of the competition and, therefore, no changes have been made 
    # to the existing code so that the results of the original submissions are reproducible
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


def split_into_train_test(data, in_num, fh):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs
    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
    train, test = data[:-fh], data[-(fh + in_num):]
    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test, y_test


def rnn_bench(x_train, y_train, x_test, fh, input_size):
    """
    Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer
    :param x_train: train data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :param input_size: number of points used as input
    :return:
    """
    # reshape to match expected input
    x_train = np.reshape(x_train, (-1, input_size, 1))
    x_test = np.reshape(x_test, (-1, input_size, 1))

    # create the model
    model = Sequential([
        SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
                  use_bias=False, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                  dropout=0.0, recurrent_dropout=0.0),
        Dense(1, use_bias=True, activation='linear')
    ])
    opt = RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # fit the model to the training data
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

    # make predictions
    y_hat_test = []
    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)


def mlp_bench(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer
    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []

    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    model.fit(x_train, y_train)

    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)


def smape(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep


def main():
    fh = 6         # forecasting horizon
    freq = 1       # data frequency
    in_size = 3    # number of points used as input for each forecast

    err_MLP_sMAPE = []
    err_MLP_MASE = []
    err_RNN_sMAPE = []
    err_RNN_MASE = []

    # ===== In this example we produce forecasts for 100 randomly generated timeseries =====
    data_all = np.array(np.random.random_integers(0, 100, (100, 20)), dtype=np.float32)
    for i in range(0, 100):
        for j in range(0, 20):
            data_all[i, j] = j * 10 + data_all[i, j]

    counter = 0
    # ===== Main loop which goes through all timeseries =====
    for j in range(len(data_all)):
        ts = data_all[j, :]

        # remove seasonality
        seasonality_in = deseasonalize(ts, freq)

        for i in range(0, len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]

        # detrending
        a, b = detrend(ts)

        for i in range(0, len(ts)):
            ts[i] = ts[i] - ((a * i) + b)

        x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)

        # RNN benchmark - Produce forecasts
        y_hat_test_RNN = np.reshape(rnn_bench(x_train, y_train, x_test, fh, in_size), (-1))

        # MLP benchmark - Produce forecasts
        y_hat_test_MLP = mlp_bench(x_train, y_train, x_test, fh)
        for i in range(0, 29):
            y_hat_test_MLP = np.vstack((y_hat_test_MLP, mlp_bench(x_train, y_train, x_test, fh)))
        y_hat_test_MLP = np.median(y_hat_test_MLP, axis=0)

        # add trend
        for i in range(0, len(ts)):
            ts[i] = ts[i] + ((a * i) + b)

        for i in range(0, fh):
            y_hat_test_MLP[i] = y_hat_test_MLP[i] + ((a * (len(ts) + i + 1)) + b)
            y_hat_test_RNN[i] = y_hat_test_RNN[i] + ((a * (len(ts) + i + 1)) + b)

        # add seasonality
        for i in range(0, len(ts)):
            ts[i] = ts[i] * seasonality_in[i % freq] / 100

        for i in range(len(ts), len(ts) + fh):
            y_hat_test_MLP[i - len(ts)] = y_hat_test_MLP[i - len(ts)] * seasonality_in[i % freq] / 100
            y_hat_test_RNN[i - len(ts)] = y_hat_test_RNN[i - len(ts)] * seasonality_in[i % freq] / 100

        # check if negative or extreme
        for i in range(len(y_hat_test_MLP)):
            if y_hat_test_MLP[i] < 0:
                y_hat_test_MLP[i] = 0
            if y_hat_test_RNN[i] < 0:
                y_hat_test_RNN[i] = 0
                
            if y_hat_test_MLP[i] > (1000 * max(ts)):
                y_hat_test_MLP[i] = max(ts)         
            if y_hat_test_RNN[i] > (1000 * max(ts)):
                y_hat_test_RNN[i] = max(ts)

        x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)

        # Calculate errors
        err_MLP_sMAPE.append(smape(y_test, y_hat_test_MLP))
        err_RNN_sMAPE.append(smape(y_test, y_hat_test_RNN))
        err_MLP_MASE.append(mase(ts[:-fh], y_test, y_hat_test_MLP, freq))
        err_RNN_MASE.append(mase(ts[:-fh], y_test, y_hat_test_RNN, freq))

        # memory handling
        ker.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        counter = counter + 1
        print("-------------TS ID: ", counter, "-------------")

    print("\n\n---------FINAL RESULTS---------")
    print("=============sMAPE=============\n")
    print("#### MLP ####\n", np.mean(err_MLP_sMAPE), "\n")
    print("#### RNN ####\n", np.mean(err_RNN_sMAPE), "\n")
    print("==============MASE=============")
    print("#### MLP ####\n", np.mean(err_MLP_MASE), "\n")
    print("#### RNN ####\n", np.mean(err_RNN_MASE), "\n")


main()
>>>>>>>> 299d6bb64c3cb80298d9e1e3b06c3779f85d222c:src/models/benchmarks/mlp.py
