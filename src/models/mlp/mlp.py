# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# Right now it is a hard copy, not implemented to our data and it includes a simple RNN model as well
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
from sklearn.neural_network import MLPRegressor
from src.models.benchmarks.model import Model
from data.data_handler import get_data
from math import sqrt
import datetime as dt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(42)


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
    #print('ts_init: ' + str(ts_init.shape))
    if window % 2 == 0:
        # ts_ma = pd.rolling_mean(ts_init, window, center=True)
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
        #print('ts_ma after rolling mean(window) ' + str(ts_ma.shape))
        # ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
        ts_ma = pd.Series(ts_init).rolling(2, center=True).mean()
        #print('ts_ma after rolling mean(2) ' + str(ts_ma.shape))
        ts_ma = np.roll(ts_ma, -1)
    else:
        # ts_ma = pd.rolling_mean(ts_init, window, center=True)
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
    #print('ts_ma shape:' + str(ts_ma.shape))
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
    #test = data
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
    print('MLP loss: ' + str(model.loss_))
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


def main():
    fh = 24  # forecasting horizon TODO: Check me out
    freq = 1  # data frequency TODO: check this one
    in_size = 24  # number of points used as input for each forecast

    err_MLP_sMAPE = []
    err_MLP_MASE = []

    # ===== In this example we produce forecasts for 100 randomly generated timeseries =====

    # data_all = np.array(np.random.random_integers(0, 100, (1, 1*7*24 + 2*7*24)), dtype=np.float32)
    ts_length = fh + in_size
    number_of_ts = 1
    data_all = np.array(np.random.randint(0, 100 + 1, size=(number_of_ts, ts_length + 1)), dtype=np.float32)
    for i in range(0, number_of_ts):
        for j in range(0, ts_length):
            data_all[i, j] = j * 10 + data_all[i, j]

    counter = 0
    """
    periods = [['07.01.2018', '14.01.2018'],
               ['15.01.2018', '28.01.2018']]
    data_all = []
    for i in range(0, len(periods)):
        # test periods (contains price now, but wont later just for easy plotting)
        df = get_data(periods[i][0], periods[i][1], ['System Price'], os.getcwd())
        data_all.append(df['System Price'].tolist())
    """
    # ===== Main loop which goes through all timeseries =====
    for j in range(len(data_all)):
        ts = np.asarray(data_all[j])

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
        print('First:')
        print(y_hat_test_MLP)
        for i in range(0, 29):
            y_hat_test_MLP = np.vstack((y_hat_test_MLP, mlp_bench(x_train, y_train, x_test, fh)))
        y_hat_test_MLP = np.median(y_hat_test_MLP, axis=0)
        print('Last:')
        print(y_hat_test_MLP)
        # add trend
        for i in range(0, len(ts)):
            ts[i] = ts[i] + ((a * i) + b)

        for i in range(0, fh):
            y_hat_test_MLP[i] = y_hat_test_MLP[i] + ((a * (len(ts) + i + 1)) + b)

        # add seasonality
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

        # Plot forecasts
        # True alone:
        plt.figure()
        plt.plot(y_test, label='True')
        plt.legend(loc='upper right')
        plt.show()

        # MLP alone
        plt.figure()
        plt.plot(y_hat_test_MLP, label='MLP')
        plt.plot(y_test, label='True')
        plt.legend(loc='upper right')
        plt.show()

        # Calculate errors
        err_MLP_sMAPE.append(smape(y_test, y_hat_test_MLP))
        err_MLP_MASE.append(mase(ts[:-fh], y_test, y_hat_test_MLP, freq))

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
