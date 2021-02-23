# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
import math
import random

from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from src.models.benchmarks.model import Model
from data.data_handler import get_data
from math import sqrt
from datetime import timedelta
import os
import copy
import pandas as pd
import numpy as np


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


#@ignore_warnings(category=ConvergenceWarning)
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
                         max_iter=1000, learning_rate='adaptive', learning_rate_init=0.001)

    model.fit(x_train, y_train)

    return model, forecast(model, x_test, fh)


def forecast(model, x_test, fh):
    y_hat_test = []
    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)


class MLP(Model):

    def __init__(self, name):
        self.residuals = None
        Model.__init__(self, name)

    def forecast(self, df: pd.DataFrame):  # df: ["Date", "Hour", "Forecast", "Upper", "Lower"]
        # TODO: should really be datetime column unless time-variables are used as features
        fh = len(df['Forecast'].values)  # Forecasting horizon
        freq = 24  # Data frequency
        in_size = fh  # Input size for fh
        ensemble = True  # Ensemble MLP used for generating PIs
        # Get data based on test_period
        training_set_size = 2000  # Set size of data the model will be trained on with a moving window
        forecast_from_date = df["Date"].iloc[-1]
        train = get_data(forecast_from_date - timedelta(hours=training_set_size), forecast_from_date, ['System Price'], os.getcwd())
        train.dropna(subset=['System Price'], inplace=True)
        ts = train['System Price'].to_numpy()

        # remove seasonality
        seasonality_in = deseasonalize(ts, freq)
        for i in range(0, len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]

        # de-trending
        a, b = detrend(ts)

        for i in range(0, len(ts)):
            ts[i] = ts[i] - ((a * i) + b)

        x_train, y_train, x_test, _ = split_into_train_test(ts, in_size, fh)
        if self.model is None:
            model, y_hat_test_MLP = mlp_bench(x_train, y_train, copy.deepcopy(x_test), fh)
            self.model = model
        else:
            y_hat_test_MLP = forecast(self.model, copy.deepcopy(x_test), fh)

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

        """Bootstrap prediction intervals using in_sample residual errors"""
        # If not performed, calculate all residuals from in-sample dataset by forecasting all periods
        if self.residuals is None:
            in_sample_residuals = []
            for i in range(0, np.shape(x_train)[0] - fh):
                y_target = y_train[i:(i + fh)]
                x_response = x_train[i, :].reshape(1, -1)
                y_forecast = forecast(self.model, x_response, fh)
                residuals = np.subtract(y_target, y_forecast).tolist()
                in_sample_residuals.extend(residuals)
            self.residuals = in_sample_residuals

        # Simulate possible future values for all forecasted values
        simulations = [0] * 100
        for estimate in y_hat_test_MLP:
            residual_sample = np.array(random.choices(self.residuals, k=100))  # K is the number of simulations per estimate
            estimate_simulations = residual_sample + estimate
            simulations = np.vstack((simulations, estimate_simulations))
        simulations = np.delete(simulations, [0], axis=0)
        # Unnecessary?: simulations = np.transpose(simulations)
        lower_quantile = np.quantile(simulations, 0.025, axis=1)
        upper_quantile = np.quantile(simulations, 0.975, axis=1)

        df['Forecast'] = y_hat_test_MLP
        df['Upper'] = upper_quantile
        df['Lower'] = lower_quantile
        return df
