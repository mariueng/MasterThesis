# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
import math
import random

from sklearn.neural_network import MLPRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from src.models.benchmarks.model import Model
from data.data_handler import get_data
from datetime import timedelta
import src.preprocessing.m4_preprocessing as mp
import os
import copy
import pandas as pd
import numpy as np


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
                         max_iter=1000, learning_rate='adaptive', learning_rate_init=0.001, early_stopping=True)

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
        training_set_size = 24 * 365  # Set size of data the model will be trained on with a moving window
        forecast_from_date = df["Date"].iloc[-1]
        train = get_data(forecast_from_date - timedelta(hours=training_set_size), forecast_from_date, ['System Price'], os.getcwd())
        train.dropna(subset=['System Price'], inplace=True)
        ts = train['System Price'].to_numpy()

        # remove seasonality
        seasonality_in = mp.deseasonalize(ts, freq)
        for i in range(0, len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]

        # de-trending
        a, b = mp.detrend(ts)

        for i in range(0, len(ts)):
            ts[i] = ts[i] - ((a * i) + b)

        x_train, y_train, x_test, _ = mp.split_into_train_test(ts, in_size, fh)
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
        diff = upper_quantile - lower_quantile
        print(diff)
        df['Forecast'] = y_hat_test_MLP
        df['Upper'] = upper_quantile
        df['Lower'] = lower_quantile
        return df
