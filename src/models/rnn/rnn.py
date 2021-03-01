# Code retrieved from: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
from src.preprocessing.m4_preprocessing import detrend, deseasonalize, moving_averages, seasonality_test, acf, split_into_train_test
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from src.models.benchmarks.model import Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as ker
from data.data_handler import get_data
from datetime import timedelta
import gc
import os
import copy
import pandas as pd
import numpy as np
import random

random.seed(42)


# @ignore_warnings(category=ConvergenceWarning)
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
    model = Sequential([SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
                                  use_bias=False, kernel_initializer='glorot_uniform',
                                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                                  dropout=0.0, recurrent_dropout=0.0), Dense(1, use_bias=True, activation='linear')
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


class RNN(Model):

    def forecast(self, df: pd.DataFrame):  # df: ["Date", "Hour", "Forecast", "Upper", "Lower"]
        # TODO: should really be datetime column unless time-variables are used as features
        fh = len(df['Forecast'].values)  # Forecasting horizon
        freq = 24  # Data frequency
        in_size = fh * 2  # Input size for fh
        ensemble = True  # Ensemble MLP used for generating PIs
        # Get data based on test_period
        training_set_size = 2000  # Set size of data the model will be trained on with a moving window
        forecast_from_date = df["Date"].iloc[-1]
        train = get_data(forecast_from_date - timedelta(hours=training_set_size), forecast_from_date, ['System Price'],
                         os.getcwd())
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

        # # Check if fitted
        # try:
        #     check_is_fitted(self.model)
        # except NotFittedError as e:
        #     self.train(df)

        y_hat_test_RNN = np.reshape(rnn_bench(x_train, y_train, copy.deepcopy(x_test), fh, in_size), (-1))

        if ensemble:
            # NB! Without deepcopy x_test is modified for each run and the models explode!
            for i in range(0, 29):
                y_hat_test_RNN = np.vstack(
                    (y_hat_test_RNN, np.reshape(rnn_bench(x_train, y_train, copy.deepcopy(x_test), fh, in_size), (-1))))
            y_hat_pred_int = np.std(y_hat_test_RNN,
                                    axis=0)  # TODO: Right now only contains model uncertainty, data has uncertainty as well
            y_hat_test_RNN = np.mean(y_hat_test_RNN,
                                     axis=0)  # TODO: Check whether these should be calculated from entire dataset

        # add trend
        for i in range(0, len(ts)):
            ts[i] = ts[i] + ((a * i) + b)

        for i in range(0, fh):
            y_hat_test_RNN[i] = y_hat_test_RNN[i] + ((a * (len(ts) + i + 1)) + b)

        # add seasonality
        for i in range(0, len(ts)):
            ts[i] = ts[i] * seasonality_in[i % freq] / 100

        for i in range(len(ts), len(ts) + fh):
            y_hat_test_RNN[i - len(ts)] = y_hat_test_RNN[i - len(ts)] * seasonality_in[i % freq] / 100

        # check if negative or extreme
        for i in range(len(y_hat_test_RNN)):
            if y_hat_test_RNN[i] < 0:
                y_hat_test_RNN[i] = 0

            if y_hat_test_RNN[i] > (1000 * max(ts)):
                y_hat_test_RNN[i] = max(ts)

        """
        Prediction Interval:
        - 75% : z=1.15
        - 90% : z=1.64
        - 95% : z=1.96
        - 99% : z=2.58
        NB! Assumes model and data generated by normal distribution and that mean and variance is known
        Right now model predictions are used to estimate the mean and variance.
        """
        z = 1.96
        upper = np.add(y_hat_test_RNN, y_hat_pred_int * z)
        lower = np.subtract(y_hat_test_RNN, y_hat_pred_int * z)

        df['Forecast'] = y_hat_test_RNN
        df['Upper'] = upper
        df['Lower'] = lower

        # memory handling
        ker.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        return df
