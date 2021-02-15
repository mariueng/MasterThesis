import os
from src.models.model_handler import *
from data import data_formatter, data_handler
from src.system.metrics import evaluate_point_and_interval_forecast
import pmdarima as pm
import numpy as np
import pandas as pd


# Get data
def load_data(start_time, end_time):
    df = data_handler.get_data(start_time, end_time, ['System Price'], os.getcwd())
    df = data_formatter.combine_hour_day_month_year_to_datetime_index(df)
    return df


# Fit new model to new data
def fit(train, verbose=False):
    arima_model = pm.auto_arima(train)
    if verbose:
        print(arima_model.summary())
    return arima_model


# Use model to predict
def predict(arima_, test):
    forecast, conf_int = arima_.predict(test.shape[0], return_conf_int=True)
    # Combine results into one array
    results = np.column_stack((forecast, conf_int))
    # Add datetime period as index for forecast data
    dataframe = pd.DataFrame(results,
                             index=test.index)
    dataframe.columns = ['System Price', 'Lower bound', 'Upper bound']
    return dataframe


def score(forecast, actual, train, nominal_coverage=0.95):
    return evaluate_point_and_interval_forecast(forecast, actual, train, nominal_coverage)


# Save model to file
def save_model(id_, name, model, forecast, scores, folder):
    save_pickle(id_, name, model, forecast, scores, folder)


if __name__ == '__main__':
    """
    start_time_ = "01.01.2018"
    end_time_ = "31.01.2018"
    data_ = load_data(start_time_, end_time_)
    split_ = 150
    train_, test_ = data_[:split_], data_[split_:]
    model_ = fit(train_)
    forecast_ = predict(model_, test_)
    scores = score(forecast_, test_, train_)
    print(scores)
    """
