# script for the most naive model: copy the 24 hours of last day for the following 14 weekdays
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import operator
from data import data_handler
import os
from pathlib import Path
import numpy as np
import random
import math

from numpy.testing._private.parameterized import param
from src.system.scores import calculate_coverage_error
from src.system.generate_periods import get_random_periods
from data.data_handler import get_data
from src.models.naive_day.naive_day import get_prob_forecast
from src.preprocessing.arcsinh import arcsinh
import statsmodels.api as sm


class ExpertModel:
    def __init__(self):
        self.name = "Expert Model"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    @staticmethod
    def forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = get_forecast(forecast_df)
        up = 1.15
        down = 0.90
        forecast_df = get_prob_forecast(forecast_df,  up, down)
        return forecast_df


def get_forecast(forecast_df):
    prev_workdir = os.getcwd()
    os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\expert_model")
    start = forecast_df.loc[0, "Date"].date()
    train_start = start - timedelta(days=7)
    train_end = train_start + timedelta(days=6)
    df = get_data(train_start, train_end, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
    df = adjust_for_holiday(df, "System Price")
    data = get_data(start, start+timedelta(days=13), ["Weekday", "Holiday"], os.getcwd(), "h")
    a, b = get_a_and_b_from_training()
    df["Trans System Price"] = arcsinh.to_arcsinh_from_a_and_b(df["System Price"], a, b)
    past_prices = df["Trans System Price"].tolist()
    weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    forecast = []
    hourly_models = get_hourly_models_as_dict()
    for i in range(len(forecast_df)):
        hour = forecast_df.loc[i, "Hour"]
        model_fit = hourly_models[hour]
        x = {"1 hour lag": past_prices[-1], "2 hour lag": past_prices[-2], "1 day lag": past_prices[-24],
             "2 day lag": past_prices[-48], "1 week lag": past_prices[-168],
             "Max Yesterday": max(past_prices[-24:]), "Min Yesterday": min(past_prices[-24:]),
             "Midnight Yesterday": past_prices[-(hour+1)]}
        for d_id, d in weekdays.items():
            x[d] = 1 if data.loc[i, "Weekday"] == d_id else 0
        x = pd.DataFrame.from_dict(x, orient="index").transpose()
        x = x.assign(intercept=1)
        prediction = model_fit.get_prediction(x)
        forecasted_value = prediction.predicted_mean[0]
        forecast.append(forecasted_value)
        past_prices.append(forecasted_value)
    forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
    data["Forecast"] = forecast
    data = adjust_for_holiday(data, "Forecast")
    forecast_df["Forecast"] = data["Forecast"]
    os.chdir(prev_workdir)
    return forecast_df


def get_hourly_models_as_dict():
    hourly_models = {}
    for i in range(24):
        hourly_models[i] = sm.load(os.getcwd() + '\\h_models\\expert_model_{}.pickle'.format(i))
    return hourly_models


def get_a_and_b_from_training():
    with open(str(os.getcwd()) + '\\preproc_parameters.txt', 'r') as fh:
        string_list = fh.readline().split(",")
        a = float(string_list[0].split(":")[1])
        b = float(string_list[1].split(":")[1])
    return a, b


def adjust_for_holiday(df, col):
    operation = operator.mul if col == "System Price" else operator.truediv
    df[col] = df.apply(lambda row: operation(row[col], get_hol_factor()) if
    row["Holiday"] == 1 and row["Weekday"] != 7 else row[col], axis=1)
    return df.drop(columns=["Holiday"])


def get_hol_factor():
    f = 1.1158
    return f


def train_model(dir_path):
    start_date = dt(2014, 7, 1)
    #start_date = dt(2018, 1, 1)
    end_date = dt(2019, 6, 2)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday"], os.getcwd(), "h")
    training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
    with open(dir_path + '\\preproc_parameters.txt', 'w') as fh:
        fh.write("a:{},b:{}".format(a, b))
        fh.close()
    assert False
    col = "Trans System Price"
    training_data["1 hour lag"] = training_data[col].shift(1)
    training_data["2 hour lag"] = training_data[col].shift(2)
    training_data["1 day lag"] = training_data[col].shift(24)
    training_data["2 day lag"] = training_data[col].shift(48)
    training_data["1 week lag"] = training_data[col].shift(168)
    days_in_year = pd.date_range(start_date + timedelta(days=1), end_date, freq='d')
    for day in days_in_year:
        yesterday_df = training_data[training_data["Date"] == day - timedelta(days=1)]
        max_yesterday = max(yesterday_df[col])
        min_yesterday = min(yesterday_df[col])
        midnight_price = yesterday_df.tail(1)[col].values[0]
        todays_df = training_data[training_data["Date"] == day]
        for index in todays_df.index:
            training_data.loc[index, "Max Yesterday"] = max_yesterday
            training_data.loc[index, "Min Yesterday"] = min_yesterday
            training_data.loc[index, "Midnight Yesterday"] = midnight_price
    training_data = training_data.dropna()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for index, row in training_data.iterrows():
        for j in range(1, 8):
            day = days[j - 1]
            if training_data.loc[index, "Weekday"] == j:
                training_data.loc[index, day] = 1
            else:
                training_data.loc[index, day] = 0
    drop_cols = ["Date", "Hour", "System Price", "Trans System Price", "Weekday"]
    x_columns = [col for col in training_data.columns if col not in drop_cols]
    for i in range(24):
        h_data = training_data.loc[training_data["Hour"] == i]
        y = h_data[[col]]
        x = h_data[x_columns]
        x = x.assign(intercept=1)
        model = sm.OLS(y, x, hasconst=True).fit()
        print_model = model.summary()
        with open(dir_path + '\\h_fits\\expert_fit_{}.txt'.format(i), 'w') as fh:
            fh.write(print_model.as_text())
            fh.close()
        model.save(dir_path + "\\h_models\\expert_model_{}.pickle".format(i))


def run(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_list.append(result_df)
    print(result_list)


if __name__ == '__main__':
    # train_model(str(Path(__file__).parent))
    model_ = ExpertModel()
    periods_ = get_random_periods(1)
    run(model_, periods_)
