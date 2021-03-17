# script for the most naive model: copy the 24 hours of last day for the following 14 weekdays
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
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

    def forecast(self, forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast = self.get_forecast(forecast_df)
        forecast_df["Forecast"] = forecast
        up = 1.15
        down = 0.90
        forecast_df = get_prob_forecast(forecast_df,  up, down)
        return forecast_df

    def get_forecast(self, forecast_df):
        model_fit = self.get_model()
        s_date = forecast_df.at[0, "Date"]
        train_start = s_date - timedelta(days=7)
        train_end = train_start + timedelta(days=6)
        df = get_data(train_start, train_end, ["System Price", "Weekday", "Month"], os.getcwd(), "h")
        df_test_time_dummies_df = get_data(s_date, s_date+timedelta(days=13), ["Weekday", "Month"], os.getcwd(), "h")
        df, a, b = arcsinh.to_arcsinh(df, "System Price")
        past_prices = df["Trans System Price"].tolist()
        weekdays = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        months = ["m{}".format(m) for m in range(1, 13)]
        forecast = []
        for i in range(len(forecast_df)):
            x = {"1 hour lag": past_prices[-1], "2 hour lag": past_prices[-2], "1 day lag": past_prices[-24],
                 "2 day lag": past_prices[-48], "1 week lag": past_prices[-168],
                 "Max Yesterday": max(past_prices[-24:]), "Min Yesterday": min(past_prices[-24:])}
            for val in weekdays.values():
                x[val] = 0
            day = df_test_time_dummies_df.loc[i, "Weekday"]
            x[weekdays[day]] = 1
            for m in months:
                x[m] = 0
            current_month = df_test_time_dummies_df.loc[i, "Month"]
            x["m{}".format(current_month)] = 1
            x = pd.DataFrame.from_dict(x, orient="index").transpose()
            prediction = model_fit.get_prediction(exog=x)
            forecasted_value = prediction.predicted_mean[0]
            forecast.append(forecasted_value)
            past_prices.append(forecasted_value)
        forecast = arcsinh.from_arcsin_to_original(forecast, a, b)
        return forecast

    @staticmethod
    def get_model():
        dir_path = str(Path(__file__).parent)
        m_path = dir_path + "\\expert_model.pickle"
        model_exists = os.path.isfile(m_path)
        if model_exists:
            model = sm.load(m_path)
        else:
            train_model(dir_path)
            model = sm.load(m_path)
        return model


def train_model(dir_path):
    start_date = dt(2017, 1, 1)
    end_date = dt(2019, 12, 31)
    training_data = get_data(start_date, end_date, ["System Price", "Weekday"], os.getcwd(), "h")
    training_data, a, b = arcsinh.to_arcsinh(training_data, "System Price")
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
        todays_df = training_data[training_data["Date"] == day]
        for index in todays_df.index:
            training_data.loc[index, "Max Yesterday"] = max_yesterday
            training_data.loc[index, "Min Yesterday"] = min_yesterday
    training_data = training_data.dropna()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for index, row in training_data.iterrows():
        for j in range(1, 8):
            day = days[j - 1]
            if training_data.loc[index, "Weekday"] == j:
                training_data.loc[index, day] = 1
            else:
                training_data.loc[index, day] = 0
    y = training_data[[col]]
    drop_cols = ["Date", "Hour", "System Price", "Trans System Price", "Weekday"]
    x_columns = [col for col in training_data.columns if col not in drop_cols]
    x = training_data[x_columns]
    model = sm.OLS(y, x).fit()
    print_model = model.summary()
    with open(dir_path + '\\expert_fit.txt', 'w') as fh:
        fh.write(print_model.as_text())
    model.save(dir_path + "\\expert_model.pickle")


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
    train_model(str(Path(__file__).parent))
    #model_ = ExpertModel()
    #periods_ = get_random_periods(1)
    #run(model_, periods_)
