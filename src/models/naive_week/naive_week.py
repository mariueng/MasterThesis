# script for the most naive model: copy the 24 hours of last day for the following 14 weekdays
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from data import data_handler
import os
import numpy as np
import random
from src.system.scores import calculate_coverage_error
from src.system.generate_periods import get_random_periods
from data.data_handler import get_data
from src.models.naive_day.naive_day import get_prob_forecast


class NaiveWeek:
    def __init__(self):
        self.name = "Naive Week"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    @staticmethod
    def forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        up = 1.12
        down = 0.87
        forecast_df = get_point_forecast(forecast_df)
        forecast_df = get_prob_forecast(forecast_df, up, down)
        return forecast_df


def get_point_forecast(forecast_df):
    start_date = forecast_df.at[0, "Date"]
    data = get_data(start_date, start_date+timedelta(days=13), ["Weekday", "Holiday"], os.getcwd(), "h")
    start_train = start_date - timedelta(days=14)
    end_train = start_train + timedelta(days=13)
    df = data_handler.get_data(start_train, end_train, ["System Price", "Weekday", "Holiday"], os.getcwd(), "h")
    df["System Price"] = df.apply(lambda row: row["System Price"] * get_hol_factor() if
    row["Holiday"] == 1 and row["Weekday"] != 7 else row["System Price"], axis=1)
    last_weeks_price = df["System Price"]
    avg_two_last_weeks = []
    weights = [0.25, 0.75]
    for i in range(168):
        avg_two_last_weeks.append(weights[0]*last_weeks_price[i] + weights[1]*last_weeks_price[i+168])
    forecast_df["Forecast"] = avg_two_last_weeks + avg_two_last_weeks
    for i in range(len(forecast_df)):
        if data.loc[i, "Holiday"] == 1 and data.loc[i, "Weekday"] != 7:
            forecast_df.loc[i, "Forecast"] /= get_hol_factor()
    return forecast_df


def get_hol_factor():
    f = 1.1158
    return f


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
    model_ = NaiveWeek()
    periods_ = get_random_periods(1)
    run(model_, periods_)
