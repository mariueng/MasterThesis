# script for the most naive model: copy the 24 hours of last day for the following 14 days
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


class NaiveDay:
    def __init__(self):
        self.name = "Naive Day"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    @staticmethod
    def forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = get_point_forecast(forecast_df)
        up = 1.15
        down = 0.90
        forecast_df = get_prob_forecast(forecast_df,  up, down)
        return forecast_df


def get_point_forecast(forecast_df):
    start_date = forecast_df.at[0, "Date"]
    prev_day = start_date - timedelta(days=1)
    prev_day_string = prev_day.strftime("%d.%m.%Y")
    prev_day_price = data_handler.get_data(prev_day_string, prev_day_string, ["System Price"], os.getcwd(), "h")
    hour_price_df = prev_day_price[["Hour", "System Price"]]
    hour_price_dict = pd.Series(hour_price_df["System Price"].values, index=hour_price_df["Hour"]).to_dict()
    for index, row in forecast_df.iterrows():
        hour_of_day = row["Hour"]
        forecast_hour = hour_price_dict[hour_of_day]
        forecast_df.at[index, "Forecast"] = forecast_hour
    return forecast_df


def get_prob_forecast(forecast_df, up, down):
    for index, row in forecast_df.iterrows():
        point_f = row["Forecast"]
        factor_up = get_factor_up(index, len(forecast_df), up)
        upper_f = point_f * factor_up
        forecast_df.at[index, "Upper"] = upper_f
        factor_down = get_factor_down(index, len(forecast_df), down)
        lower_f = point_f * factor_down
        forecast_df.at[index, "Lower"] = lower_f
    return forecast_df


def get_factor_up(index, horizon, up_factor):
    return up_factor + ((index / horizon) / 2) * up_factor


def get_factor_down(index, horizon, down_factor):
    return down_factor - ((index / horizon) / 2) * down_factor


def find_best_up_and_down_factor(model, periods):
    up_factors = [round(1.04 + i*0.02, 3) for i in range(6)]
    down_factors = [round(0.96 - i*0.02, 3) for i in range(6)]
    aces = {}
    for up in up_factors:
        for down in down_factors:
            print("Testing for {} and {}".format(up, down))
            results = get_results(up, down, model, periods)
            scores = []
            for result in results:
                cov, a = calculate_coverage_error(result)
                scores.append(a)
            ace = sum(scores)/len(scores)
            aces["{}, {}".format(up, down)] = ace
            print("ACE: " + str(ace))
    print("Min ACE up and down: " + str(min(aces, key=aces.get)))


def get_results(up, down, model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df, up, down)
        true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_list.append(result_df)
    return result_list


if __name__ == '__main__':
    model = NaiveDay()
    periods_ = get_random_periods(20)
    find_best_up_and_down_factor(model, periods_)