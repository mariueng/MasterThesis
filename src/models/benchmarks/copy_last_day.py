# script for the most naive model: copy the 24 hours of last day for the following 14 days
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from data import data_handler
import os
import numpy as np
import random


class CopyLastDayModel:
    def __init__(self):
        self.name = "Copy Last Day"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    @staticmethod
    def get_point_forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast"]
        start_date = forecast_df.at[0, "Date"]
        prev_day = start_date - timedelta(days=1)
        prev_day_string = prev_day.strftime("%d.%m.%Y")
        prev_day_price = data_handler.get_data(prev_day_string, prev_day_string, ["System Price"], os.getcwd())
        hour_price_df = prev_day_price[["Hour", "System Price"]]
        hour_price_dict = pd.Series(hour_price_df["System Price"].values, index=hour_price_df["Hour"]).to_dict()
        for index, row in forecast_df.iterrows():
            hour_of_day = row["Hour"]
            forecast_hour = hour_price_dict[hour_of_day]
            forecast_df.at[index, "Forecast"] = forecast_hour
        return forecast_df

    @staticmethod
    def get_prob_forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        for index, row in forecast_df.iterrows():
            point_f = row["Forecast"]
            random.seed(1)
            random_factor_up = random.randint(105, 120)
            upper_f = point_f * (random_factor_up / 100)
            forecast_df.at[index, "Upper"] = upper_f
            random_factor_down = random.randint(80, 95)
            lower_f = point_f * (random_factor_down / 100)
            forecast_df.at[index, "Lower"] = lower_f
        return forecast_df

if __name__ == '__main__':
    model = CopyLastDayModel()
    start_date_ = "04.02.2019"
    end_date_ = "17.02.2019"
    time_list_ = data_handler.get_data(start_date_, end_date_, [], os.getcwd())
    time_list_["Forecast"] = np.nan
    forecast_ = model.get_point_forecast(time_list_)
    print(forecast_)
