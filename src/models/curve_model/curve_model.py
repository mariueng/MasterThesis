from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import numpy as np
import random
from src.system.scores import calculate_coverage_error
from shapely.geometry import LineString
from src.system.generate_periods import get_random_periods
from data.data_handler import get_data
from data.data_handler import get_auction_data
from src.models.curve_model.daily_demand import predict_daily_demand
from src.models.curve_model.hourly_demand import decompose_daily_to_hourly_demand
import matplotlib.pyplot as plt
first_color = "steelblue"
sec_color = "firebrick"


class CurveModel:
    def __init__(self):
        self.name = "CurveModel"
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
        forecast_df = get_prob_forecast(forecast_df)
        return forecast_df


def get_point_forecast(forecast_df):
    start_date = forecast_df.at[0, "Date"]
    end_date = start_date + timedelta(days=13)
    prev_day = start_date - timedelta(days=1)
    yesterday_supply = get_auction_data(prev_day, prev_day, "s", os.getcwd())
    demand_column = "Total Vol"
    window = 180
    hist_demand = get_data(prev_day - timedelta(days=window - 1), prev_day, [demand_column, "Weekday"], os.getcwd(),
                           "d")
    day_demand_forecast = predict_daily_demand(hist_demand, start_date, end_date)
    hourly_demand_forecast = decompose_daily_to_hourly_demand(day_demand_forecast, start_date, end_date)
    for i in range(len(forecast_df)):
        demand = hourly_demand_forecast.loc[i, "Demand Forecast"]
        hour = forecast_df.loc[i, "Hour"]
        supply = yesterday_supply.iloc[hour]
        volumes = [supply[key] for key in supply.keys() if key != "Date" and key != "Hour"]
        prices = [int(key[2:]) for key in supply.keys() if key != "Date" and key != "Hour"]
        supply_line = LineString(np.column_stack((volumes, prices)))
        if demand > max(volumes):
            demand_line = LineString([(max(volumes), -10), (max(volumes), 210)])
        elif demand < min(volumes):
            demand_line = LineString([(min(volumes), -10), (min(volumes), 210)])
        else:
            demand_line = LineString([(demand, -10), (demand, 210)])
        forecast_df.loc[i, "Forecast"] = supply_line.intersection(demand_line).y
    return forecast_df


def get_prob_forecast(forecast_df):
    up = 1.1
    down = 0.90
    forecast_df["Upper"] = forecast_df["Forecast"] * up
    forecast_df["Lower"] = forecast_df["Forecast"] * down
    return forecast_df


def run_model(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        start_date = time_df.at[0, "Date"].date()
        print("Forecasting from {} to {}".format(start_date, start_date + timedelta(days=13)))
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0] - timedelta(days=7), period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_df["Hour Time"] = pd.to_datetime(result_df['Hour'], format="%H").dt.time
        result_df["DateTime"] = result_df.apply(lambda r: dt.combine(r['Date'], r['Hour Time']), 1)
        plt.subplots(figsize=(13, 7))
        plt.plot(result_df["DateTime"], result_df["Forecast"], label="Forecast")
        plt.plot(result_df["DateTime"], result_df["System Price"], label="True")
        plt.legend()
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.title("Expert day")
        plt.tight_layout()
        plt.show()
        result_list.append(result_df)
        print(result_df)


if __name__ == '__main__':
    model_ = CurveModel()
    periods_ = get_random_periods(1)
    periods_ = [(dt(2019,1,22), dt(2019,2,4))]
    run_model(model_, periods_)
