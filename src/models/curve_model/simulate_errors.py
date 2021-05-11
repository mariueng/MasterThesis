from src.system.generate_periods import get_random_periods
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import numpy as np
from src.models.curve_model.daily_demand_naive import predict_daily_demand_naive
from src.models.curve_model.hourly_demand import decompose_daily_to_hourly_demand
import random
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


def estimate_hourly_demand_errors():
    periods = get_all_2019_periods()
    columns = ["Day {}".format(i) for i in range(1, 15)]
    df = pd.DataFrame(columns=columns)
    for i in range(len(periods)):
        p_df = pd.DataFrame(columns=df.columns)
        period = periods[i]
        start = period[0]
        print("Running from {}".format(start))
        end = period[1]
        trans_table = pd.read_csv("demand_temp_transitions.csv")
        day_demand_forecast = predict_daily_demand_naive(start, end, trans_table)
        hourly_demand_forecast = decompose_daily_to_hourly_demand(day_demand_forecast, start, end, plot=False)
        for j in range(14):
            start_idx = j*24
            day_j = hourly_demand_forecast.iloc[start_idx:start_idx+24, :].reset_index(drop=True)
            p_df["Day {}".format(j+1)] = day_j["Curve Demand"] - day_j["Demand Forecast"]
        df = df.append(p_df, ignore_index=True)
    df.to_csv("demand_errors.csv", index=False, float_format="%g")
    print(df)
    print("-----------------------\n")
    for col in df.columns:
        print("{} mean {:.2f} MWh".format(col, abs(df[col]).mean()))


def get_upper_and_lower_bound(alpha, demand, n, error_series, supply_line, price_errors, point_forecast):
    max_vol = max(supply_line.coords.xy[0])
    min_vol = min(supply_line.coords.xy[0])
    upper_index = int(n * alpha)
    lower_index = int(n * (1-alpha))
    chosen_price_errors = np.random.choice(price_errors, n)
    chosen_price_errors.sort()
    min_upper = point_forecast + chosen_price_errors[upper_index]
    max_lower = point_forecast + chosen_price_errors[lower_index]
    chosen_errors = np.random.choice(error_series, n)
    chosen_errors.sort()
    upper_demand = demand + chosen_errors[upper_index]
    lower_demand = demand + chosen_errors[lower_index]
    if upper_demand > max_vol:
        upper_demand = max_vol
    if lower_demand < min_vol:
        lower_demand = min_vol
    u_d_line = LineString([(upper_demand, -10), (upper_demand, 210)])
    l_d_line = LineString([(lower_demand, -10), (lower_demand, 210)])
    upper_inter = supply_line.intersection(u_d_line)
    if type(upper_inter) is MultiPoint:
        print(supply_line.coords.xy[0])
        print(supply_line.coords.xy[1])
        print(u_d_line)
    lower_inter = supply_line.intersection(l_d_line)
    if type(lower_inter) is MultiPoint:
        print(supply_line.coords.xy[0])
        print(supply_line.coords.xy[1])
        print(l_d_line)
    upper_bound = max(min_upper, upper_inter.y)
    lower_bound = min(max_lower, lower_inter.y)

    return upper_bound, lower_bound


def estimate_daily_price_errors():
    from src.models.curve_model.curve_model import CurveModel
    periods = get_random_periods(2)
    periods = get_all_2019_periods()
    columns = ["Day {}".format(i) for i in range(1, 15)]
    df = pd.DataFrame(columns=columns)
    model = CurveModel()
    for i in range(len(periods)):
        p_df = pd.DataFrame(columns=df.columns)
        period = periods[i]
        print("Running from {}".format(period[0]))
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0], period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        for j in range(14):
            day_j_forecast = result_df.iloc[j*24:j*24+24, :]
            errors = day_j_forecast["System Price"] - day_j_forecast["Forecast"]
            p_df["Day {}".format(1+j)] = errors.reset_index(drop=True)
        df = df.append(p_df, ignore_index=True)
    df = df.round(2)
    df.to_csv("price_errors.csv", index=False, float_format="%g")
    print("-----------------------\n")
    for col in df.columns:
        print("{} mean {:.2f} Euro".format(col, abs(df[col]).mean()))


if __name__ == '__main__':
    print("Running methods")
    # estimate_hourly_demand_errors()
    estimate_daily_price_errors()
