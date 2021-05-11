from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from data.data_handler import get_data
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from src.system.generate_periods import get_all_2019_periods
from src.system.generate_periods import get_random_periods
import warnings
random.seed(1)
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)


def get_decomp_new(hist):
    hist = hist.reset_index(drop=True)
    demand_column = "Curve Demand"
    decomp = seasonal_decompose(hist[demand_column], model='multiplicative', period=7)
    hist["Factor"] = decomp.seasonal
    hist["Trend"] = decomp.trend
    hist["Adj Trend"] = hist["Curve Demand"] / hist["Factor"]
    hist.loc[0, "Trend"] = sum(hist["Adj Trend"].head(3) * np.asarray([0.2, 0.3, 0.5]))
    hist.loc[len(hist)-1, "Trend"] = sum(hist["Adj Trend"].tail(3) * np.asarray([0.2, 0.3, 0.5]))
    hist["Trend"] = hist["Trend"].interpolate(method='quadratic', limit_area="inside")
    hist = hist.drop(columns=["Adj Trend"])
    return hist


def predict_daily_demand_ets(start, end):
    warnings.filterwarnings("ignore")
    window = 80
    ses_start = start-timedelta(days=window)
    ses_end = start-timedelta(days=1)
    reg_hist_df = get_data(ses_start, ses_end, ["Curve Demand", "Weekday", "Temp Norway"], os.getcwd(), "d")
    hist_df = get_decomp_new(reg_hist_df)
    trend = hist_df[["Trend"]].values
    temp = reg_hist_df[["Temp Norway"]].values
    reg = LinearRegression()
    reg.fit(temp, trend)
    train_predict = reg.predict(temp)
    errors = trend - train_predict
    ets = ExponentialSmoothing(errors, damped_trend=True, trend="add", initialization_method="estimated")
    model_fit = ets.fit(optimized=True, remove_bias=True)
    predicted_errors = model_fit.predict(start=len(trend), end=len(trend) + 13)
    true_temp = get_data(start, end, ["Temp Norway"], os.getcwd(), "d")[["Temp Norway"]].values
    reg_predict = reg.predict(true_temp)[:, 0]
    result_df = get_data(start, end, ["Weekday"], os.getcwd(), "d")
    result_df["Trend Forecast"] = reg_predict + predicted_errors
    result_df["Factor"] = hist_df["Factor"].tail(14).reset_index(drop=True)
    result_df["Demand Forecast"] = result_df["Trend Forecast"] * result_df["Factor"]
    plot = True
    if plot:
        plt.subplots(figsize=full_fig)
        plt.plot(result_df["Date"], result_df["Trend Forecast"], label="Trend forecast")
        plt.plot(result_df["Date"], result_df["Demand Forecast"], label="Demand forecast")
        true = get_data(start, end, ["Curve Demand", "Temp Norway"], os.getcwd(), "d")
        plt.plot(result_df["Date"], true["Curve Demand"], label="True demand")
        plt.plot(pd.date_range(start-timedelta(days=window), start-timedelta(days=1)), trend, label="Trend")
        plt.legend()
        plt.show()
        plt.close()
        plt.subplots(figsize=full_fig)
        plt.plot(reg_hist_df["Date"], reg_hist_df["Temp Norway"], label="Temp Norway train")
        plt.plot(true["Date"], true["Temp Norway"], label="Temp Norway test")
        plt.legend()
        plt.show()
        assert False
    result_df = result_df[["Date", "Trend Forecast", "Demand Forecast"]]
    return result_df


def evaluate_ets_demand():
    periods = get_random_periods(30)
    periods = get_all_2019_periods()
    periods = [(dt(2019, 5, 14).date(), dt(2019, 5, 27).date())]
    df = pd.DataFrame(columns=["Period", "Date", "Demand Forecast", "Curve Demand"])
    result = pd.DataFrame(columns=["Period", "Start", "End", "MAE", "MAPE"])
    for i in range(len(periods)):
        p = periods[i]
        start = p[0]
        end = p[1]
        p_df = pd.DataFrame(columns=df.columns)
        forecast_df = predict_daily_demand_ets(start, end)
        p_df["Date"] = forecast_df["Date"]
        p_df["Demand Forecast"] = forecast_df["Demand Forecast"]
        p_df["Trend Forecast"] = forecast_df["Trend Forecast"]
        p_df["Curve Demand"] = get_data(start, end, ["Curve Demand"], os.getcwd(), "d")["Curve Demand"]
        p_df["Period"] = int(i+1)
        df = df.append(p_df[["Period", "Date", "Demand Forecast", "Curve Demand"]], ignore_index=True)
        ae = (abs(p_df["Curve Demand"] - p_df["Demand Forecast"])).mean()
        ape = 100 * ae / p_df["Curve Demand"].mean()
        result = result.append({"Period": int(i+1), "Start": start, "End": end, "MAE": ae, "MAPE": ape}, ignore_index=True)
    print(result)
    print("-----------------------")
    print("MAE {:.2f}, MAPE {:.2f}".format(result["MAE"].mean(), result["MAPE"].mean()))
    print(result.sort_values(by='MAPE', ascending=False).head(3))


if __name__ == '__main__':
    evaluate_ets_demand()
