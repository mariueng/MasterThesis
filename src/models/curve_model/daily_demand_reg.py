from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from data.data_handler import get_data
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from src.system.generate_periods import get_all_2019_periods
from src.system.generate_periods import get_random_periods
random.seed(1)
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)


def get_poly_stuff(hist_df, reg_window):
    df = hist_df.tail(reg_window)
    df.index = np.arange(1, len(df)+1)
    df_x = df.reset_index()[["index", "Temp Norway"]]
    x_train = df_x.values
    pol_reg = LinearRegression()
    pol_reg.fit(x_train, df["Trend"])
    lin_reg = LinearRegression().fit(df_x[["index"]].values, df["Trend"])
    lin_trend = lin_reg.predict(df_x[["index"]].values)
    plot = False
    if plot:
        plt.subplots(figsize=full_fig)
        plt.plot(df["Date"], df["Curve Demand"], label="True Demand")
        plt.plot(df["Date"], df["Trend"], label="True trend")
        plt.plot(df["Date"], lin_trend, label="Lin trend")
        plt.plot(df["Date"], pol_reg.predict(x_train), label="Forecast trend")
        plt.legend()
        plt.show()
        plt.close()
    return pol_reg



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


def predict_daily_demand_reg(start, end):
    reg_window = 40
    ses_window = 60
    ses_start = start-timedelta(days=ses_window)
    ses_end = start-timedelta(days=1)
    reg_hist_df = get_data(ses_start, ses_end, ["Curve Demand", "Weekday", "Temp Norway"], os.getcwd(), "d")
    hist_df = get_decomp_new(reg_hist_df)
    pol_reg = get_poly_stuff(hist_df, reg_window)
    test_data = get_data(start, end, ["Temp Norway"], os.getcwd(), "d")
    test_data.index = np.arange(reg_window, reg_window + 14)
    x_test = test_data.reset_index()[["index", "Temp Norway"]].values
    trend_forecast = pol_reg.predict(x_test)
    result_df = get_data(start, end, ["Weekday"], os.getcwd(), "d")
    result_df["Trend Forecast"] = trend_forecast
    result_df["Factor"] = hist_df["Factor"].tail(14).reset_index(drop=True)
    result_df["Demand Forecast"] = result_df["Trend Forecast"] * result_df["Factor"]
    plot = False
    if plot:
        plt.subplots(figsize=full_fig)
        plt.plot(result_df["Date"], result_df["Trend Forecast"], label="Trend forecast")
        plt.plot(result_df["Date"], result_df["Demand Forecast"], label="Demand forecast")
        true = get_data(start, end, ["Curve Demand", "Temp Norway"], os.getcwd(), "d")
        plt.plot(result_df["Date"], true["Curve Demand"], label="True demand")
        plt.legend()
        plt.show()
        plt.close()
        plt.subplots(figsize=full_fig)
        plt.plot(reg_hist_df["Date"], reg_hist_df["Temp Norway"], label="Temp Norway train")
        plt.plot(true["Date"], true["Temp Norway"], label="Temp Norway test")
        plt.legend()
        plt.show()
    result_df = result_df[["Date", "Trend Forecast", "Demand Forecast"]]
    return result_df


def evaluate_reg_demand():
    periods = get_random_periods(30)
    periods = get_all_2019_periods()
    df = pd.DataFrame(columns=["Period", "Date", "Demand Forecast", "Curve Demand"])
    result = pd.DataFrame(columns=["Period", "Start", "End", "MAE", "MAPE"])
    for i in range(len(periods)):
        p = periods[i]
        start = p[0]
        end = p[1]
        p_df = pd.DataFrame(columns=df.columns)
        forecast_df = predict_daily_demand_reg(start, end)
        p_df["Date"] = forecast_df["Date"]
        p_df["Demand Forecast"] = forecast_df["Demand Forecast"]
        p_df["Trend Forecast"] = forecast_df["Trend Forecast"]
        p_df["Curve Demand"] = get_data(start, end, ["Curve Demand"], os.getcwd(), "d")["Curve Demand"]
        p_df["Period"] = int(i+1)
        df = df.append(p_df[["Period", "Date", "Demand Forecast", "Curve Demand"]], ignore_index=True)
        ae = (abs(p_df["Curve Demand"] - p_df["Demand Forecast"])).mean()
        ape = 100 * ae / p_df["Curve Demand"].mean()
        result = result.append({"Period": int(i+1), "Start": start, "End": end, "MAE": ae, "MAPE": ape}, ignore_index=True)
        plot_period = False
        if plot_period:
            plot_evaluaion_period(p_df)
    print(result)
    print("-----------------------")
    print("MAE {:.2f}, MAPE {:.2f}".format(result["MAE"].mean(), result["MAPE"].mean()))
    print(result.sort_values(by='MAPE', ascending=False).head(3))


def plot_evaluaion_period():
    print("hi")


def eda_temperature_and_demand():
    start = dt(2018, 12, 22).date()
    end = dt(2019, 2, 22).date()
    data = get_data(start, end, ["Curve Demand", "Temp Norway"], os.getcwd(), "d")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(13, 12))
    fig.suptitle("Demand and Temperature {} - {}".format(start, end))
    ax1.plot(data["Date"], data["Curve Demand"], label="Demand", color=first_color)
    ax2.plot(data["Date"], data["Temp Norway"], label="Temp Norway", color=sec_color)
    for ax in [ax1, ax2]:
        for line in ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.06), fancybox=True,
                              shadow=True).get_lines():
            line.set_linewidth(2)
    plt.show()


if __name__ == '__main__':
    # eda_temperature_and_demand()
    # test_poly_regression()
    evaluate_reg_demand()
