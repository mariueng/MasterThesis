from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import os
import numpy as np
import random
from src.system.scores import calculate_interval_score
from src.system.scores import calculate_coverage_error
from src.system.scores import get_all_point_metrics
from shapely.geometry import LineString
from src.system.generate_periods import get_random_periods
from src.system.generate_periods import get_all_2019_periods
from data.data_handler import get_data
from data.data_handler import get_auction_data
from src.models.curve_model.daily_demand_naive import predict_daily_demand_naive
from src.models.curve_model.supply_curve import get_supply_curve
from src.models.curve_model.supply_curve import get_supply_curve_water_values
from src.models.curve_model.hourly_demand import decompose_daily_to_hourly_demand
from src.models.curve_model.simulate_errors import get_upper_and_lower_bound
import matplotlib.pyplot as plt
import shutil
import pickle
import time


class CurveModel:
    def __init__(self):
        prev_workdir = os.getcwd()
        os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
        self.name = "Curve Model Best"
        today = dt.today()
        self.creation_date = str(today)[0:10] + "_" + str(today)[11:16]
        print("'{}' is instantiated\n".format(self.name))
        plot_paths = ["daily_demand_plots", "hourly_demand_plots", "price_curves", "prices"]
        self.plot = plot
        if self.plot:
            for p in plot_paths:
                if os.path.exists(p):
                    shutil.rmtree(p)  # delete old
                os.makedirs(p)  # create new
        os.chdir(prev_workdir)

    def get_name(self):
        return self.name

    def get_time(self):
        return self.creation_date

    @staticmethod
    def forecast(forecast_df):  # forecast_df is dataframe with ["Date", "Hour", "Forecast", "Upper", "Lower"]
        forecast_df = get_forecast(forecast_df)
        return forecast_df


d_classes = [-10, 0, 1, 5, 11, 20, 32, 46, 75, 107, 195, 210]
s_classes = [-10, -4, -1, 0, 1, 3, 5, 8, 12, 15, 19, 22, 24, 26, 28, 30, 32, 35, 39, 42, 46, 51, 56, 66, 75, 105, 165, 210]
d_color = plt.get_cmap("tab10")(0)
s_color = plt.get_cmap("tab10")(1)
first_color = "steelblue"
sec_color = "firebrick"
label_pad = 12
title_pad = 20
full_fig = (13, 7)
plot = False


def get_forecast(forecast_df):
    prev_workdir = os.getcwd()
    os.chdir("\\".join(prev_workdir.split("\\")[:6]) + "\\models\\curve_model")
    start_date = forecast_df.at[0, "Date"]
    end_date = start_date + timedelta(days=13)
    true_supply = get_auction_data(start_date, end_date, "s", os.getcwd())
    help_data = get_data(start_date-timedelta(days=7), end_date, ["Season", "Total Hydro Dev", "Coal"], os.getcwd(), "d")
    last_week_coal = help_data.head(7)["Coal"].mean()
    if plot:
        if not os.path.exists("price_curves/curves_{}_{}".format(start_date.date(), end_date.date())):
            os.makedirs("price_curves/curves_{}_{}".format(start_date.date(), end_date.date()))
            curve_data = get_auction_data(start_date, end_date, ["s", "d"], os.getcwd())  # only used for plotting
    data, h_demand_forecast, supply_mean_week, wv_table = get_data_demand_and_supply(start_date, end_date)
    wv_model = pickle.load(open("wv_model.pickle", 'rb'))
    for i in range(len(forecast_df)):
        demand = h_demand_forecast.loc[i, "Demand Forecast"]
        hour = forecast_df.loc[i, "Hour"]
        month = data.loc[i, "Month"]
        weekend = data.loc[i, "Weekend"]
        mean_supply_curve = supply_mean_week.iloc[hour]
        last_week_row = help_data[help_data["Date"] == forecast_df.loc[i, "Date"]-timedelta(days=7)]
        help_row = help_data[help_data["Date"] == forecast_df.loc[i, "Date"]]
        #safe_supply = get_supply_curve(month, hour, weekend, mean_supply_curve, safe=True) # VERSION 0
        safe_supply = get_supply_curve_water_values(month, hour, weekend, mean_supply_curve, help_row, last_week_row, wv_model, last_week_coal, safe=True) # VERSION 1
        #safe_supply = true_supply.iloc[i][2:]
        volumes = safe_supply.values
        max_volume, min_volume = (max(volumes), min(volumes))
        supply_line = LineString(np.column_stack((volumes, s_classes)))
        demand = max_volume if demand > max_volume else demand
        demand = min_volume if demand < min_volume else demand
        demand_line = LineString([(demand, -10), (demand, 210)])
        if plot:
            curve_hour = curve_data.iloc[i]
            plot_true_and_estimated_curves(curve_hour, demand_line, supply_line, start_date, end_date)
        point_forecast = supply_line.intersection(demand_line).y
        forecast_df.loc[i, "Forecast"] = point_forecast
        upper, lower = (point_forecast * 1.2, point_forecast * 0.8)
        # upper, lower = get_prob_forecast(0.95, demand, 1000, month, hour, weekend, mean_supply_curve, i, point_forecast)
        forecast_df.loc[i, "Upper"] = upper
        forecast_df.loc[i, "Lower"] = lower
    os.chdir(prev_workdir)
    return forecast_df


def get_data_demand_and_supply(start_date, end_date):
    data = get_data(start_date, end_date, ["Month", "Weekend"], os.getcwd(), "h")
    trans_table = pd.read_csv("demand_temp_transitions.csv")
    day_demand_forecast = predict_daily_demand_naive(start_date, end_date, trans_table, plot)
    hourly_demand_forecast = decompose_daily_to_hourly_demand(day_demand_forecast, start_date, end_date, plot)
    last_week_supply = get_auction_data(start_date-timedelta(days=7), start_date-timedelta(days=1), "s", os.getcwd())
    supply_mean_week = last_week_supply.groupby(by="Hour").mean().reset_index()
    trans_table_water_values = pd.read_csv("water_value_hydro_dev_profiles_2.csv")
    return data, hourly_demand_forecast, supply_mean_week, trans_table_water_values


def get_prob_forecast(alpha, demand, iterations, month, hour, weekend, mean_supply_curve, i, point_forecast):
    d_errors = pd.read_csv("demand_errors.csv")
    demand_errors = d_errors["Day {}".format((i // 24) + 1)].values
    p_errors = pd.read_csv("price_errors.csv")
    price_errors = p_errors["Day {}".format((i // 24) + 1)].values
    prob_supply = get_supply_curve(month, hour, weekend, mean_supply_curve, safe=False)
    prob_supply_line = LineString(np.column_stack((prob_supply.values, s_classes)))
    upper, lower = get_upper_and_lower_bound(alpha, demand, iterations, demand_errors, prob_supply_line, price_errors,
                                             point_forecast)
    return upper, lower


def plot_true_and_estimated_curves(curve_hour, d_line, s_line, s_date, e_date):
    date = curve_hour["Date"].date()
    hour = curve_hour["Hour"]
    demand_volumes = curve_hour.iloc[2:14].values
    supply_volumes = curve_hour.iloc[14:len(curve_hour)].values
    plt.subplots(figsize=full_fig)
    plt.plot(demand_volumes, d_classes, linestyle="dotted", color=d_color, label="True demand")
    plt.plot(supply_volumes, s_classes, linestyle="dotted", color=s_color, label="True supply")
    plt.plot(d_line.coords.xy[0], d_line.coords.xy[1], color=d_color, label="Est demand")
    plt.plot(s_line.coords.xy[0], s_line.coords.xy[1], color=s_color, label="Est supply")
    for line in plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03), fancybox=True,
                           shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title("Curve Forecast {} - {}".format(date, hour), pad=title_pad)
    plt.xlabel("Date", labelpad=label_pad)
    plt.ylabel("Volume [MWh]", labelpad=label_pad)
    plt.tight_layout()
    plt.savefig("price_curves/curves_{}_{}/{}_{}.png".format(s_date.date(), e_date.date(), date, hour))
    plt.close()


def run_model(model, periods):
    result_list = []
    for period in periods:
        time_df = get_data(period[0], period[1], [], os.getcwd(), "h")
        start_date = time_df.at[0, "Date"].date()
        end_date = start_date + timedelta(days=13)
        print("Forecasting from {} to {}".format(start_date, end_date))
        time_df["Forecast"] = np.nan
        time_df["Upper"] = np.nan
        time_df["Lower"] = np.nan
        forecast_df = model.forecast(time_df)
        true_price_df = get_data(period[0] - timedelta(days=7), period[1], ["System Price"], os.getcwd(), "h")
        result_df = true_price_df.merge(forecast_df, on=["Date", "Hour"], how="outer")
        result_df["Hour Time"] = pd.to_datetime(result_df['Hour'], format="%H").dt.time
        result_df["DateTime"] = result_df.apply(lambda r: dt.combine(r['Date'], r['Hour Time']), 1)
        plt.subplots(figsize=full_fig)
        plt.plot(result_df["DateTime"], result_df["System Price"], label="True", color=first_color)
        plt.plot(result_df["DateTime"], result_df["Forecast"], label="Forecast", color=sec_color)
        plt.legend()
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.title("Curve Model {} to {}".format(start_date, end_date))
        plt.tight_layout()
        plt.savefig("prices/{}_{}.png".format(start_date, end_date))
        result_df = result_df.dropna()
        result_list.append(result_df)
    result = pd.concat(result_list).reset_index(drop=True)
    print(result[["Date", "System Price", "Forecast"]])
    print("-----------------\n")
    pm = get_all_point_metrics(result)
    print("Mape {:.2f}, smape {:.2f}, mae {:.2f}, rmse {:.2f}".format(pm["mape"], pm["smape"], pm["mae"], pm["rmse"]))
    print("IS Score: {:.2f}".format(calculate_interval_score(result)))
    err = calculate_coverage_error(result)
    print("Cov Error Score: {:.2f}, mean coverage {:.2f}%".format(err[0], err[1]))


def explore_supply_outlier():
    if os.path.exists("supply_yesterday_vs_week.csv"):
        result = pd.read_csv("supply_yesterday_vs_week.csv")
    else:
        periods = get_all_2019_periods()
        result = pd.DataFrame(columns=["Period", "Yesterday MAE", "Week MAE"])
        for i in range(len(periods)):
            p = periods[i]
            start_date = p[0]
            end_date = start_date + timedelta(days=13)
            prev_day = start_date - timedelta(days=1)
            yesterday_supply = get_auction_data(prev_day, prev_day, "s", os.getcwd())
            last_week_supply = get_auction_data(prev_day - timedelta(days=6), prev_day, "s", os.getcwd())
            supply_mean = last_week_supply.groupby(by="Hour").mean().reset_index()
            true_supply = get_auction_data(start_date, end_date, "s", os.getcwd())
            yesterday_aes = []
            weekly_aes = []
            print(i)
            for j in range(len(true_supply)):
                true = true_supply.iloc[j]
                hour = true["Hour"]
                yesterday = yesterday_supply[yesterday_supply["Hour"] == hour].iloc[0]
                week = supply_mean[supply_mean["Hour"] == hour].iloc[0]
                yest_aes = [abs(yesterday[i] - true[i]) for i in range(2, len(true))]
                yesterday_aes.append(round(sum(yest_aes) / len(yest_aes), 2))
                week_aes = [abs(week[i-1] - true[i]) for i in range(2, len(true))]
                weekly_aes.append(round(sum(week_aes) / len(week_aes), 2))
            row = {"Period": i+1, "Yesterday MAE": sum(yesterday_aes)/len(yesterday_aes), "Week MAE": sum(weekly_aes) /
                   len(weekly_aes)}
            result = result.append(row, ignore_index=True)
        result["Weekly is best"] = np.where(result["Week MAE"] < result["Yesterday MAE"], True, False)
        result.to_csv("supply_yesterday_vs_week.csv", index=False, float_format="%g")
    week_is_best = result[result["Weekly is best"] == True]
    print("Week is best in {:.2f}% of the cases".format(100*len(week_is_best)/len(result)))


def explore_demand_period():
    start = dt(2019, 6, 12)
    end = dt(2019, 6, 25)
    hist = get_data(start-timedelta(days=180), start-timedelta(days=1), ["Curve Demand"], os.getcwd(), "d")
    hist_and_period = get_data(start-timedelta(days=180), end, ["Curve Demand"], os.getcwd(), "d")
    plt.subplots(figsize=full_fig)
    plt.plot(hist_and_period["Date"], hist_and_period["Curve Demand"], color=sec_color)
    plt.plot(hist["Date"], hist["Curve Demand"], color=first_color)
    plt.show()
    plt.close()
    plt.subplots(figsize=full_fig)
    dec_hist = get_decomp(hist)
    dec_hist_per = get_decomp(hist_and_period)
    plt.plot(dec_hist_per["Date"], dec_hist_per["Trend"], color=sec_color)
    plt.plot(dec_hist["Date"], dec_hist["Trend"], color=first_color)
    plt.show()
    plt.close()


def get_decomp(df):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(df["Curve Demand"], model='multiplicative', period=7)
    df["Factor"] = decomp.seasonal
    df["Trend"] = decomp.trend
    df["Trend"] = df["Trend"].interpolate()
    df['Trend'] = df.apply(
        lambda row: row["Curve Demand"] / row['Factor'] if np.isnan(row['Trend']) else row['Trend'], axis=1)
    return df


def explore_temp_and_price():
    from statsmodels.tsa.seasonal import seasonal_decompose
    start = dt(2019, 1, 25)
    end = dt(2019, 2, 7)
    window = 180
    prev_day = start - timedelta(days=1)
    columns = ["System Price", "Curve Demand", "Temp Norway"]
    hist = get_data(start - timedelta(days=window), prev_day, columns, os.getcwd(), "d")
    hist_hor = get_data(start - timedelta(days=window), end, columns, os.getcwd(), "d")
    for col in columns:
        plt.subplots(figsize=full_fig)
        hist_hor["Trend"] = seasonal_decompose(hist_hor[col], model='a', period=7, extrapolate_trend='freq').trend
        hist_hor["Shift"] = hist_hor["Trend"] - hist_hor["Trend"].shift(1)
        plt.plot(hist_hor["Date"], hist_hor["Shift"], label="Horizon {}".format(col))
        hist["Trend"] = seasonal_decompose(hist[col], model='a', period=7, extrapolate_trend='freq').trend
        hist["Shift"] = hist["Trend"] - hist["Trend"].shift(1)
        plt.plot(hist["Date"], hist["Shift"], label="Hist {}".format(col))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    # explore_supply_outlier()
    # explore_demand_period()
    # explore_temp_and_price()
    model_ = CurveModel()
    periods_ = get_random_periods(1)
    periods_ = [(dt(2019, 7, 7), dt(2019, 7, 20))]
    run_model(model_, periods_)
